"""Train one ResT Siamese session on the GPDS/GDPS subset with sampled pairs.

Default is intentionally light for a local GPU:

    120 train writers, 30 test writers
    8 balanced sampled pairs per train writer per epoch
    64 balanced sampled pairs per test writer per evaluation
    5 epochs

The script prints live progress to CMD/PowerShell: epoch, step, loss, speed,
ETA, and peak GPU memory. It saves the best checkpoint by test accuracy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import re
import time

import torch
from torch.utils.data import DataLoader

from rest_siamese import ContrastiveLoss, ResTConfig, ResTEncoder, SiameseResT
from train_cedar import (
    CedarPairDataset,
    atomic_torch_save,
    evaluate,
    load_vit_b16_imagenet21k,
    seed_everything,
)


GPDS_PATTERN = re.compile(r"^(cf|c)-(\d+)-(\d+)(?: \(Copy\))?\.[^.]+$", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpds-root", type=Path, required=True)
    parser.add_argument("--vit-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-writers", type=int, default=120)
    parser.add_argument("--pairs-per-writer", type=int, default=8)
    parser.add_argument("--eval-pairs-per-writer", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--session-id", type=int, default=1)
    parser.add_argument("--random-erasing", action="store_true")
    parser.add_argument("--log-every", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument(
        "--process-log",
        type=Path,
        default=None,
        help="Optional JSONL progress log.",
    )
    return parser.parse_args()


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def append_process_log(log_path: Path | None, event: str, payload: dict) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": event,
        "timestamp_unix": time.time(),
        **payload,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def scan_gpds(root: Path) -> dict[str, dict[str, list[Path]]]:
    """Scan this GPDS archive layout and deduplicate copied train/test files."""

    if not root.exists():
        raise FileNotFoundError(f"GPDS root does not exist: {root}")

    writers: dict[str, dict[str, list[Path]]] = {}
    seen: set[tuple[str, str, int]] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        match = GPDS_PATTERN.match(path.name)
        if not match:
            continue
        prefix, writer_text, sample_text = match.groups()
        writer = f"gpds:{int(writer_text):03d}"
        kind = "forged" if prefix.lower() == "cf" else "genuine"
        sample = int(sample_text)
        key = (writer, kind, sample)
        if key in seen:
            continue
        seen.add(key)
        writers.setdefault(writer, {"genuine": [], "forged": []})[kind].append(path)

    if len(writers) != 150:
        raise RuntimeError(f"Expected 150 GPDS writers, found {len(writers)} at {root}")

    for writer, samples in writers.items():
        samples["genuine"].sort()
        samples["forged"].sort()
        if len(samples["genuine"]) != 24 or len(samples["forged"]) != 30:
            raise RuntimeError(
                f"{writer}: expected 24 genuine/30 forged, found "
                f"{len(samples['genuine'])}/{len(samples['forged'])}"
            )
    return writers


def split_writers(
    writer_ids: list[str], train_count: int, seed: int
) -> tuple[list[str], list[str]]:
    if train_count <= 0 or train_count >= len(writer_ids):
        raise ValueError("--train-writers must be between 1 and 149")
    shuffled = sorted(writer_ids)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:train_count]), sorted(shuffled[train_count:])


def make_loader(dataset: CedarPairDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this ResT ViT-B/16 training script.")

    if args.process_log is not None:
        args.process_log.parent.mkdir(parents=True, exist_ok=True)
        args.process_log.write_text("", encoding="utf-8")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    writers = scan_gpds(args.gpds_root)
    train_writers, test_writers = split_writers(
        list(writers), train_count=args.train_writers, seed=args.seed
    )

    train_dataset = CedarPairDataset(
        writers,
        train_writers,
        pairs_per_writer=args.pairs_per_writer,
        training=True,
        seed=args.seed,
        random_erasing=args.random_erasing,
    )
    test_dataset = CedarPairDataset(
        writers,
        test_writers,
        pairs_per_writer=args.eval_pairs_per_writer,
        training=False,
        seed=args.seed + 100_000,
    )
    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers)
    test_loader = make_loader(test_dataset, args.batch_size, args.num_workers)

    config = ResTConfig(gradient_checkpointing=True)
    encoder = ResTEncoder(config)
    initialization = {
        "source": "Google ViT-B/16 ImageNet-21k",
        "path": str(args.vit_npz.resolve()),
        **load_vit_b16_imagenet21k(encoder, args.vit_npz),
    }
    model = SiameseResT(encoder).to(device)
    criterion = ContrastiveLoss(margin=1.0, positive_weight=1.0, negative_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    train_pairs_per_epoch = len(train_dataset)
    eval_pairs_per_epoch = len(test_dataset)
    planned_total_pairs = args.epochs * (train_pairs_per_epoch + eval_pairs_per_epoch)
    setup = {
        "script": Path(__file__).name,
        "session_id": args.session_id,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "parameters": parameter_count,
        "gpds_root": str(args.gpds_root.resolve()),
        "writers_found": len(writers),
        "train_writers_count": len(train_writers),
        "test_writers_count": len(test_writers),
        "train_pairs_per_epoch": train_pairs_per_epoch,
        "eval_pairs_per_epoch": eval_pairs_per_epoch,
        "planned_train_pairs_total": train_pairs_per_epoch * args.epochs,
        "planned_eval_pairs_total": eval_pairs_per_epoch * args.epochs,
        "planned_total_pairs_train_plus_eval": planned_total_pairs,
        "pairs_per_training_writer_per_epoch": args.pairs_per_writer,
        "pairs_per_test_writer": args.eval_pairs_per_writer,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "random_erasing": args.random_erasing,
        "augmentation": [
            "Gaussian blur + Otsu + inversion + centering",
            "training only: random rotation [-5,+5] degrees",
            "training only: horizontal flip p=0.5",
            "training only: random 224x224 crop from 256x256",
            "optional RandomErasing if --random-erasing is used",
        ],
        "normalization": [
            "ImageNet input mean/std",
            "BatchNorm in shared Conv, SPD-Conv, and ResNet-18",
            "GroupNorm in SCConv SRU",
            "LayerNorm in ViT",
        ],
        "initialization": initialization,
    }
    print("setup=" + json.dumps(setup, indent=2), flush=True)
    append_process_log(args.process_log, "setup", setup)

    history: list[dict] = []
    best_accuracy = -math.inf
    started = time.time()
    print(
        f"START GPDS sampled training: {args.epochs} epochs, "
        f"{train_pairs_per_epoch} train pairs/epoch, {eval_pairs_per_epoch} eval pairs/epoch",
        flush=True,
    )

    for epoch in range(args.epochs):
        train_dataset.resample(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        processed = 0
        epoch_started = time.time()
        limit = args.max_train_batches or len(train_loader)
        torch.cuda.reset_peak_memory_stats()

        for step, (reference, query, label) in enumerate(train_loader, start=1):
            if step > limit:
                break

            reference = reference.to(device, non_blocking=True)
            query = query.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, _, distance = model(reference, query)
                loss = criterion(distance, label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach()) * reference.shape[0]
            processed += reference.shape[0]

            if step == 1 or step % args.log_every == 0 or step == limit:
                elapsed = time.time() - epoch_started
                rate = processed / max(elapsed, 1e-6)
                remaining_epoch_pairs = max(0, train_pairs_per_epoch - processed)
                remaining_epochs_after_this = max(0, args.epochs - epoch - 1)
                remaining_train_pairs = (
                    remaining_epoch_pairs
                    + remaining_epochs_after_this * train_pairs_per_epoch
                )
                conservative_remaining_eval_pairs = (
                    (args.epochs - epoch) * eval_pairs_per_epoch
                )
                eta_seconds = (
                    remaining_train_pairs + conservative_remaining_eval_pairs
                ) / max(rate, 1e-6)
                memory = torch.cuda.max_memory_allocated() / (1024**3)
                loss_value = running_loss / max(processed, 1)
                progress = {
                    "epoch": epoch + 1,
                    "epochs": args.epochs,
                    "step": step,
                    "steps": limit,
                    "running_train_loss": loss_value,
                    "processed_train_pairs_epoch": processed,
                    "pairs_per_second": rate,
                    "eta_seconds": eta_seconds,
                    "eta_human": format_seconds(eta_seconds),
                    "peak_vram_gb": memory,
                }
                append_process_log(args.process_log, "progress", progress)
                print(
                    f"session={args.session_id} epoch={epoch + 1}/{args.epochs} "
                    f"step={step}/{limit} loss={loss_value:.5f} "
                    f"pairs_per_sec={rate:.2f} eta~{format_seconds(eta_seconds)} "
                    f"peak_vram_gb={memory:.2f}",
                    flush=True,
                )

        metrics = evaluate(model, test_loader, device)
        record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(processed, 1),
            "processed_train_pairs": processed,
            "seconds": time.time() - epoch_started,
            **metrics,
        }
        history.append(record)
        print("evaluation=" + json.dumps(record), flush=True)
        append_process_log(args.process_log, "epoch_result", record)

        if metrics["accuracy"] >= best_accuracy:
            best_accuracy = metrics["accuracy"]
            payload = {
                "format": "ResT-GPDS-sampled-v1",
                "paper": "Symmetry 2026, 18, 149",
                "session_id": args.session_id,
                "epoch": epoch + 1,
                "model_config": asdict(config),
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
                "history": history,
                "initialization": initialization,
                "train_writers": train_writers,
                "test_writers": test_writers,
                "training_args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "augmentation": setup["augmentation"],
                "normalization": setup["normalization"],
            }
            atomic_torch_save(payload, args.output)
            print(
                f"saved_best_checkpoint={args.output.resolve()} "
                f"accuracy={metrics['accuracy']:.4f} auc={metrics['auc']:.4f}",
                flush=True,
            )
            append_process_log(
                args.process_log,
                "checkpoint_saved",
                {
                    "epoch": epoch + 1,
                    "checkpoint": str(args.output.resolve()),
                    "metrics": metrics,
                },
            )

    report = {
        **setup,
        "checkpoint": str(args.output.resolve()),
        "checkpoint_bytes": args.output.stat().st_size if args.output.exists() else 0,
        "elapsed_seconds": time.time() - started,
        "elapsed_human": format_seconds(time.time() - started),
        "best_accuracy": best_accuracy,
        "train_writers": train_writers,
        "test_writers": test_writers,
        "history": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("training_complete=" + json.dumps(report), flush=True)
    append_process_log(args.process_log, "training_complete", report)


if __name__ == "__main__":
    main()
