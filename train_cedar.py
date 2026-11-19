"""Train the reconstructed ResT Siamese network on the CEDAR dataset.

The script follows the paper's CEDAR writer split (50 train, 5 test), balanced
genuine-genuine and genuine-forgery pairs, 224x224 inputs, Adam at 5e-5, and
contrastive loss with alpha=beta=margin=1. It supports the official Google
ViT-B/16 ImageNet-21k NPZ checkpoint used by the paper's Transformer branch.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import combinations, product
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from rest_siamese import ContrastiveLoss, ResTConfig, ResTEncoder, SiameseResT


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vit-npz", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/rest_cedar_checkpoint.pt")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("outputs/rest_cedar_training_report.json")
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--pairs-per-writer",
        type=int,
        default=16,
        help="Balanced pair count sampled per training writer per epoch.",
    )
    parser.add_argument("--eval-pairs-per-writer", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--random-writer-split",
        action="store_true",
        help="Shuffle writer IDs with --seed before selecting 50 train / 5 test writers.",
    )
    parser.add_argument("--session-id", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Optional debugging cap; 0 trains the complete sampled epoch.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def otsu_threshold(image: np.ndarray) -> int:
    histogram = np.bincount(image.reshape(-1), minlength=256).astype(np.float64)
    total = image.size
    sum_total = np.dot(np.arange(256, dtype=np.float64), histogram)
    background_weight = 0.0
    background_sum = 0.0
    best_variance = -1.0
    threshold = 127
    for level in range(256):
        background_weight += histogram[level]
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += level * histogram[level]
        background_mean = background_sum / background_weight
        foreground_mean = (sum_total - background_sum) / foreground_weight
        variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if variance > best_variance:
            best_variance = variance
            threshold = level
    return threshold


class SignatureTransform:
    """Paper-style Otsu, inversion, centering, resize, and augmentation."""

    def __init__(
        self,
        training: bool,
        random_erasing: bool = False,
        random_erasing_probability: float = 0.5,
    ) -> None:
        self.training = training
        self.random_erasing = random_erasing
        self.random_erasing_probability = random_erasing_probability
        self.mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)

    def _random_erase(self, tensor: Tensor) -> Tensor:
        """Dependency-free equivalent of torchvision.transforms.RandomErasing."""

        if random.random() >= self.random_erasing_probability:
            return tensor
        _, height, width = tensor.shape
        area = height * width
        for _ in range(10):
            target_area = random.uniform(0.02, 0.33) * area
            aspect_ratio = math.exp(random.uniform(math.log(0.3), math.log(3.3)))
            erase_height = int(round(math.sqrt(target_area * aspect_ratio)))
            erase_width = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < erase_height < height and 0 < erase_width < width:
                top = random.randint(0, height - erase_height)
                left = random.randint(0, width - erase_width)
                tensor[:, top : top + erase_height, left : left + erase_width] = 0.0
                break
        return tensor

    def __call__(self, path: Path | Image.Image) -> Tensor:
        if isinstance(path, Image.Image):
            grayscale = path.convert("L").filter(ImageFilter.GaussianBlur(0.8))
        else:
            with Image.open(path) as source:
                grayscale = source.convert("L").filter(ImageFilter.GaussianBlur(0.8))
        pixels = np.asarray(grayscale, dtype=np.uint8)
        threshold = otsu_threshold(pixels)

        # Keep dark foreground values as positive intensities on a zero background.
        foreground = np.where(pixels <= threshold, 255 - pixels, 0).astype(np.uint8)
        ys, xs = np.nonzero(foreground)
        if len(xs):
            margin = 3
            left = max(0, int(xs.min()) - margin)
            right = min(foreground.shape[1], int(xs.max()) + margin + 1)
            top = max(0, int(ys.min()) - margin)
            bottom = min(foreground.shape[0], int(ys.max()) + margin + 1)
            signature = Image.fromarray(foreground[top:bottom, left:right], mode="L")
        else:
            signature = Image.fromarray(foreground, mode="L")

        signature.thumbnail((236, 236), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (256, 256), color=0)
        x = (256 - signature.width) // 2
        y = (256 - signature.height) // 2
        canvas.paste(signature, (x, y))

        if self.training:
            canvas = canvas.rotate(
                random.uniform(-5.0, 5.0),
                resample=Image.Resampling.BILINEAR,
                fillcolor=0,
            )
            if random.random() < 0.5:
                canvas = ImageOps.mirror(canvas)
            left = random.randint(0, 32)
            top = random.randint(0, 32)
            canvas = canvas.crop((left, top, left + 224, top + 224))
        else:
            canvas = canvas.resize((224, 224), Image.Resampling.LANCZOS)

        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
        tensor = (tensor - self.mean) / self.std
        if self.training and self.random_erasing:
            tensor = self._random_erase(tensor)
        return tensor


def scan_cedar(data_root: Path) -> dict[int, dict[str, list[Path]]]:
    if not data_root.exists():
        raise FileNotFoundError(f"CEDAR directory does not exist: {data_root}")
    writers: dict[int, dict[str, list[Path]]] = {}
    for writer_dir in data_root.iterdir():
        if not writer_dir.is_dir() or not writer_dir.name.isdigit():
            continue
        writer = int(writer_dir.name)
        genuine: list[Path] = []
        forged: list[Path] = []
        for path in writer_dir.iterdir():
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            name = path.name.lower()
            if name.startswith("original"):
                genuine.append(path)
            elif name.startswith("forgeries"):
                forged.append(path)
        if genuine and forged:
            writers[writer] = {
                "genuine": sorted(genuine),
                "forged": sorted(forged),
            }
    if len(writers) != 55:
        raise RuntimeError(f"Expected 55 CEDAR writers, found {len(writers)}.")
    for writer, samples in writers.items():
        if len(samples["genuine"]) != 24 or len(samples["forged"]) != 24:
            raise RuntimeError(
                f"Writer {writer}: expected 24 genuine and 24 forged, found "
                f"{len(samples['genuine'])} and {len(samples['forged'])}."
            )
    return writers


class CedarPairDataset(Dataset):
    def __init__(
        self,
        writers: dict[int, dict[str, list[Path]]],
        writer_ids: Iterable[int],
        pairs_per_writer: int,
        training: bool,
        seed: int,
        random_erasing: bool = False,
    ) -> None:
        if pairs_per_writer < 2:
            raise ValueError("pairs_per_writer must be at least 2.")
        self.writers = writers
        self.writer_ids = list(writer_ids)
        self.pairs_per_writer = pairs_per_writer
        self.transform = SignatureTransform(
            training, random_erasing=random_erasing
        )
        self.training = training
        self.seed = seed
        self.pairs: list[tuple[Path, Path, float]] = []
        self.resample(0)

    def resample(self, epoch: int) -> None:
        rng = random.Random(self.seed + epoch)
        pairs: list[tuple[Path, Path, float]] = []
        positive_count = self.pairs_per_writer // 2
        negative_count = self.pairs_per_writer - positive_count
        for writer in self.writer_ids:
            genuine = self.writers[writer]["genuine"]
            forged = self.writers[writer]["forged"]
            positive_pool = list(combinations(genuine, 2))
            negative_pool = list(product(genuine, forged))
            for first, second in rng.sample(positive_pool, positive_count):
                pairs.append((first, second, 1.0))
            for first, second in rng.sample(negative_pool, negative_count):
                pairs.append((first, second, 0.0))
        rng.shuffle(pairs)
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        reference, query, label = self.pairs[index]
        return (
            self.transform(reference),
            self.transform(query),
            torch.tensor(label, dtype=torch.float32),
        )


@torch.no_grad()
def load_vit_b16_imagenet21k(encoder: ResTEncoder, checkpoint: Path) -> dict:
    """Map Google's official JAX ViT-B/16 NPZ weights into this PyTorch ViT."""

    if encoder.config.vit_dim != 768 or encoder.config.vit_depth != 12:
        raise ValueError("ImageNet-21k ViT-B/16 weights require width 768 and depth 12.")
    branch = encoder.global_branch
    loaded_tensors = 0
    loaded_elements = 0

    def assign(parameter: Tensor, array: np.ndarray) -> None:
        nonlocal loaded_tensors, loaded_elements
        value = torch.from_numpy(np.asarray(array)).to(dtype=parameter.dtype)
        if tuple(value.shape) != tuple(parameter.shape):
            raise ValueError(
                f"Weight shape mismatch: checkpoint {tuple(value.shape)}, "
                f"model {tuple(parameter.shape)}"
            )
        parameter.copy_(value)
        loaded_tensors += 1
        loaded_elements += parameter.numel()

    with np.load(checkpoint) as weights:
        assign(
            branch.patch_embed.weight,
            weights["embedding/kernel"].transpose(3, 2, 0, 1),
        )
        assign(branch.patch_embed.bias, weights["embedding/bias"])
        assign(branch.class_token, weights["cls"])
        assign(
            branch.position,
            weights["Transformer/posembed_input/pos_embedding"],
        )

        for index, block in enumerate(branch.blocks):
            prefix = f"Transformer/encoderblock_{index}"
            attention_prefix = f"{prefix}/MultiHeadDotProductAttention_1"
            assign(block.norm1.weight, weights[f"{prefix}/LayerNorm_0/scale"])
            assign(block.norm1.bias, weights[f"{prefix}/LayerNorm_0/bias"])

            qkv_weights = []
            qkv_biases = []
            for name in ("query", "key", "value"):
                kernel = weights[f"{attention_prefix}/{name}/kernel"]
                qkv_weights.append(kernel.reshape(768, 768).T)
                qkv_biases.append(
                    weights[f"{attention_prefix}/{name}/bias"].reshape(768)
                )
            assign(block.attention.in_proj_weight, np.concatenate(qkv_weights, axis=0))
            assign(block.attention.in_proj_bias, np.concatenate(qkv_biases, axis=0))
            out_kernel = weights[f"{attention_prefix}/out/kernel"].reshape(768, 768)
            assign(block.attention.out_proj.weight, out_kernel.T)
            assign(
                block.attention.out_proj.bias,
                weights[f"{attention_prefix}/out/bias"],
            )

            assign(block.norm2.weight, weights[f"{prefix}/LayerNorm_2/scale"])
            assign(block.norm2.bias, weights[f"{prefix}/LayerNorm_2/bias"])
            assign(
                block.mlp.net[0].weight,
                weights[f"{prefix}/MlpBlock_3/Dense_0/kernel"].T,
            )
            assign(
                block.mlp.net[0].bias,
                weights[f"{prefix}/MlpBlock_3/Dense_0/bias"],
            )
            assign(
                block.mlp.net[3].weight,
                weights[f"{prefix}/MlpBlock_3/Dense_1/kernel"].T,
            )
            assign(
                block.mlp.net[3].bias,
                weights[f"{prefix}/MlpBlock_3/Dense_1/bias"],
            )

        assign(branch.norm.weight, weights["Transformer/encoder_norm/scale"])
        assign(branch.norm.bias, weights["Transformer/encoder_norm/bias"])

    return {"tensors": loaded_tensors, "elements": loaded_elements}


def compute_metrics(distances: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    positive = distances[labels == 1]
    negative = distances[labels == 0]
    if not len(positive) or not len(negative):
        raise ValueError("Metrics require both positive and negative pairs.")
    thresholds = np.linspace(float(distances.min()), float(distances.max()), 1001)
    best_accuracy = -1.0
    best_threshold = float(thresholds[0])
    eer = 1.0
    eer_threshold = best_threshold
    best_gap = float("inf")
    best_far = 1.0
    best_frr = 1.0
    for threshold in thresholds:
        far = float(np.mean(negative <= threshold))
        frr = float(np.mean(positive > threshold))
        accuracy = float(np.mean((distances <= threshold) == labels.astype(bool)))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            eer = (far + frr) / 2.0
            eer_threshold = float(threshold)
            best_far = far
            best_frr = frr
    comparisons = positive[:, None] - negative[None, :]
    auc = float(np.mean(comparisons < 0) + 0.5 * np.mean(comparisons == 0))
    return {
        "accuracy": best_accuracy,
        "threshold": best_threshold,
        "auc": auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "far_at_eer": best_far,
        "frr_at_eer": best_frr,
        "positive_mean_distance": float(positive.mean()),
        "negative_mean_distance": float(negative.mean()),
    }


@torch.no_grad()
def evaluate(
    model: SiameseResT, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    distances: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for reference, query, label in loader:
        reference = reference.to(device, non_blocking=True)
        query = query.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            _, _, distance = model(reference, query)
        distances.append(distance.float().cpu().numpy())
        labels.append(label.numpy())
    return compute_metrics(np.concatenate(distances), np.concatenate(labels))


def atomic_torch_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this ViT-B/16 training script.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    writers = scan_cedar(args.data_root)
    writer_ids = sorted(writers)
    if args.random_writer_split:
        split_rng = random.Random(args.seed)
        split_rng.shuffle(writer_ids)
        test_writers = sorted(writer_ids[:5])
        train_writers = sorted(writer_ids[5:])
        split_mode = "seeded random writer-independent split"
    else:
        train_writers = writer_ids[:50]
        test_writers = writer_ids[50:]
        split_mode = "fixed writers 1-50 train / 51-55 test"
    train_dataset = CedarPairDataset(
        writers,
        train_writers,
        pairs_per_writer=args.pairs_per_writer,
        training=True,
        seed=args.seed,
    )
    test_dataset = CedarPairDataset(
        writers,
        test_writers,
        pairs_per_writer=args.eval_pairs_per_writer,
        training=False,
        seed=args.seed + 100_000,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    config = ResTConfig(gradient_checkpointing=True)
    encoder = ResTEncoder(config)
    initialization: dict = {"source": "random"}
    if args.vit_npz is not None:
        stats = load_vit_b16_imagenet21k(encoder, args.vit_npz)
        initialization = {
            "source": "Google ViT-B/16 ImageNet-21k",
            "path": str(args.vit_npz.resolve()),
            **stats,
        }

    model = SiameseResT(encoder).to(device)
    criterion = ContrastiveLoss(margin=1.0, positive_weight=1.0, negative_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "parameters": parameter_count,
                "train_writers": train_writers,
                "test_writers": test_writers,
                "train_pairs_per_epoch": len(train_dataset),
                "test_pairs": len(test_dataset),
                "effective_batch_size": args.batch_size * args.accumulation_steps,
                "session_id": args.session_id,
                "split_mode": split_mode,
                "initialization": initialization,
            },
            indent=2,
        ),
        flush=True,
    )

    history: list[dict] = []
    best_accuracy = -math.inf
    started = time.time()
    for epoch in range(args.epochs):
        train_dataset.resample(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        processed = 0
        epoch_started = time.time()
        limit = args.max_train_batches or len(train_loader)
        for step, (reference, query, label) in enumerate(train_loader, start=1):
            if step > limit:
                break
            reference = reference.to(device, non_blocking=True)
            query = query.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, _, distance = model(reference, query)
                raw_loss = criterion(distance, label)
                loss = raw_loss / args.accumulation_steps
            scaler.scale(loss).backward()
            running_loss += float(raw_loss.detach()) * reference.shape[0]
            processed += reference.shape[0]

            should_step = step % args.accumulation_steps == 0 or step == limit
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if step == 1 or step % args.log_every == 0 or step == limit:
                elapsed = time.time() - epoch_started
                rate = processed / max(elapsed, 1e-6)
                memory = torch.cuda.max_memory_allocated() / (1024**3)
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step}/{limit} "
                    f"loss={running_loss / max(processed, 1):.5f} "
                    f"pairs_per_sec={rate:.2f} peak_vram_gb={memory:.2f}",
                    flush=True,
                )

        metrics = evaluate(model, test_loader, device)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(processed, 1),
            "processed_train_pairs": processed,
            "seconds": time.time() - epoch_started,
            **metrics,
        }
        history.append(epoch_record)
        print("evaluation=" + json.dumps(epoch_record), flush=True)

        if metrics["accuracy"] >= best_accuracy:
            best_accuracy = metrics["accuracy"]
            payload = {
                "format": "ResT-CEDAR-v1",
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
                "split_mode": split_mode,
                "training_args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            }
            atomic_torch_save(payload, args.output)
            print(f"saved_best_checkpoint={args.output.resolve()}", flush=True)

    report = {
        "checkpoint": str(args.output.resolve()),
        "checkpoint_bytes": args.output.stat().st_size,
        "device": torch.cuda.get_device_name(0),
        "session_id": args.session_id,
        "seed": args.seed,
        "split_mode": split_mode,
        "train_writers": train_writers,
        "test_writers": test_writers,
        "parameter_count": parameter_count,
        "elapsed_seconds": time.time() - started,
        "best_accuracy": best_accuracy,
        "history": history,
        "initialization": initialization,
        "paper_protocol": {
            "train_writers": 50,
            "test_writers": 5,
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "contrastive_alpha": 1.0,
            "contrastive_beta": 1.0,
            "contrastive_margin": 1.0,
            "model_input": "224x224",
        },
        "practical_sampling_note": (
            "The paper uses all 276 genuine pairs per writer and an equal number "
            "of sampled negative pairs. This run samples a balanced subset per "
            "writer per epoch as specified by pairs_per_writer."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("training_complete=" + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
