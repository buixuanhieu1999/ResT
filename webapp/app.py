"""Phone-friendly Gradio app: signature scanning and ResT pair verification.

    python webapp/app.py --checkpoint checkpoints/rest_gpds_32pairs_10epochs_best.pt
    python webapp/app.py --share          # HTTPS link, needed for phone cameras

The checkpoint can also be given through the REST_CKPT environment variable.
If the model cannot be loaded, the Scan tab still works and Verify is disabled.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import inspect
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

import gradio as gr
import numpy as np
from PIL import Image
import torch

WEBAPP_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEBAPP_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(WEBAPP_DIR))

from cleanup import clean_signature  # noqa: E402
from rest_siamese import ResTConfig, ResTEncoder, SiameseResT  # noqa: E402
from train_cedar import SignatureTransform  # noqa: E402

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "rest_gpds_32pairs_10epochs_best.pt"
LFS_POINTER_MAX_BYTES = 10 * 1024
FALLBACK_THRESHOLD = 0.4618
DISCLAIMER = (
    "**Mock-test use only.** This model reached ~75% accuracy (AUC 0.825, EER 25.4%) "
    "on 30 GPDS test writers, and phone photos are out of its training domain. "
    "Do not use the verdict for any real decision."
)


class Verifier:
    """Loads the Siamese ResT checkpoint and scores cleaned signature pairs."""

    def __init__(self, checkpoint: Path, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(os.cpu_count() or 1)

        payload = self._read_checkpoint(checkpoint)
        known = {field.name for field in dataclasses.fields(ResTConfig)}
        raw_config = payload.get("model_config") or {}
        config_kwargs = {k: v for k, v in raw_config.items() if k in known}
        config_kwargs["gradient_checkpointing"] = False  # saved True; inference-only here
        config = ResTConfig(**config_kwargs)
        state = payload.get("model_state_dict", payload)
        state = {_strip_prefixes(key): value for key, value in state.items()}

        model = SiameseResT(ResTEncoder(config))
        model.load_state_dict(state)
        self.encoder = model.encoder.eval().to(self.device)
        self.transform = SignatureTransform(training=False)

        metrics = payload.get("metrics") or {}
        self.threshold = float(metrics.get("threshold", FALLBACK_THRESHOLD))
        self.eer_threshold = metrics.get("eer_threshold")
        self.epoch = payload.get("epoch")
        self.parameters = sum(p.numel() for p in self.encoder.parameters())

    @staticmethod
    def _read_checkpoint(path: Path) -> dict:
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}. Pass --checkpoint or set REST_CKPT."
            )
        size = path.stat().st_size
        if size < LFS_POINTER_MAX_BYTES:
            raise RuntimeError(
                f"{path} is only {size} bytes, so it is a Git LFS pointer, not the "
                "weights. Run `git lfs pull --include=\"checkpoints/*.pt\"` "
                "(about 388 MB) or point --checkpoint at a downloaded copy."
            )
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Our own trusted training checkpoint may contain non-tensor objects.
            return torch.load(path, map_location="cpu", weights_only=False)

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    @torch.inference_mode()
    def distance(self, reference: Image.Image, query: Image.Image) -> tuple[float, float]:
        """Return (L2 distance, milliseconds) for one cleaned pair."""

        batch = torch.stack((self.transform(reference), self.transform(query)))
        batch = batch.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with self._autocast():
            embeddings = self.encoder(batch).float()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        value = torch.linalg.vector_norm(embeddings[0] - embeddings[1]).item()
        return value, elapsed_ms

    def warm_up(self) -> float:
        blank = Image.new("RGB", (300, 120), "white")
        return self.distance(blank, blank)[1]

    def describe(self) -> str:
        name = str(self.device)
        if self.device.type == "cuda":
            name += f" ({torch.cuda.get_device_name(self.device)}, fp16 autocast)"
        else:
            name += f" ({torch.get_num_threads()} threads, fp32)"
        return name


def _strip_prefixes(key: str) -> str:
    for prefix in ("module.", "_orig_mod."):
        while key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _save_png(array: np.ndarray, stem: str) -> str:
    handle = tempfile.NamedTemporaryFile(prefix=f"{stem}_", suffix=".png", delete=False)
    handle.close()
    Image.fromarray(array).save(handle.name)
    return handle.name


def scan(image, sensitivity, min_speck, remove_border, ink_colour):
    if image is None:
        raise gr.Error("Take or upload a photo of the signature first.")
    result = clean_signature(image, sensitivity, min_speck, remove_border, ink_colour)
    if not result.found_ink:
        raise gr.Error(
            "No ink found. Try raising sensitivity, lowering speck removal, or "
            "retaking the photo closer with even light."
        )
    return (
        result.rgb,
        _save_png(result.rgba, "signature_transparent"),
        _save_png(result.rgb, "signature_white"),
    )


def build_app(verifier: Verifier | None, load_error: str | None) -> gr.Blocks:
    def image_input(label: str) -> gr.Image:
        kwargs = dict(label=label, sources=["upload", "webcam"], type="numpy", height=320)
        # Mirrored webcam captures would flip the signature left-to-right.
        params = inspect.signature(gr.Image.__init__).parameters
        if "webcam_options" in params and hasattr(gr, "WebcamOptions"):
            kwargs["webcam_options"] = gr.WebcamOptions(mirror=False)
        elif "mirror_webcam" in params:
            kwargs["mirror_webcam"] = False
        return gr.Image(**kwargs)

    def cleanup_controls():
        sensitivity = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Sensitivity (higher keeps fainter ink)")
        min_speck = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Speck removal")
        remove_border = gr.Checkbox(value=True, label="Remove border blobs (paper edges, shadows)")
        return sensitivity, min_speck, remove_border

    def verify(reference, query, threshold, sensitivity, min_speck, remove_border):
        if verifier is None:
            raise gr.Error(f"Model not loaded: {load_error}")
        if reference is None or query is None:
            raise gr.Error("Provide both the reference and the signature to check.")
        cleaned = []
        for name, image in (("Reference", reference), ("Check", query)):
            result = clean_signature(image, sensitivity, min_speck, remove_border, "black")
            if not result.found_ink:
                raise gr.Error(f"{name} photo: no ink found. Adjust sensitivity or retake it.")
            cleaned.append(result.rgb)
        try:
            value, elapsed_ms = verifier.distance(*(Image.fromarray(c) for c in cleaned))
        except Exception as error:  # surface inference failures in the UI
            traceback.print_exc()
            raise gr.Error(f"Inference failed: {error}") from error

        genuine = value <= threshold
        verdict = "✅ Likely GENUINE (same writer)" if genuine else "❌ Likely FORGED / different writer"
        report = (
            f"## {verdict}\n\n"
            f"| | |\n|---|---|\n"
            f"| Distance | **{value:.4f}** {'≤' if genuine else '>'} threshold {threshold:.4f} |\n"
            f"| Margin | {threshold - value:+.4f} |\n"
            f"| Inference | {elapsed_ms:.0f} ms for the pair |\n"
            f"| Device | {verifier.describe()} |\n\n"
            f"{DISCLAIMER}"
        )
        return report, cleaned[0], cleaned[1]

    with gr.Blocks(title="ResT Signature Scanner") as app:
        gr.Markdown(
            "# ResT signature scanner & verifier\n"
            "On a phone, tap the upload area to open the native camera (sharper than the webcam view). "
            "Fill the frame with the signature and avoid strong shadows."
        )
        with gr.Tab("Scan"):
            scan_image = image_input("Signature photo")
            sensitivity, min_speck, remove_border = cleanup_controls()
            ink_colour = gr.Radio(["black", "blue"], value="black", label="Ink colour")
            scan_button = gr.Button("Clean up", variant="primary")
            scan_preview = gr.Image(label="Cleaned signature", type="numpy", interactive=False)
            with gr.Row():
                transparent_file = gr.File(label="Transparent PNG")
                white_file = gr.File(label="White-background PNG")
            scan_button.click(
                scan,
                [scan_image, sensitivity, min_speck, remove_border, ink_colour],
                [scan_preview, transparent_file, white_file],
                api_name="scan",
            )

        with gr.Tab("Verify"):
            if verifier is None:
                gr.Markdown(f"⚠️ **Verification unavailable — model failed to load.**\n\n```\n{load_error}\n```")
                default_threshold = FALLBACK_THRESHOLD
            else:
                gr.Markdown(
                    f"Model: epoch {verifier.epoch}, {verifier.parameters / 1e6:.1f}M params on "
                    f"{verifier.describe()}. Checkpoint threshold {verifier.threshold:.4f}"
                    + (f" (EER threshold {verifier.eer_threshold:.4f})" if verifier.eer_threshold else "")
                    + "."
                )
                default_threshold = verifier.threshold
            gr.Markdown(DISCLAIMER)
            with gr.Row():
                reference_image = image_input("Reference (known genuine)")
                query_image = image_input("Signature to check")
            threshold = gr.Slider(0.05, 1.5, value=round(default_threshold, 4), step=0.0001, label="Decision threshold (distance ≤ threshold → genuine)")
            with gr.Accordion("Cleanup options", open=False):
                v_sensitivity, v_min_speck, v_remove_border = cleanup_controls()
            verify_button = gr.Button("Verify", variant="primary", interactive=verifier is not None)
            verdict = gr.Markdown()
            with gr.Row():
                reference_clean = gr.Image(label="Cleaned reference", type="numpy", interactive=False)
                query_clean = gr.Image(label="Cleaned check", type="numpy", interactive=False)
            verify_button.click(
                verify,
                [reference_image, query_image, threshold, v_sensitivity, v_min_speck, v_remove_border],
                [verdict, reference_clean, query_clean],
                api_name="verify",
            )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("REST_CKPT", DEFAULT_CHECKPOINT)))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--share", action="store_true", help="create a public HTTPS gradio.live link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0", help="bind address (0.0.0.0 allows LAN access)")
    args = parser.parse_args()

    verifier, load_error = None, None
    try:
        print(f"Loading checkpoint {args.checkpoint} ...", flush=True)
        verifier = Verifier(args.checkpoint, args.device)
        warm_ms = verifier.warm_up()
        print(f"Model ready on {verifier.describe()}; warm-up pair took {warm_ms:.0f} ms; "
              f"threshold={verifier.threshold:.4f}", flush=True)
    except Exception as error:
        load_error = f"{type(error).__name__}: {error}"
        print(f"WARNING: model not loaded, Verify disabled. {load_error}", flush=True)

    build_app(verifier, load_error).queue().launch(
        server_name=args.host, server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
