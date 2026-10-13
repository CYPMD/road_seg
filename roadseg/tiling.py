"""Full-resolution probability averaging with complete border coverage."""

import math
import warnings

import torch
from torch.nn import functional as F

from .utils import check_model_finite


def _predict_tile_batch(model, tiles, use_amp):
    """Return finite FP32 probabilities and the precision for subsequent batches."""
    expected = (tiles.shape[0], 1, tiles.shape[2], tiles.shape[3])

    def forward(enabled):
        with torch.autocast(device_type=tiles.device.type, dtype=torch.float16, enabled=enabled):
            logits = model(tiles.float())
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected:
            raise ValueError(f"Method must return raw logits with shape {expected}")
        return logits

    logits = forward(use_amp)
    if not torch.isfinite(logits).all() and use_amp:
        # Casting failed FP16 logits to float32 is too late: rerun the forward pass.
        logits = forward(False)
        use_amp = False
        if torch.isfinite(logits).all():
            warnings.warn(
                "AMP produced nonfinite logits; retried this tile batch in float32. "
                "Using float32 for the remaining tiles of this image.",
                RuntimeWarning, stacklevel=2,
            )
    if not torch.isfinite(logits).all():
        check_model_finite(model, "tiled inference")
        raise FloatingPointError(
            "Model produced nonfinite logits even in float32. "
            "Parameters and buffers are finite; check input normalization and training stability. "
            "Restart with --no-amp if this model was trained with AMP."
        )
    # Check logits before sigmoid: sigmoid(+/-inf) would otherwise hide overflow.
    probabilities = torch.sigmoid(logits.float()).cpu()[:, 0]
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("Model produced nonfinite probabilities in float32")
    return probabilities, use_amp


def tile_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length < 1 or patch_size < 1 or not 1 <= stride <= patch_size:
        raise ValueError("Need length > 0, patch_size > 0 and 1 <= stride <= patch_size")
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


@torch.inference_mode()
def predict_tiled(model, image: torch.Tensor, device, patch_size=256,
                  overlap=0.5, batch_size=16, amp=False) -> torch.Tensor:
    """Take normalized [3,H,W] input; return CPU float32 probabilities [H,W].

    Sigmoid is applied per patch. Overlapping probabilities are averaged before
    thresholding. Only small images are padded; the returned size is unchanged.
    GPU memory scales with tile batch size, not full-image dimensions.
    Nonfinite AMP logits trigger a float32 retry; invalid FP32 outputs still fail.
    """
    if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[-2:]) < 1:
        raise ValueError("Expected normalized RGB image [3,H,W]")
    if not isinstance(patch_size, int) or patch_size < 1 or batch_size < 1:
        raise ValueError("patch_size and batch_size must be positive integers")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0,1)")
    raw_stride = patch_size * (1 - overlap)
    stride = round(raw_stride)
    if stride < 1 or not math.isclose(raw_stride, stride, abs_tol=1e-6):
        raise ValueError("patch_size * (1 - overlap) must be a positive integer")
    device = torch.device(device)
    image = image.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(image).all():
        raise FloatingPointError("Input image contains nonfinite values; check preprocessing")
    height, width = image.shape[-2:]
    pad_h, pad_w = max(0, patch_size - height), max(0, patch_size - width)
    if pad_h or pad_w:
        image = F.pad(image[None], (0, pad_w, 0, pad_h), mode="replicate")[0]
    padded_h, padded_w = image.shape[-2:]
    coordinates = [(y, x) for y in tile_starts(padded_h, patch_size, stride)
                   for x in tile_starts(padded_w, patch_size, stride)]
    total = torch.zeros((padded_h, padded_w), dtype=torch.float32)
    count = torch.zeros_like(total)
    was_training = model.training
    model.eval()
    use_amp = bool(amp and device.type == "cuda")
    try:
        for start in range(0, len(coordinates), batch_size):
            locations = coordinates[start:start + batch_size]
            tiles = torch.stack([image[:, y:y + patch_size, x:x + patch_size]
                                 for y, x in locations]).to(device)
            probabilities, use_amp = _predict_tile_batch(model, tiles, use_amp)
            for probability, (y, x) in zip(probabilities, locations):
                total[y:y + patch_size, x:x + patch_size] += probability
                count[y:y + patch_size, x:x + patch_size] += 1
    finally:
        model.train(was_training)
    return (total / count)[:height, :width].contiguous()
