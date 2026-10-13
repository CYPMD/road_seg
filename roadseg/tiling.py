"""Full-resolution probability averaging with complete border coverage."""

import math

import torch
from torch.nn import functional as F


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
    try:
        for start in range(0, len(coordinates), batch_size):
            locations = coordinates[start:start + batch_size]
            tiles = torch.stack([image[:, y:y + patch_size, x:x + patch_size]
                                 for y, x in locations]).to(device)
            with torch.autocast(device_type=device.type,
                                enabled=bool(amp and device.type == "cuda")):
                logits = model(tiles)
            expected = (len(locations), 1, patch_size, patch_size)
            if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected:
                raise ValueError(f"Method must return raw logits with shape {expected}")
            probabilities = torch.sigmoid(logits.float()).cpu()[:, 0]
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("Model produced nonfinite probabilities")
            for probability, (y, x) in zip(probabilities, locations):
                total[y:y + patch_size, x:x + patch_size] += probability
                count[y:y + patch_size, x:x + patch_size] += 1
    finally:
        model.train(was_training)
    return (total / count)[:height, :width].contiguous()
