import warnings

import numpy as np
import torch
from tqdm.auto import tqdm

from .data import read_rgb, read_mask, image_tensor
from .metrics import BinaryMetrics
from .tiling import predict_tiled
from .utils import check_model_finite


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, amp, grad_clip_norm):
    device = torch.device(device)
    amp = bool(amp and device.type == "cuda")
    check_model_finite(model, "before training")
    model.train()
    loss_sum, samples = 0.0, 0
    skipped = 0
    progress = tqdm(loader, desc="train", leave=False)
    for batch_index, (images, masks) in enumerate(progress, start=1):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        if not torch.isfinite(images).all() or not torch.isfinite(masks).all():
            raise FloatingPointError(f"Nonfinite training input/target in batch {batch_index}")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
        if not torch.isfinite(logits).all():
            check_model_finite(model, f"training batch {batch_index}")
            raise FloatingPointError(
                f"Nonfinite training logits in batch {batch_index}. "
                "Restart with --no-amp in a new output directory; check normalization and learning rate."
            )
        # Keep the complete loss calculation outside autocast (BCEDiceLoss uses FP32).
        with torch.autocast(device_type=device.type, enabled=False):
            loss = criterion(logits.float(), masks.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss in batch {batch_index}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            raise RuntimeError("Training produced no parameter gradients")
        gradients_finite = torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all()
        if not gradients_finite:
            if not scaler.is_enabled():
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Nonfinite gradients in batch {batch_index}; optimizer update was not applied. "
                    "Check normalization and reduce the learning rate."
                )
            # unscale_ recorded the overflow. GradScaler skips this update and
            # reduces its scale; do not clip inf/NaN gradients into other values.
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            skipped += 1
            continue
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        samples += images.shape[0]
        loss_sum += loss.item() * images.shape[0]
        progress.set_postfix(loss=f"{loss_sum / samples:.4f}")
    check_model_finite(model, "after training, before validation")
    if samples == 0:
        raise FloatingPointError(
            "No optimizer updates completed this epoch (empty loader or all gradients overflowed). "
            "Restart with --no-amp and check the data."
        )
    if skipped:
        warnings.warn(f"AMP skipped {skipped} training batches with nonfinite gradients; "
                      "reported loss includes only completed updates.", RuntimeWarning, stacklevel=2)
    return loss_sum / samples


def evaluate(model, pairs, config, device, amp=False):
    infer = config["inference"]
    metrics = BinaryMetrics()
    model.eval()
    for pair in tqdm(pairs, desc="validate full images", leave=False):
        image, mask = read_rgb(pair.image), read_mask(pair.mask)
        if image.size != mask.size:
            raise ValueError(f"Image/mask size mismatch: {pair.image}")
        try:
            probabilities = predict_tiled(
                model, image_tensor(image, config["preprocessing"]), device,
                patch_size=infer["patch_size"], overlap=infer["overlap"],
                batch_size=infer["batch_size"], amp=amp,
            )
        except FloatingPointError as error:
            raise FloatingPointError(f"Validation image {pair.image}: {error}") from error
        metrics.update(probabilities.numpy() >= infer["threshold"],
                       np.array(mask) >= config["data"]["mask_threshold"])
    return metrics.compute()
