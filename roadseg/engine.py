import numpy as np
import torch
from tqdm.auto import tqdm

from .data import read_rgb, read_mask, image_tensor
from .metrics import BinaryMetrics
from .tiling import predict_tiled


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, amp, grad_clip_norm):
    model.train()
    loss_sum, samples = 0.0, 0
    progress = tqdm(loader, desc="train", leave=False)
    for images, masks in progress:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        samples += images.shape[0]
        loss_sum += loss.item() * images.shape[0]
        progress.set_postfix(loss=f"{loss_sum / samples:.4f}")
    return loss_sum / samples


def evaluate(model, pairs, config, device, amp=False):
    infer = config["inference"]
    metrics = BinaryMetrics()
    model.eval()
    for pair in tqdm(pairs, desc="validate full images", leave=False):
        image, mask = read_rgb(pair.image), read_mask(pair.mask)
        if image.size != mask.size:
            raise ValueError(f"Image/mask size mismatch: {pair.image}")
        probabilities = predict_tiled(
            model, image_tensor(image, config["preprocessing"]), device,
            patch_size=infer["patch_size"], overlap=infer["overlap"],
            batch_size=infer["batch_size"], amp=amp,
        )
        metrics.update(probabilities.numpy() >= infer["threshold"],
                       np.array(mask) >= config["data"]["mask_threshold"])
    return metrics.compute()
