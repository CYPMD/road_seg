from pathlib import Path
import copy
import math

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    for key in ("seed", "method", "data", "preprocessing", "loss", "train", "inference"):
        if key not in config:
            raise ValueError(f"Missing config section: {key}")
    data, train, infer = config["data"], config["train"], config["inference"]
    for name, value in [
        ("crop_size", data["crop_size"]), ("crops_per_image", data["crops_per_image"]),
        ("epochs", train["epochs"]), ("batch_size", train["batch_size"]),
        ("patch_size", infer["patch_size"]), ("inference.batch_size", infer["batch_size"]),
    ]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if data["crop_size"] < 32 or infer["patch_size"] < 32:
        raise ValueError("Crop and patch sizes must be at least 32")
    if train["num_workers"] < 0:
        raise ValueError("num_workers must be >= 0")
    if not 0 < data["val_fraction"] < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if not 1 <= data["mask_threshold"] <= 255:
        raise ValueError("mask_threshold must be between 1 and 255")
    if not data["image_suffix"] or not data["mask_suffix"]:
        raise ValueError("Image and mask suffixes must be nonempty")
    if not 0 <= infer["overlap"] < 1 or not 0 < infer["threshold"] < 1:
        raise ValueError("overlap must be in [0, 1); threshold must be in (0, 1)")
    stride = infer["patch_size"] * (1 - infer["overlap"])
    if not math.isclose(stride, round(stride), abs_tol=1e-6) or stride < 1:
        raise ValueError("patch_size * (1 - overlap) must be a positive integer")
    pre = config["preprocessing"]
    if len(pre["mean"]) != 3 or len(pre["std"]) != 3:
        raise ValueError("RGB preprocessing needs three mean and std values")
    if not all(math.isfinite(v) for v in pre["mean"] + pre["std"]) or min(pre["std"]) <= 0:
        raise ValueError("Normalization must be finite, with positive std values")
    weights = [config["loss"]["bce_weight"], config["loss"]["dice_weight"]]
    if min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("Loss weights must be nonnegative, with at least one positive")
    if train["lr"] <= 0 or train["weight_decay"] < 0 or train["grad_clip_norm"] < 0:
        raise ValueError("Invalid optimizer settings")


def check_resume_config(old: dict, new: dict) -> None:
    """Allow execution/location changes, not silent changes to the experiment."""
    a, b = copy.deepcopy(old), copy.deepcopy(new)
    for conf in (a, b):
        conf["data"].pop("root", None)
        for key in ("epochs", "output_dir", "device", "num_workers", "amp"):
            conf["train"].pop(key, None)
    if a != b:
        raise ValueError("Resume config changes the experiment. Only epochs, output_dir, "
                         "data.root, device, num_workers and amp may change.")
