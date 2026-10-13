from pathlib import Path
import os
import random

import numpy as np
import torch

from methods import create_model


@torch.no_grad()
def check_model_finite(model, context="model"):
    """Validate parameters AND buffers; GradScaler does not protect BN buffers."""
    for kind, tensors in (("parameter", model.named_parameters()),
                          ("buffer", model.named_buffers())):
        for name, tensor in tensors:
            if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
                raise FloatingPointError(
                    f"{context}: nonfinite {kind} '{name}'. "
                    "The model state is invalid; disabling AMP cannot repair it. "
                    "Restart with --no-amp in a new output directory, or resume a known-good checkpoint."
                )


def select_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    # Only tensors and plain Python values are saved; no pickled model objects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return checkpoint


def model_from_checkpoint(checkpoint, device, method=None):
    settings = checkpoint["config"]["method"]
    if method is not None and method != settings["name"]:
        raise ValueError(f"Requested method {method!r} does not match checkpoint {settings['name']!r}")
    params = dict(settings["params"])
    if "pretrained" in params:
        # Checkpoint weights fully replace initialization; avoid network/cache dependency.
        params["pretrained"] = False
    model = create_model(settings["name"], **params).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    check_model_finite(model, "loaded checkpoint")
    return model.eval()
