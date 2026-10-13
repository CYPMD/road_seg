"""Each method module exports build_model(**params) -> torch.nn.Module."""

from importlib import import_module
from pkgutil import iter_modules
import re

from torch import nn


def available_methods() -> list[str]:
    return sorted(item.name for item in iter_modules(__path__) if not item.name.startswith("_"))


def create_model(name: str, **params) -> nn.Module:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name not in available_methods():
        raise ValueError(f"Unknown method {name!r}. Available: {available_methods()}")
    module = import_module(f"methods.{name}")
    if not callable(getattr(module, "build_model", None)):
        raise TypeError(f"methods/{name} must export build_model(**params)")
    model = module.build_model(**params)
    if not isinstance(model, nn.Module):
        raise TypeError("build_model must return a torch.nn.Module")
    return model
