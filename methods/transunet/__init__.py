"""R50-ViT-B/16 TransUNet, adapted to the shared road-segmentation interface.

Architecture and encoder weight names follow the supplied Beckschen/TransUNet
implementation (Apache-2.0; see LICENSE in this directory). The CNN has (3, 4, 9)
bottleneck units, followed by 12 transformer blocks and three decoder skips.
Only PyTorch and NumPy, already required by this project, are needed.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.hub import download_url_to_file, get_dir

from ._modeling import VisionTransformer


_PRETRAINED_URL = "https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz"
_PRETRAINED_SHA256 = "ff009bf39bb4f9198b834cfe46aba2bfdaf730e933ab3e3c4b1edf4226eaafbe"


def _pretrained_weights_path(pretrained_path):
    """Use an explicit file or download the joint ImageNet-21k encoder once."""
    if pretrained_path is not None:
        path = Path(pretrained_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"TransUNet pretrained weights not found: {path}")
        return path

    path = Path(get_dir()) / "checkpoints" / "R50+ViT-B_16-ff009bf3.npz"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            download_url_to_file(_PRETRAINED_URL, str(path),
                                 hash_prefix=_PRETRAINED_SHA256, progress=True)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                "Could not download the pretrained ResNet + ViT encoder. "
                "Download R50+ViT-B_16.npz from the TransUNet project's weights folder "
                "(https://github.com/Beckschen/TransUNet#usage) and set "
                "method.params.pretrained_path to that local file. "
                "Pretrained initialization is required; training has not started."
            ) from error
    return path


def _make_config(img_size, out_channels):
    """Make an independent copy of the upstream R50-ViT-B_16 configuration."""
    return SimpleNamespace(
        patches={"grid": (img_size // 16, img_size // 16)},
        hidden_size=768,
        transformer={
            "mlp_dim": 3072,
            "num_heads": 12,
            "num_layers": 12,
            "attention_dropout_rate": 0.0,
            "dropout_rate": 0.1,
        },
        resnet=SimpleNamespace(num_layers=(3, 4, 9), width_factor=1),
        classifier="seg",
        decoder_channels=(256, 128, 64, 16),
        skip_channels=[512, 256, 64, 0],
        n_skip=3,
        n_classes=out_channels,
    )


class TransUNet(VisionTransformer):
    """Hybrid TransUNet returning full-resolution raw logits.

    ``img_size`` sets the learned positional grid; input sizes may differ.
    Nonmultiples of 16 are edge-padded and the resulting logits are cropped
    back to the input extent. Images are never resized. RGB is the normal
    input; single-channel inputs are repeated to RGB as in the reference.

    ``pretrained=True`` (the default) initializes both the hybrid ResNet and ViT
    from the joint ImageNet-21k checkpoint. Set ``pretrained_path`` to a local
    R50+ViT-B_16 .npz file, or leave it None to download/cache the official file.
    ``pretrained=False`` ignores ``pretrained_path`` so the existing resume
    and inference code can restore saved model states without that file.
    """

    def __init__(self, in_channels=3, out_channels=1, img_size=256,
                 pretrained=True, pretrained_path=None):
        if isinstance(in_channels, bool) or in_channels not in (1, 3):
            raise ValueError("TransUNet supports in_channels=3 (RGB) or 1 (repeated to RGB)")
        if isinstance(out_channels, bool) or not isinstance(out_channels, int) or out_channels < 1:
            raise ValueError("out_channels must be a positive integer")
        if (isinstance(img_size, bool) or not isinstance(img_size, int)
                or img_size < 32 or img_size % 16):
            raise ValueError("img_size must be an integer >= 32 and a multiple of 16")
        if not isinstance(pretrained, bool):
            raise ValueError("pretrained must be true or false")
        path = _pretrained_weights_path(pretrained_path) if pretrained else None

        super().__init__(_make_config(img_size, out_channels), img_size=img_size,
                         num_classes=out_channels)
        self.in_channels = in_channels
        if path is not None:
            with np.load(path, allow_pickle=False) as weights:
                self.load_from(weights)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"TransUNet expects [B, {self.in_channels}, H, W] input")
        height, width = x.shape[-2:]
        if min(height, width) < 32:
            raise ValueError("TransUNet requires image dimensions of at least 32")
        pad_h, pad_w = (-height) % 16, (-width) % 16
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        logits = super().forward(x)
        return logits[..., :height, :width]  # The shared pipeline applies sigmoid.


def build_model(in_channels=3, out_channels=1, img_size=256,
                pretrained=True, pretrained_path=None) -> nn.Module:
    return TransUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
    )
