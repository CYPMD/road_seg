import numpy as np
import pytest
import torch
from torch import nn

from roadseg.tiling import tile_starts, predict_tiled


class Pointwise(nn.Module):
    def forward(self, x):
        return x[:, :1]


class TileMean(nn.Module):
    def forward(self, x):
        return x[:, :1].mean(dim=(2, 3), keepdim=True).expand(-1, 1, x.shape[-2], x.shape[-1])


def test_1024_uses_exactly_49_patches():
    starts = tile_starts(1024, 256, 128)
    assert starts == [0, 128, 256, 384, 512, 640, 768]
    assert len(starts) ** 2 == 49


@pytest.mark.parametrize("height,width", [(1024, 1024), (319, 411), (73, 99), (73, 511)])
def test_reconstruction_and_border_coverage(height, width):
    image = torch.randn(3, height, width)
    model = Pointwise().train()
    output = predict_tiled(model, image, "cpu", patch_size=256, overlap=0.5, batch_size=5)
    assert output.shape == (height, width)
    assert model.training  # The helper preserves caller mode.
    torch.testing.assert_close(output, torch.sigmoid(image[0]))


def test_average_probabilities_before_thresholding():
    # Tiles produce different predictions; pointwise identity alone cannot test averaging.
    image = torch.linspace(-5, 8, 96).view(1, 1, 96).expand(3, 80, 96).clone()
    actual = predict_tiled(TileMean(), image, "cpu", patch_size=64, overlap=0.5, batch_size=3)
    total, counts = np.zeros((80, 96)), np.zeros((80, 96))
    for y in (0, 16):
        for x in (0, 32):
            value = image[0, y:y + 64, x:x + 64].mean().item()
            total[y:y + 64, x:x + 64] += 1 / (1 + np.exp(-value))
            counts[y:y + 64, x:x + 64] += 1
    np.testing.assert_allclose(actual.numpy(), total / counts, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("overlap", [-0.1, 1.0, 0.999, 0.333])
def test_invalid_overlap_rejected(overlap):
    with pytest.raises(ValueError):
        predict_tiled(Pointwise(), torch.zeros(3, 256, 256), "cpu", overlap=overlap)


def test_wrong_method_output_rejected():
    with pytest.raises(ValueError, match="raw logits"):
        predict_tiled(nn.Identity(), torch.zeros(3, 256, 256), "cpu")
