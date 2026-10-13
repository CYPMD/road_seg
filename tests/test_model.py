import pytest
import torch

from methods import create_model, available_methods
from roadseg.losses import BCEDiceLoss
from roadseg.metrics import BinaryMetrics


def test_unet_backward_and_odd_image_size():
    assert "unet" in available_methods()
    model = create_model("unet", base_channels=4)
    output = model(torch.randn(2, 3, 65, 67))
    assert output.shape == (2, 1, 65, 67)
    loss = BCEDiceLoss()(output, torch.zeros_like(output))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="Unknown method"):
        create_model("missing")


def test_metrics_and_empty_foreground():
    metrics = BinaryMetrics()
    metrics.update([[1, 1], [0, 0]], [[1, 0], [1, 0]])
    result = metrics.compute()
    assert result["iou"] == pytest.approx(1 / 3)
    assert result["dice"] == pytest.approx(0.5)
    empty = BinaryMetrics()
    empty.update([[0]], [[0]])
    assert empty.compute()["iou"] == 1
