import pytest
import torch

from methods import available_methods, create_model
from roadseg.config import load_config
from roadseg.losses import BCEDiceLoss
from roadseg.tiling import predict_tiled
from roadseg.utils import atomic_save, load_checkpoint, model_from_checkpoint


@pytest.mark.parametrize("height,width", [(256, 256), (65, 67), (32, 32)])
def test_unet_resnet34_output_size_and_raw_logits(height, width):
    assert "unet_resnet34" in available_methods()
    model = create_model("unet_resnet34").eval()
    # A forced negative head output detects an unwanted sigmoid or ReLU.
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(-2)
        output = model(torch.randn(1, 3, height, width))
    assert output.shape == (1, 1, height, width)
    torch.testing.assert_close(output, torch.full_like(output, -2))


def test_unet_resnet34_backward_reaches_encoder_and_decoder():
    model = create_model("unet_resnet34", base_channels=4)
    output = model(torch.randn(2, 3, 65, 67))
    loss = BCEDiceLoss()(output, torch.zeros_like(output))
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_unet_resnet34_checkpoint_roundtrip(tmp_path):
    config = load_config("configs/unet_resnet34.yaml")
    config["method"]["params"]["base_channels"] = 4
    model = create_model(config["method"]["name"], **config["method"]["params"]).eval()
    image = torch.randn(1, 3, 65, 67)
    with torch.no_grad():
        expected = model(image)
    checkpoint_path = tmp_path / "model.pt"
    atomic_save({"format_version": 1, "config": config, "model": model.state_dict()},
                checkpoint_path)
    restored = model_from_checkpoint(load_checkpoint(checkpoint_path), "cpu", "unet_resnet34")
    with torch.no_grad():
        torch.testing.assert_close(restored(image), expected)


def test_unet_resnet34_tiled_inference():
    model = create_model("unet_resnet34", base_channels=4).train()
    probabilities = predict_tiled(model, torch.randn(3, 79, 97), "cpu",
                                  patch_size=64, overlap=0.5, batch_size=2)
    assert probabilities.shape == (79, 97)
    assert torch.isfinite(probabilities).all()
    assert probabilities.min() >= 0 and probabilities.max() <= 1
    assert model.training


def test_unet_resnet34_config_preserves_experiment_settings():
    baseline = load_config("configs/unet.yaml")
    config = load_config("configs/unet_resnet34.yaml")
    assert config["method"]["name"] == "unet_resnet34"
    assert config["train"]["output_dir"] == "runs/unet_resnet34"
    config["method"]["name"] = baseline["method"]["name"]
    config["train"]["output_dir"] = baseline["train"]["output_dir"]
    assert config == baseline
