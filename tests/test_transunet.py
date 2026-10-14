import pytest
import torch

from methods import available_methods, create_model
from methods import transunet
from methods.transunet._modeling import _prepare_position_embeddings
from roadseg.config import load_config
from roadseg.losses import BCEDiceLoss
from roadseg.tiling import predict_tiled
from roadseg.utils import atomic_save, load_checkpoint, model_from_checkpoint


@pytest.fixture
def small_transunet(monkeypatch):
    """Exercise the same hybrid path with fewer channels/blocks for CPU tests."""
    make_config = transunet._make_config

    def small_config(img_size, out_channels):
        config = make_config(img_size, out_channels)
        config.hidden_size = 32
        config.transformer.update(mlp_dim=64, num_heads=4, num_layers=2)
        config.resnet.num_layers = (1, 1, 1)
        config.resnet.width_factor = 0.5
        config.skip_channels = [256, 128, 32, 0]
        config.decoder_channels = (32, 16, 8, 4)
        return config

    monkeypatch.setattr(transunet, "_make_config", small_config)


def test_transunet_default_architecture_and_256_forward():
    assert "transunet" in available_methods()
    assert "_modeling" not in available_methods()
    config = load_config("configs/transunet.yaml")
    model = create_model("transunet", **config["method"]["params"]).eval()
    encoder = model.transformer
    assert len(encoder.encoder.layer) == 12
    assert encoder.embeddings.position_embeddings.shape == (1, 256, 768)
    assert [len(block) for block in encoder.embeddings.hybrid_model.body] == [3, 4, 9]
    assert model.config.n_skip == 3
    assert model.config.decoder_channels == (256, 128, 64, 16)
    with torch.no_grad():
        output = model(torch.randn(1, 3, 256, 256))
    assert output.shape == (1, 1, 256, 256)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("height,width", [(256, 256), (65, 97), (32, 32), (48, 80)])
def test_transunet_sizes_and_raw_logits(small_transunet, height, width):
    model = create_model("transunet").eval()
    # Negative logits detect accidental sigmoid, softmax or ReLU on the output.
    with torch.no_grad():
        model.segmentation_head[0].weight.zero_()
        model.segmentation_head[0].bias.fill_(-2)
        output = model(torch.randn(1, 3, height, width))
    assert output.shape == (1, 1, height, width)
    torch.testing.assert_close(output, torch.full_like(output, -2))


def test_transunet_backward_reaches_cnn_transformer_and_decoder(small_transunet):
    model = create_model("transunet")
    output = model(torch.randn(2, 3, 65, 97))
    loss = BCEDiceLoss()(output, torch.zeros_like(output))
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for parameter in (
        model.transformer.embeddings.hybrid_model.root.conv.weight,
        model.transformer.embeddings.position_embeddings,
        model.transformer.encoder.layer[0].attn.query.weight,
        model.decoder.blocks[0].conv1[0].weight,
        model.segmentation_head[0].weight,
    ):
        assert parameter.grad.abs().sum() > 0


def test_transunet_checkpoint_does_not_need_pretrained_file(small_transunet, tmp_path):
    config = load_config("configs/transunet.yaml")
    config["method"]["params"].update(
        pretrained=True, pretrained_path=str(tmp_path / "unavailable.npz")
    )
    params = dict(config["method"]["params"], pretrained=False)
    model = create_model("transunet", **params).eval()
    image = torch.randn(1, 3, 65, 97)
    with torch.no_grad():
        expected = model(image)
    checkpoint_path = tmp_path / "model.pt"
    atomic_save({"format_version": 1, "config": config, "model": model.state_dict()},
                checkpoint_path)
    restored = model_from_checkpoint(load_checkpoint(checkpoint_path), "cpu", "transunet")
    with torch.no_grad():
        torch.testing.assert_close(restored(image), expected)


def test_transunet_tiled_inference(small_transunet):
    model = create_model("transunet").train()
    probabilities = predict_tiled(model, torch.randn(3, 79, 113), "cpu",
                                  patch_size=64, overlap=0.5, batch_size=2)
    assert probabilities.shape == (79, 113)
    assert torch.isfinite(probabilities).all()
    assert probabilities.min() >= 0 and probabilities.max() <= 1
    assert model.training


def test_transunet_single_channel_and_multiple_output_channels(small_transunet):
    model = create_model("transunet", in_channels=1, out_channels=2).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 48, 64))
    assert output.shape == (1, 2, 48, 64)
    with pytest.raises(ValueError, match="expects"):
        model(torch.randn(1, 3, 48, 64))
    with pytest.raises(ValueError, match="at least 32"):
        model(torch.randn(1, 1, 16, 16))


@pytest.mark.parametrize("with_class_token", [False, True])
def test_pretrained_positions_remove_class_token_and_resize(with_class_token):
    positions = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
    if with_class_token:
        positions = torch.cat([torch.full((1, 1, 1), -100.0), positions], dim=1)
    resized = _prepare_position_embeddings(positions, (3, 5), 1)
    expected = torch.tensor([
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [1.0, 1.25, 1.5, 1.75, 2.0],
        [2.0, 2.25, 2.5, 2.75, 3.0],
    ])
    torch.testing.assert_close(resized.reshape(3, 5), expected)


def test_pretrained_positions_reject_invalid_grid():
    with pytest.raises(ValueError, match="square grid"):
        _prepare_position_embeddings(torch.zeros(1, 7, 4), (2, 2), 4)


def test_transunet_pretrained_requires_explicit_existing_file(tmp_path):
    with pytest.raises(ValueError, match="pretrained_path"):
        create_model("transunet", pretrained=True)
    with pytest.raises(FileNotFoundError, match="pretrained weights"):
        create_model("transunet", pretrained=True, pretrained_path=str(tmp_path / "missing.npz"))


@pytest.mark.parametrize("params,match", [
    ({"in_channels": 2}, "in_channels"),
    ({"out_channels": 0}, "out_channels"),
    ({"img_size": 255}, "img_size"),
    ({"pretrained": "false"}, "pretrained"),
])
def test_transunet_rejects_invalid_constructor_parameters(params, match):
    with pytest.raises(ValueError, match=match):
        create_model("transunet", **params)


def test_transunet_config_preserves_experiment_settings():
    baseline = load_config("configs/unet.yaml")
    config = load_config("configs/transunet.yaml")
    assert config["method"]["name"] == "transunet"
    assert config["method"]["params"]["pretrained"] is False
    assert config["method"]["params"]["img_size"] == 256
    assert config["train"]["output_dir"] == "runs/transunet"
    config["method"] = baseline["method"]
    config["train"]["output_dir"] = baseline["train"]["output_dir"]
    assert config == baseline
