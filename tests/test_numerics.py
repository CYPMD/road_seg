"""Regression checks for overflow, failed gradients and invalid model buffers."""

import math

import pytest
import torch
from torch import nn

from roadseg.engine import train_one_epoch
from roadseg.losses import BCEDiceLoss
from roadseg.tiling import _predict_tile_batch, predict_tiled
from roadseg.utils import check_model_finite


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_amp_nonfinite_logits_retry_the_forward_pass_in_float32(bad_value):
    # Simulate an AMP-only failure using the real autocast context. This keeps
    # control-flow coverage available on hosts without FP16 CPU/CUDA kernels.
    calls = []

    class AutocastFailure(nn.Module):
        def forward(self, inputs):
            enabled = torch.is_autocast_enabled(inputs.device.type)
            calls.append(enabled)
            return torch.full_like(inputs[:, :1], bad_value) if enabled else inputs[:, :1]

    model = AutocastFailure().eval()
    inputs = torch.ones(2, 3, 32, 32)
    with torch.inference_mode(), pytest.warns(RuntimeWarning, match="retried.*float32"):
        probabilities, use_amp = _predict_tile_batch(model, inputs, True)
    assert calls == [True, False]
    assert use_amp is False
    assert probabilities.dtype == torch.float32
    assert probabilities.device.type == "cpu"
    torch.testing.assert_close(probabilities, torch.sigmoid(inputs[:, 0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_tiling_falls_back_for_remaining_batches_and_preserves_mode():
    model = nn.Conv2d(3, 1, 1, bias=False).cuda().train()
    with torch.no_grad():
        model.weight.fill_(300)
    dtypes = []
    hook = model.register_forward_hook(lambda module, args, result: dtypes.append(result.dtype))
    with pytest.warns(RuntimeWarning, match="retried.*float32") as caught:
        probabilities = predict_tiled(
            model, torch.full((3, 48, 48), 100.), "cuda",
            patch_size=32, overlap=0.5, batch_size=1, amp=True,
        )
    hook.remove()
    assert len(caught) == 1
    assert dtypes == [torch.float16] + [torch.float32] * 4
    assert model.training
    torch.testing.assert_close(probabilities, torch.ones(48, 48))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_float32_logits_still_raise_and_restore_model_mode(bad_value):
    class InvalidOutput(nn.Module):
        def forward(self, inputs):
            return torch.full_like(inputs[:, :1], bad_value)

    model = InvalidOutput().train()
    with pytest.raises(FloatingPointError, match="nonfinite logits even in float32"):
        predict_tiled(model, torch.zeros(3, 32, 32), "cpu", patch_size=32)
    assert model.training


def test_failed_amp_retry_does_not_fabricate_probabilities():
    class AlwaysInvalid(nn.Module):
        def forward(self, inputs):
            return inputs[:, :1] * float("nan")

    with torch.inference_mode(), pytest.raises(FloatingPointError, match="even in float32"):
        _predict_tile_batch(AlwaysInvalid(), torch.ones(1, 3, 32, 32), True)


def test_nonfinite_input_is_rejected_before_forward():
    class MustNotRun(nn.Module):
        def forward(self, inputs):
            pytest.fail("invalid input reached the model")

    image = torch.zeros(3, 32, 32)
    image[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="Input image"):
        predict_tiled(MustNotRun(), image, "cpu", patch_size=32)


@pytest.mark.parametrize("name", ["weight", "running_mean", "running_var"])
def test_invalid_model_state_is_named_and_not_repaired(name):
    model = nn.BatchNorm2d(3)
    with torch.no_grad():
        getattr(model, name)[0] = float("nan")
    with pytest.raises(FloatingPointError, match=name):
        check_model_finite(model, "test checkpoint")
    assert torch.isnan(getattr(model, name)[0])


def _loader(batches=1):
    return [(torch.ones(2, 3, 32, 32), torch.zeros(2, 1, 32, 32))
            for _ in range(batches)]


@pytest.mark.parametrize("grad_clip_norm", [0.0, 1.0])
def test_fp32_nonfinite_gradients_do_not_update_parameters(grad_clip_norm):
    model = nn.Conv2d(3, 1, 1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with pytest.raises(FloatingPointError, match="Nonfinite gradients"):
        train_one_epoch(model, _loader(), optimizer, scaler, BCEDiceLoss(),
                        torch.device("cpu"), False, grad_clip_norm)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


@pytest.mark.parametrize("grad_clip_norm", [0.0, 1.0])
def test_grad_scaler_skips_bad_gradients_and_continues(grad_clip_norm):
    torch.manual_seed(10)
    model = nn.Conv2d(3, 1, 1)
    reference = nn.Conv2d(3, 1, 1)
    reference.load_state_dict(model.state_dict())
    calls = []

    def overflow_first_gradient(grad):
        calls.append(True)
        return torch.full_like(grad, float("inf")) if len(calls) == 1 else grad

    model.weight.register_hook(overflow_first_gradient)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # CPU GradScaler exercises the real skip/update logic without CUDA.
    scaler = torch.amp.GradScaler("cpu", init_scale=16)
    with pytest.warns(RuntimeWarning, match="skipped 1 training batches"):
        loss = train_one_epoch(model, _loader(2), optimizer, scaler, BCEDiceLoss(),
                               torch.device("cpu"), False, grad_clip_norm)
    reference_loss = train_one_epoch(
        reference, _loader(), torch.optim.SGD(reference.parameters(), lr=0.1),
        torch.amp.GradScaler("cuda", enabled=False), BCEDiceLoss(),
        torch.device("cpu"), False, grad_clip_norm,
    )
    assert math.isfinite(loss) and loss == pytest.approx(reference_loss)
    assert scaler.get_scale() == 8
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)


def test_all_skipped_batches_cannot_report_a_successful_epoch():
    model = nn.Conv2d(3, 1, 1)
    model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    with pytest.raises(FloatingPointError, match="No optimizer updates"):
        train_one_epoch(
            model, _loader(), torch.optim.SGD(model.parameters(), lr=0.1),
            torch.amp.GradScaler("cpu", init_scale=16), BCEDiceLoss(),
            torch.device("cpu"), False, 1.0,
        )


def test_finite_loss_does_not_hide_corrupted_running_statistics():
    class BadRunningStatistics(nn.Conv2d):
        def __init__(self):
            super().__init__(3, 1, 1)
            self.register_buffer("running_var", torch.ones(1))

        def forward(self, inputs):
            self.running_var.fill_(float("inf"))
            return super().forward(inputs)  # Finite outputs/loss/gradients.

    model = BadRunningStatistics()
    with pytest.raises(FloatingPointError, match="before validation:.*running_var"):
        train_one_epoch(
            model, _loader(), torch.optim.SGD(model.parameters(), lr=0.1),
            torch.amp.GradScaler("cuda", enabled=False), BCEDiceLoss(),
            torch.device("cpu"), False, 1.0,
        )
