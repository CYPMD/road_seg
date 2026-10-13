# Fix for nonfinite validation probabilities

This package contains updated files for the project in your attached source
export. Extract it into your existing repository, preserving the directories.

## Run in Colab

Upload `road_seg_updated_files.zip` to `/content`, then run:

```python
%cd /content/road_seg
!unzip -o /content/road_seg_updated_files.zip -d /content/road_seg
!python train.py --config configs/unet_resnet34.yaml --method unet_resnet34 \
    --data-root /content/deepglobe-road-extraction-dataset \
    --output-dir runs/unet_resnet34_fp32 --no-amp
```

Use a **new output directory**. Your failed run may already contain `config.yaml`
and `splits.json`, even if validation failed before any checkpoint was written.
The existing protection against overwriting a run remains in place.

Full precision still uses your CUDA GPU when available. It can use more memory
and run more slowly than AMP. If memory is insufficient, reduce
`train.batch_size` and/or `inference.batch_size` in the YAML before starting.

To continue a known-good completed checkpoint instead:

```python
!python train.py --resume runs/unet_resnet34/last.pt \
    --data-root /content/deepglobe-road-extraction-dataset \
    --epochs 50 --no-amp
```

`--epochs` is the total desired epoch count. Resume requires both `best.pt` and
`last.pt` in that run directory. The command-line precision setting overrides
the checkpoint's saved `train.amp` value. A checkpoint with NaN/Inf parameters
or buffers is rejected; changing precision cannot repair an already-invalid
model state.

## What caused the exception?

The attached trainer passes its AMP setting to validation. On CUDA, the old
inference path performed the model forward pass in float16 and stopped when
the probabilities were nonfinite, without attempting float32 inference.

The supplied traceback establishes invalid validation outputs. It does not
identify the first failing layer or distinguish inference overflow from
invalid model parameters or BatchNorm running statistics. The checkpoint,
Colab GPU, and dataset were not available here, so that exact origin is not
claimed as reproduced. AMP instability is a plausible cause addressed by this
patch. The original BCE/Dice loss already uses float32 reductions.

## Changes

| File | Change |
| --- | --- |
| `configs/unet_resnet34.yaml` | Defaults to `train.amp: false`; learning rate, batch sizes, architecture, loss weights, and data split settings are retained. |
| `train.py` | Adds `--amp` / `--no-amp` overrides for fresh runs and resume. |
| `roadseg/tiling.py` | Rejects nonfinite inputs; checks logits before sigmoid; reruns a failed AMP tile batch in float32 and uses float32 for the rest of that image. A failed float32 pass still raises. |
| `roadseg/engine.py` | Keeps the loss outside autocast, validates training inputs/logits and unscaled gradients, preserves GradScaler's skipped-update behavior, and checks model state before training and before validation. Validation errors include the image path. |
| `roadseg/utils.py` | Checks parameters and buffers, including BatchNorm statistics, with named diagnostics; validates loaded inference checkpoints. |
| `tests/test_numerics.py` | Regression coverage for AMP retry, NaN/Inf rejection, invalid buffers, and optimizer protection. Includes a CUDA-only integration test. |
| `tests/test_unet_resnet34.py` | Updates the config assertion for the safer precision default. |
| `README.md`, `VERIFICATION.md`, `NUMERICAL_FIX.md` | Usage and verification details. |

The patch does not replace NaNs with zeros, reset BatchNorm statistics, or
silently discard invalid validation predictions. These would make the metrics
unreliable. Training batches with overflowing scaled gradients follow
GradScaler's normal skip-and-reduce-scale behavior; an epoch with no completed
updates fails explicitly.

## Verification

- `python -m pytest -q`: **42 passed, 1 skipped**. The skipped test requires CUDA.
- The real CLI smoke test passed: train, checkpoint, resume, full-image
  evaluation, and 49-tile inference with saved PNG/NumPy outputs.
- A separate ResNet-34 U-Net CLI run trained for one epoch and validated a
  1024-by-1024 image with 49 tiles. It used a reduced decoder width, 64px
  training crops, generated data, CPU execution, and `pretrained: false`.
  `--no-amp` correctly overrode a YAML setting of `amp: true`.
- AMP-only NaN/Inf outputs were simulated under a real autocast context to
  check that the model forward pass is retried with autocast disabled.
- This host cannot initialize float16 CPU kernels. Real CUDA/float16 execution,
  ImageNet weight downloads, and a complete DeepGlobe training run were not
  tested. No convergence or accuracy claim is made.

Test environment: Python 3.12.14, PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu,
NumPy 2.5.2, Pillow 12.3.0, PyYAML 6.0.3, tqdm 4.70.1, pytest 9.1.1.
The user's Python 3.13 Colab environment was not available for verification.

## PyTorch references

- [AMP numerical range and gradient scaling](https://docs.pytorch.org/docs/stable/amp.html#gradient-scaling)
- [Unscale before inspecting or clipping gradients](https://docs.pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping)

The fallback follows PyTorch's guidance to run problematic operations with
autocast disabled and float32 inputs. Gradient checks occur after unscaling,
before clipping and optimizer updates.
