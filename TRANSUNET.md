# TransUNet with pretrained ResNet and ViT

Extract the accompanying ZIP into the existing `road_seg` repository root,
preserving its directories. The archive contains the complete TransUNet addition.
Relative to the first archive, this revision changes only the TransUNet factory,
its config, its tests and this document. The original project's files, existing
methods, dependencies and shared pipeline code remain unchanged.

## Train

Run from the repository root with the existing requirements installed:

```bash
python train.py --config configs/transunet.yaml --method transunet \
  --data-root /absolute/path/to/deepglobe --output-dir runs/transunet
```

**Both the hybrid ResNet and ViT now start from ImageNet-21k pretrained weights.**
The first fresh run downloads the joint checkpoint (about 461 MB / 440 MiB),
verifies its SHA-256 checksum and caches it in Torch's checkpoint directory.
Subsequent fresh runs reuse that file. The decoder and road-output head start
with random weights; the complete model, including both encoders, is fine-tuned.

Use a new output directory when switching from a scratch-trained run, for example
`--output-dir runs/transunet_pretrained`. Resuming an old run restores its own
weights and does not replace them with ImageNet initialization.

The new config keeps the U-Net baseline's data split, crop sampling, augmentations,
normalization, BCE/Dice objective, AdamW settings, precision setting and tiled
inference settings. Only the method configuration and output directory differ.
Use `--no-amp` if full precision is desired. If GPU memory is insufficient, lower
`train.batch_size` and `inference.batch_size` in the new config before training.

Resume, evaluate and infer using the existing commands:

```bash
python train.py --resume runs/transunet/last.pt --epochs 100
python evaluate.py --checkpoint runs/transunet/best.pt
python infer.py --checkpoint runs/transunet/best.pt \
  --input /absolute/path/to/deepglobe/test --output-dir predictions/transunet
```

## Model and initialization

This is the 2D **R50-ViT-B/16 TransUNet** from the supplied Beckschen repository:
its weight-standardized, GroupNorm ResNetV2 hybrid encoder has bottleneck depths
`(3, 4, 9)`, the transformer has 12 layers, 12 attention heads, hidden width 768
and MLP width 3072, and the cascaded decoder uses three CNN skip connections.
The decoder widths are `(256, 128, 64, 16)`. The binary head returns raw logits
`[B, 1, H, W]`; the existing pipeline handles sigmoid and loss computation.

`img_size: 256` sets a learned 16-by-16 positional grid. Other spatial sizes are
supported by interpolating positional embeddings. Nonmultiples of 16 are padded
at the bottom/right and logits are cropped back to the original extent. Images
and masks are not resized. Input height and width must each be at least 32.

Pretrained initialization is enabled in both `build_model()` and
`configs/transunet.yaml`:

```yaml
    pretrained: true
    pretrained_path: null   # Automatically download/cache the joint checkpoint.
```

The joint **R50+ViT-B_16** checkpoint contains the pretrained hybrid ResNetV2
backbone, patch projection, positional embeddings and all ViT encoder blocks.
It is the matching checkpoint used by the supplied TransUNet implementation.
The supplied 0.5 mean/std normalization maps RGB pixels to [-1, 1].

For offline use, set `pretrained_path` to your local `R50+ViT-B_16.npz` file.
An explicit local path is used directly and must exist. If the download fails,
model creation stops with instructions to supply the file; it never silently
starts from random weights. A copy is linked from the
[TransUNet project README](https://github.com/Beckschen/TransUNet#usage).
Pure ViT files and torchvision ResNet checkpoints are not interchangeable with
this joint hybrid checkpoint.

When restoring a project checkpoint, the existing trainer/inference code sets
`pretrained=False`, so neither a download nor the original `.npz` file is needed.
The pretrained weights are downloaded separately and are not bundled in this ZIP.

## Files

| New file | Purpose |
| --- | --- |
| `methods/transunet/__init__.py` | Pretrained default, verified download/cache, method factory and input adapter |
| `methods/transunet/_modeling.py` | Adapted reference transformer, decoder and weight loader |
| `methods/transunet/_resnet.py` | Adapted reference hybrid ResNetV2 encoder |
| `methods/transunet/LICENSE` | Upstream Apache-2.0 license |
| `configs/transunet.yaml` | TransUNet experiment configuration |
| `tests/test_transunet.py` | Architecture, shapes, gradients, checkpoint and tiling checks |
| `TRANSUNET.md` | Installation, use and verification notes |

Method discovery already supports packages, so `methods/__init__.py` needs no edit.
The adapted implementation uses existing PyTorch and NumPy dependencies; it does
not require `ml_collections`, SciPy or the upstream medical-data training stack.

## Verification

Run the new checks with the existing development requirements installed:

```bash
python -m pytest -q tests/test_transunet.py
```

Unit tests explicitly disable pretraining or mock downloads so they run offline.
The default architecture gets a real 256-by-256 forward check. The gradient,
shape, tiling and checkpoint tests use a smaller hybrid configuration to keep
CPU test costs low; this does not change the model shipped in the YAML config.

Checks performed for this addition on CPU with PyTorch 2.6.0:

- Full project suite: **63 passed, 1 skipped** (the existing CUDA-only test).
- Downloaded the actual official ImageNet-21k checkpoint and verified SHA-256
  `ff009bf39bb4f9198b834cfe46aba2bfdaf730e933ab3e3c4b1edf4226eaafbe`.
  Confirmed exact checkpoint weights in every ResNet block and all 12 ViT
  attention blocks, then ran a finite 256-by-256 forward pass. All model
  parameters remain trainable.
- Verified download caching, clear failure on unavailable weights, and checkpoint
  restoration without accessing either automatic or local initialization files.
- The full 105,322,001-parameter model's state dictionary is compatible with
  the supplied reference; 256-by-256 logits are bitwise identical after loading
  the same weights into both implementations.
- A synthetic Google-format `.npz` initialized the same encoder parameters as
  the upstream loader, including positional-grid resizing.
- The real train, resume, evaluate and infer entry points completed using a
  reduced hybrid model and synthetic data. The initialization `.npz` was deleted
  before resume; checkpoint restoration and saved PNG/probability outputs passed.

CUDA mixed precision and DeepGlobe training quality were not tested in this CPU
environment.

## Attribution

Adapted from the attached [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet)
source, licensed under Apache-2.0. The adapted files carry modification notices;
the upstream license is included in the method package. The existing repository
license is unchanged.

Pretrained hybrid model source:
[Google Research's Vision Transformer repository](https://github.com/google-research/vision_transformer).

Reference: Chen et al., *TransUNet: Transformers Make Strong Encoders for Medical
Image Segmentation*, arXiv:2102.04306v1 (2021), supplied with this request.
