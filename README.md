# DeepGlobe road segmentation

A small PyTorch experiment repo with **one included method: U-Net**. Data loading,
loss, training, evaluation, checkpointing and inference are shared. Add a model
under `methods/` to test another architecture without changing the pipeline.

- Training: fresh random **256 × 256 crops** from the original images, with paired
  flips and 90° rotations. Images are never resized to 256 × 256.
- Inference and validation: **256 × 256 patches, 50% overlap, stride 128**.
  Sigmoid probabilities are averaged at overlapping pixels, then thresholded.
- A 1024 × 1024 image produces **7 × 7 = 49 patches** and a 1024 × 1024 output.
- Default objective: `0.5 * BCEWithLogits + 0.5 * soft Dice loss`.
- Checkpoints, resume, full-image IoU/Dice metrics and CSV history are included.

## 1. Install

Use Python 3.10 or newer. Run commands from this repository's root.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

Install a PyTorch build appropriate for your GPU/CPU using the
[official installation selector](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements.txt
```

For a CPU-only environment you can install Torch first with:

```bash
python -m pip install 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The plain U-Net uses PyTorch only. The ResNet-34 U-Net uses torchvision and, when
`pretrained: true`, downloads the official ImageNet-1K ResNet-34 weights on the
first fresh training run. The plain U-Net enables CUDA mixed precision by default.
ResNet-34 U-Net defaults to float32 for numerical stability. MPS and CPU use
float32. Override training precision with `--amp` or `--no-amp`.

## 2. Point the config at your dataset

The default layout expects these paths:

| Example path under the dataset root | Purpose |
| --- | --- |
| `train/100_sat.jpg` | RGB satellite image |
| `train/100_mask.png` | Binary mask for that image |
| `train/101_sat.jpg` | Next image |
| `train/101_mask.png` | Next mask |
| `valid/200_sat.jpg` or `test/300_sat.jpg` | Optional images for inference |

Set `data.root` in `configs/unet.yaml`, or pass `--data-root` at training time.
Only `train/` is used by the default configuration. It holds out 15% of labeled
images for validation using seed 42, **before any crops are sampled**. It does not
assume that your downloaded `valid/` or `test/` directories have ground truth.

By default, pixels with mask value >=128 are roads. White is road, black is
background. RGB copies of a grayscale mask also work. If masks contain literal
0/1 values, set `mask_threshold: 1`. Other class encodings must first be converted
to binary masks. Images and masks must have identical spatial dimensions.

Alternative directory arrangements are configuration changes:

| Your layout | Configuration |
| --- | --- |
| Pairs directly inside the root | `train_images: .`, `train_masks: null` |
| Separate `images/` and `masks/` | `train_images: images`, `train_masks: masks` |
| Labeled, separate validation folder | `val_images: valid`, `val_masks: null` |
| Separate validation image/mask folders | `val_images: valid/images`, `val_masks: valid/masks` |
| Different naming | Change `image_suffix` and `mask_suffix` |

All directories must be inside `data.root`. Pairing is recursive and preserves
relative subdirectories. For example, `images/region/a_sat.jpg` pairs with
`masks/region/a_mask.png`. A missing mask fails early instead of silently dropping
the image. When specifying a labeled validation folder, every image needs a mask.

The image split is saved in `splits.json` and the checkpoints. With unchanged
data, seed and config, new architecture runs receive the same image split and
crop sequence. This is an image-level random split, not a geographic split;
provide separate train/validation folders if nearby scenes must be kept apart.

## 3. Train

```bash
python train.py --config configs/unet.yaml --method unet \
  --data-root /absolute/path/to/deepglobe --output-dir runs/unet
```

Train the ResNet-34 U-Net with ImageNet initialization:

```bash
python train.py --config configs/unet_resnet34.yaml --method unet_resnet34 \
  --data-root /absolute/path/to/deepglobe --output-dir runs/unet_resnet34
```

The ResNet-34 config uses ImageNet mean/std normalization and sets `pretrained: true`.
Resume and checkpoint-based inference do not download ImageNet weights again; the
saved model state is restored directly.

If a previous run failed with nonfinite validation probabilities, install the
updated files and start with `--no-amp` in a **new** output directory. A finite
training loss alone does not prove that BatchNorm running statistics are valid.
The updated trainer checks model parameters/buffers and unscaled gradients;
invalid full-precision gradients are rejected before the optimizer update.
Valid checkpoints can also be resumed with `--no-amp`, which overrides their
saved AMP setting. See [NUMERICAL_FIX.md](NUMERICAL_FIX.md) for Colab commands.

The default U-Net uses four downsampling stages and base width 32, with batch
normalization, transpose-convolution upsampling and skip connections. It trains
from scratch. Set `base_channels: 64` for a wider model, or lower the batch size
if memory is limited.

Each image supplies `crops_per_image` independent random crops per epoch (default
4). An epoch therefore contains `number_of_training_images * crops_per_image`
samples. Coordinates are uniformly random; roads are not oversampled, so some
crops contain no roads. New coordinates and augmentations are sampled each epoch.

Images are converted to RGB float32 in [0,1], then normalized with the configured
mean/std (default 0.5/0.5, yielding [-1,1]). The same normalization is restored from
the checkpoint for validation and inference. Mask labels are never interpolated.

AdamW optimizes the shared BCE/Dice objective. `ReduceLROnPlateau` halves the
learning rate after its five-epoch patience on validation IoU. Validation runs on
all held-out full images after each epoch, using the exact tiled inference path.
This can take substantial time because each 1024px image requires 49 patches.
Adjust `inference.batch_size` to fit available memory.

| File inside `runs/unet/` | Contents |
| --- | --- |
| `best.pt` | Checkpoint with highest pixel-pooled validation road IoU |
| `last.pt` | Most recent completed epoch, including optimizer/scheduler/RNG state |
| `config.yaml` | Resolved settings for the run |
| `splits.json` | Train/validation image and mask paths, relative to data root |
| `history.csv` | Loss, learning rate, validation metrics and epoch time |

Existing nonempty output directories are protected from accidental overwriting.
Resume from the run's latest checkpoint:

```bash
python train.py --resume runs/unet/last.pt --epochs 100
```

`--epochs` is the total target epoch count. Resume uses the saved config and image
split, and writes to the checkpoint's directory. Keep `best.pt` and `last.pt`
together when moving a run. Use `--data-root` if the dataset moved. A supplied
resume config may change total epochs, data root, device, worker count or AMP;
changes to experiment settings are rejected. It resumes at an epoch boundary,
not mid-epoch. Reproducibility across different hardware/library versions is not
guaranteed.

## 4. Infer

```bash
python infer.py --checkpoint runs/unet/best.pt \
  --input /absolute/path/to/deepglobe/test --output-dir predictions/unet
```

Or predict one image and also save its probability map:

```bash
python infer.py --checkpoint runs/unet/best.pt \
  --input /absolute/path/to/deepglobe/test/300_sat.jpg \
  --output-dir predictions/single --save-probabilities
```

The checkpoint chooses the method and its constructor parameters automatically.
You can pass `--method unet` to verify that the checkpoint has the expected method.
Loading a checkpoint with a different architecture is rejected.

Inference uses the checkpoint's patch size, overlap, batch size, threshold and
normalization. The first four can be overridden, for example:

```bash
python infer.py --checkpoint runs/unet/best.pt \
  --input /absolute/path/to/deepglobe/test --output-dir predictions/unet \
  --patch-size 256 --overlap 0.5 --batch-size 8 --threshold 0.5
```

At the defaults, tile origins along each axis are 0, 128, 256, 384, 512, 640 and
768 for a 1024px image. The code applies sigmoid to each tile's logits, sums the
probabilities at each covered pixel, divides by the number of contributing
tiles, and only then applies the threshold. It does not average binary masks.
If AMP produces nonfinite logits, that tile batch is recomputed in float32 and
the remaining tiles of the image use float32. Invalid float32 outputs still
raise an error; values are never replaced with fabricated probabilities.

For arbitrary image dimensions, an additional tile is anchored to each final
edge if needed; its overlap can exceed 50%. Images smaller than a patch use edge
padding, which is removed after prediction. Every output retains the original
image dimensions. Only a tile batch is moved to the GPU; stitching uses CPU RAM.

Outputs are `<id>_mask.png` (uint8, 0 background / 255 road) and optionally
`<id>_prob.npy` (float32 [H,W], values in [0,1]). Nested input folders are preserved.
Directory inputs match `data.image_suffix`; use `--image-suffix .png` for other
image naming. A single input file need not match the suffix. Keep output outside
the input directory. Existing predictions require explicit `--overwrite`.

## 5. Evaluate

Evaluate the exact saved validation split:

```bash
python evaluate.py --checkpoint runs/unet/best.pt \
  --data-root /absolute/path/to/deepglobe
```

Evaluate another labeled directory, using the checkpoint's filename suffixes:

```bash
python evaluate.py --checkpoint runs/unet/best.pt \
  --images /absolute/path/to/held-out/images \
  --masks /absolute/path/to/held-out/masks
```

The JSON result contains foreground IoU, Dice/F1, precision, recall, pixel
accuracy and mean per-image IoU. All except `mean_image_iou` pool pixel counts
across full images. `iou` selects the best checkpoint. Empty prediction and
empty ground truth yield IoU/Dice 1. An empty prediction with nonempty ground
truth has precision 0; recall is 1 for empty ground truth. Threshold defaults to
0.5. These metrics are explicitly defined here; this is not an implementation
of a challenge submission or leaderboard evaluator.

## 6. Add an architecture

Add `methods/my_method.py`, or a `methods/my_method/` package. Export:

```python
def build_model(in_channels=3, out_channels=1, width=32):
    return MyModel(in_channels=in_channels, out_channels=out_channels, width=width)
```

Define or import your `MyModel` class in that module. The shared interface is:

| Item | Required contract |
| --- | --- |
| Factory | `build_model(**params)` returns `torch.nn.Module` |
| Forward input | Float tensor `[B, 3, H, W]`, already normalized |
| Forward output | One tensor of **raw logits** `[B, 1, H, W]` |
| Spatial dimensions | Output matches input, including 256 × 256 patches |
| Probability conversion | Handled centrally; do **not** apply sigmoid in the model |

Models with dictionaries, auxiliary heads or different output resolutions need
a small adapter inside their method module to satisfy this contract. Keep model
helper modules private (prefix `_`) or inside the method's package. Public method
modules are discovered automatically; no registry edit is required.

Copy `configs/unet.yaml` to `configs/my_method.yaml` and replace the method block:

```yaml
method:
  name: my_method
  params:
    in_channels: 3
    out_channels: 1
    width: 32
```

Keep the other experiment settings fixed for architecture comparisons, then:

```bash
python train.py --config configs/my_method.yaml --method my_method \
  --data-root /absolute/path/to/deepglobe --output-dir runs/my_method

python infer.py --checkpoint runs/my_method/best.pt \
  --input /absolute/path/to/deepglobe/test --output-dir predictions/my_method
```

The only bundled implementation is U-Net. Architectures requiring different
losses or training strategies would additionally need deliberate changes to the
shared pipeline.

## Code map

| Path | Responsibility |
| --- | --- |
| `methods/` | Model discovery and U-Net architecture |
| `roadseg/data.py` | Pairing, image splits, normalization and random crops |
| `roadseg/tiling.py` | Patch batching and overlap probability averaging |
| `roadseg/engine.py` | Shared train/evaluation loops |
| `roadseg/losses.py`, `roadseg/metrics.py` | Binary objective and metrics |
| `roadseg/config.py`, `roadseg/utils.py` | Config validation, device and checkpoints |
| `train.py`, `infer.py`, `evaluate.py` | Command-line entry points |
| `configs/unet.yaml` | All default settings |
| `tests/`, `scripts/smoke_test.py` | Focused tests and an end-to-end synthetic check |

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/smoke_test.py
```

The smoke script creates four synthetic 1024 × 1024 image/mask pairs, trains a
small U-Net on random 256 × 256 crops, resumes for a second epoch, evaluates full
images, and saves a prediction using 49 overlapping patches. It checks mask and
probability dimensions/ranges. Its output directory must be empty; use a new
`--work-dir` for another run.

See `VERIFICATION.md` for the actual environment and checks performed. These
checks verify execution and numerical stitching, not segmentation quality on
your downloaded DeepGlobe data. No dataset or pretrained weights are included.

## References

- [DeepGlobe 2018 dataset/challenge paper](https://arxiv.org/abs/1805.06561)
- [U-Net paper](https://arxiv.org/abs/1505.04597)
- [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch checkpoint loading](https://docs.pytorch.org/docs/stable/generated/torch.load.html)

The code is MIT-licensed. Dataset terms remain those of your dataset provider.
