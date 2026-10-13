# Verification record

Verified on 2026-10-13, on CPU, with:

| Component | Version |
| --- | --- |
| Python | 3.12.14 |
| PyTorch | 2.6.0+cpu |
| NumPy | 2.3.5 |
| Pillow | 12.3.0 |
| PyYAML | 6.0.3 |
| tqdm | 4.70.1 |

`python -m pytest -q`: **18 passed**.

Checks cover:

- Exact 7-by-7 tile grid for 1024px images with 256px patches and stride 128.
- Probability averaging when overlapping tiles produce different predictions.
- Complete reconstruction for 1024px, irregular, and smaller-than-patch images.
- Invalid overlap and incompatible model-output rejection.
- Matching 256px image/mask crops after flips and rotations; fresh crops by epoch.
- Reproducible image-level train/validation splits and saved split round-tripping.
- Separate nested image/mask folders and explicit errors for missing labels.
- U-Net forward/backward passes with odd spatial dimensions.
- Foreground metrics and empty-mask conventions.

`python scripts/smoke_test.py --work-dir <new-directory>`: **passed**.

The smoke test used four generated 1024-by-1024 image/mask pairs. It ran the real
command-line entry points to train a base-width-4 U-Net, resume training for a
second epoch, evaluate the saved validation split, and perform 256px / 50%-overlap
inference. It verified 1024-by-1024 binary PNG and float32 probability outputs,
two-epoch history, and the saved three-image/one-image train/validation split.

The host emitted CPU/NNPACK capability warnings and completed using the available
CPU kernels. All train, resume, evaluate and infer processes exited successfully.

The actual DeepGlobe dataset was not available in this workspace. No DeepGlobe
accuracy, GPU performance, or convergence result is claimed. CUDA mixed
precision and MPS execution were not tested. Dependency ranges in requirements
are intended compatibility ranges, not a claim that every version was tested.
