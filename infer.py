"""Predict binary road masks from a single image or a directory of images."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from roadseg.config import validate_config
from roadseg.data import read_rgb, image_tensor
from roadseg.tiling import predict_tiled
from roadseg.utils import select_device, load_checkpoint, model_from_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="One RGB image, or a directory searched recursively")
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--method", help="Optional verification; normally read automatically from checkpoint")
    parser.add_argument("--image-suffix", help="For directory inputs; default is checkpoint data.image_suffix")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--save-probabilities", action="store_true", help="Also save float32 probability .npy files")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing prediction files")
    args = parser.parse_args(argv)
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    for key in ("patch_size", "overlap", "batch_size", "threshold"):
        value = getattr(args, key)
        if value is not None:
            config["inference"][key] = value
    validate_config(config)
    settings = config["inference"]
    suffix = args.image_suffix if args.image_suffix is not None else config["data"]["image_suffix"]
    if not suffix:
        raise ValueError("image-suffix cannot be empty")
    device = select_device(args.device)
    model = model_from_checkpoint(checkpoint, device, args.method)
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir() and (output == source or output.is_relative_to(source)):
        raise ValueError("Keep output-dir outside the input directory to avoid reading previous predictions")
    inputs = sorted(p for p in source.rglob(f"*{suffix}") if p.is_file()) if source.is_dir() else [source]
    if not inputs:
        raise ValueError(f"No *{suffix} images found in {source}")
    jobs = []
    destinations = set()
    for path in inputs:
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        stem = path.name[:-len(suffix)] if path.name.endswith(suffix) else path.stem
        mask_path = output / relative.parent / f"{stem}_mask.png"
        probability_path = output / relative.parent / f"{stem}_prob.npy"
        for target in [mask_path] + ([probability_path] if args.save_probabilities else []):
            if target == path or target in destinations:
                raise ValueError(f"Prediction filename collision: {target}")
            if target.exists() and not args.overwrite:
                raise FileExistsError(f"{target} exists; use --overwrite to replace it")
            destinations.add(target)
        jobs.append((path, mask_path, probability_path))
    for path, mask_path, probability_path in tqdm(jobs, desc="inference"):
        probabilities = predict_tiled(
            model, image_tensor(read_rgb(path), config["preprocessing"]), device,
            patch_size=settings["patch_size"], overlap=settings["overlap"],
            batch_size=settings["batch_size"], amp=config["train"]["amp"],
        ).numpy()
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((probabilities >= settings["threshold"]).astype(np.uint8) * 255).save(mask_path)
        if args.save_probabilities:
            np.save(probability_path, probabilities.astype(np.float32))
    print(f"Saved {len(jobs)} masks to {output}; patch={settings['patch_size']}, "
          f"overlap={settings['overlap']}, threshold={settings['threshold']}")


if __name__ == "__main__":
    main()
