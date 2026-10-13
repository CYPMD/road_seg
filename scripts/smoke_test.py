"""Run the real CLIs on generated 1024px images, including checkpoint resume.

This verifies wiring and output shape, not model quality on DeepGlobe.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import yaml


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments):
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("MKL_NUM_THREADS", "2")
    subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="smoke-output", help="Must be absent or empty")
    args = parser.parse_args()
    work = Path(args.work_dir).resolve()
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"{work} is not empty; choose a new --work-dir")
    train = work / "data" / "train"
    train.mkdir(parents=True)
    rng = np.random.default_rng(123)
    for index in range(4):
        mask = Image.new("L", (1024, 1024))
        draw = ImageDraw.Draw(mask)
        draw.line([(0, 200 + index * 30), (500, 500), (1023, 700)], fill=255, width=12)
        draw.line([(400, 0), (600, 1023)], fill=255, width=10)
        image = rng.integers(20, 100, (1024, 1024, 3), dtype=np.uint8)
        image[np.array(mask) > 0] = [180, 180, 180]
        Image.fromarray(image).save(train / f"{index}_sat.jpg")
        mask.save(train / f"{index}_mask.png")
    config = yaml.safe_load((ROOT / "configs" / "unet.yaml").read_text())
    config["method"]["params"]["base_channels"] = 4
    config["data"].update(root=str(work / "data"), crops_per_image=1, val_fraction=0.25)
    config["train"].update(output_dir=str(work / "run"), epochs=1, batch_size=2,
                            num_workers=0, device="cpu", amp=False)
    config["inference"]["batch_size"] = 8
    config_path = work / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    run("train.py", "--config", config_path)
    checkpoint = work / "run" / "last.pt"
    run("train.py", "--resume", checkpoint, "--epochs", "2")
    run("evaluate.py", "--checkpoint", checkpoint, "--device", "cpu")
    run("infer.py", "--checkpoint", checkpoint, "--input", train / "0_sat.jpg",
        "--output-dir", work / "predictions", "--device", "cpu", "--save-probabilities")
    with Image.open(work / "predictions" / "0_mask.png") as predicted:
        assert predicted.size == (1024, 1024)
        assert set(np.unique(predicted)).issubset({0, 255})
    probabilities = np.load(work / "predictions" / "0_prob.npy")
    assert probabilities.shape == (1024, 1024)
    assert probabilities.dtype == np.float32
    assert np.isfinite(probabilities).all() and probabilities.min() >= 0 and probabilities.max() <= 1
    history = (work / "run" / "history.csv").read_text().splitlines()
    assert len(history) == 3  # Header + two epochs.
    split = json.loads((work / "run" / "splits.json").read_text())
    assert len(split["train"]) == 3 and len(split["val"]) == 1
    print("PASS: training, resume, evaluation, 49-tile inference, and saved output checks")


if __name__ == "__main__":
    main()
