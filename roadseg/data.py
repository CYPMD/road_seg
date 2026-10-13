from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Pair:
    image: Path
    mask: Path


def discover_pairs(images_dir, masks_dir=None, image_suffix="_sat.jpg", mask_suffix="_mask.png"):
    images_dir = Path(images_dir).resolve()
    masks_dir = Path(masks_dir).resolve() if masks_dir is not None else images_dir
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(f"Image/mask directory does not exist: {images_dir}, {masks_dir}")
    images = sorted(path for path in images_dir.rglob(f"*{image_suffix}") if path.is_file())
    if not images:
        raise ValueError(f"No *{image_suffix} files in {images_dir}")
    pairs, missing = [], []
    for image in images:
        relative = image.relative_to(images_dir)
        mask_name = image.name[:-len(image_suffix)] + mask_suffix
        mask = (masks_dir / relative.parent / mask_name).resolve()
        if not mask.is_file():
            missing.append(str(mask))
        else:
            pairs.append(Pair(image.resolve(), mask))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} masks. First: {missing[0]}. "
                                "Use a labeled directory; unlabeled valid/test images are for inference.")
    return pairs


def prepare_splits(config: dict) -> tuple[list[Pair], list[Pair]]:
    data = config["data"]
    root = Path(data["root"]).expanduser().resolve()

    def discover(kind):
        images = root / data[f"{kind}_images"]
        masks = root / data[f"{kind}_masks"] if data.get(f"{kind}_masks") else None
        return discover_pairs(images, masks, data["image_suffix"], data["mask_suffix"])

    pairs = discover("train")
    if data.get("val_images"):
        train, val = pairs, discover("val")
    else:
        if len(pairs) < 2:
            raise ValueError("Need at least two labeled images to create train/validation splits")
        random.Random(config["seed"]).shuffle(pairs)
        count = max(1, min(len(pairs) - 1, round(len(pairs) * data["val_fraction"])))
        val, train = pairs[:count], pairs[count:]
    check_disjoint(train, val)
    return train, val


def check_disjoint(train, val):
    if not train or not val:
        raise ValueError("Training and validation must both be nonempty")
    for attribute in ("image", "mask"):
        train_paths = {getattr(pair, attribute).resolve() for pair in train}
        val_paths = {getattr(pair, attribute).resolve() for pair in val}
        if train_paths & val_paths:
            raise ValueError(f"Train/validation {attribute} paths overlap")


def encode_splits(train, val, root):
    root = Path(root).expanduser().resolve()
    try:
        return {name: [{"image": p.image.relative_to(root).as_posix(),
                        "mask": p.mask.relative_to(root).as_posix()} for p in pairs]
                for name, pairs in (("train", train), ("val", val))}
    except ValueError as exc:
        raise ValueError("All image and mask directories must be inside data.root") from exc


def decode_splits(splits, root):
    root = Path(root).expanduser().resolve()
    result = {}
    for name in ("train", "val"):
        result[name] = []
        for entry in splits[name]:
            image, mask = (root / entry["image"]).resolve(), (root / entry["mask"]).resolve()
            if not image.is_relative_to(root) or not mask.is_relative_to(root):
                raise ValueError("Split paths must be inside data.root")
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(f"Saved split file missing: {image} or {mask}")
            result[name].append(Pair(image, mask))
    check_disjoint(result["train"], result["val"])
    return result["train"], result["val"]


def read_rgb(path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def read_mask(path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L")


def image_tensor(image, preprocessing):
    array = np.array(image, dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = tensor.new_tensor(preprocessing["mean"])[:, None, None]
    std = tensor.new_tensor(preprocessing["std"])[:, None, None]
    return (tensor - mean) / std


class RandomCropDataset(Dataset):
    def __init__(self, pairs, crop_size, crops_per_image, preprocessing,
                 mask_threshold=128, augment=True, seed=42):
        self.pairs = pairs
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.preprocessing = preprocessing
        self.mask_threshold = mask_threshold
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.pairs) * self.crops_per_image

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        pair = self.pairs[index % len(self.pairs)]
        image, mask = read_rgb(pair.image), read_mask(pair.mask)
        if image.size != mask.size:
            raise ValueError(f"Image/mask size mismatch: {pair.image}")
        width, height = image.size
        size = self.crop_size
        if min(width, height) < size:
            raise ValueError(f"{pair.image} is smaller than crop_size={size}")
        # Local RNG gives different crops each epoch, independent of worker count.
        rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
        left, top = rng.randrange(width - size + 1), rng.randrange(height - size + 1)
        box = (left, top, left + size, top + size)
        image, mask = image.crop(box), mask.crop(box)
        if self.augment:
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            rotation = rng.choice([None, Image.Transpose.ROTATE_90,
                                   Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270])
            if rotation is not None:
                image, mask = image.transpose(rotation), mask.transpose(rotation)
        target = torch.from_numpy((np.array(mask) >= self.mask_threshold).astype(np.float32))[None]
        return image_tensor(image, self.preprocessing), target
