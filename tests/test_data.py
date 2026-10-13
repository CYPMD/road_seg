import numpy as np
from PIL import Image
import pytest
import torch

from roadseg.config import load_config
from roadseg.data import (discover_pairs, RandomCropDataset, prepare_splits,
                          encode_splits, decode_splits)


def test_random_crops_keep_image_mask_alignment(tmp_path):
    yy, xx = np.indices((384, 384))
    mask = ((xx // 7 + yy // 13) % 2).astype(np.uint8) * 255
    rgb = np.stack([mask, xx % 256, yy % 256], axis=-1).astype(np.uint8)
    Image.fromarray(rgb).save(tmp_path / "one_sat.png")
    Image.fromarray(mask).save(tmp_path / "one_mask.png")
    pairs = discover_pairs(tmp_path, image_suffix="_sat.png")
    dataset = RandomCropDataset(pairs, 256, 4, {"mean": [0, 0, 0], "std": [1, 1, 1]}, augment=True)
    assert len(dataset) == 4
    first, first_target = dataset[0]
    again, _ = dataset[0]
    assert first.shape == (3, 256, 256) and first_target.shape == (1, 256, 256)
    torch.testing.assert_close(first, again)
    for epoch in range(3):
        dataset.set_epoch(epoch)
        for index in range(len(dataset)):
            image, target = dataset[index]
            torch.testing.assert_close(image[:1], target)
    changed, _ = dataset[0]
    assert not torch.equal(first, changed)


def test_split_is_reproducible_and_roundtrips(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    for i in range(10):
        (train / f"{i}_sat.jpg").touch()
        (train / f"{i}_mask.png").touch()
    config = load_config("configs/unet.yaml")
    config["data"]["root"] = str(tmp_path)
    first, second = prepare_splits(config)
    assert not {pair.image for pair in first} & {pair.image for pair in second}
    assert (first, second) == prepare_splits(config)
    saved = encode_splits(first, second, tmp_path)
    assert (first, second) == decode_splits(saved, tmp_path)


def test_separate_nested_mask_directory(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    (images / "region").mkdir(parents=True)
    (masks / "region").mkdir(parents=True)
    (images / "region" / "a_sat.jpg").touch()
    (masks / "region" / "a_mask.png").touch()
    pairs = discover_pairs(images, masks)
    assert pairs[0].mask == masks / "region" / "a_mask.png"


def test_missing_masks_fail_clearly(tmp_path):
    (tmp_path / "a_sat.jpg").touch()
    with pytest.raises(FileNotFoundError, match="Missing 1 masks"):
        discover_pairs(tmp_path)
