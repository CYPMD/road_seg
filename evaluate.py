"""Evaluate the saved validation split, or a separate labeled directory."""

import argparse
import json

from roadseg.config import validate_config
from roadseg.data import decode_splits, discover_pairs
from roadseg.engine import evaluate
from roadseg.utils import select_device, load_checkpoint, model_from_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", help="Relocate the dataset referenced by the saved split")
    parser.add_argument("--images", help="Alternative labeled image directory (ignores the saved split)")
    parser.add_argument("--masks", help="Mask directory for --images; defaults to alongside images")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--method")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args(argv)
    if args.masks and not args.images:
        parser.error("--masks requires --images")
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    if args.data_root:
        config["data"]["root"] = args.data_root
    for key in ("batch_size", "threshold"):
        if getattr(args, key) is not None:
            config["inference"][key] = getattr(args, key)
    validate_config(config)
    if args.images:
        pairs = discover_pairs(args.images, args.masks, config["data"]["image_suffix"],
                               config["data"]["mask_suffix"])
    else:
        _, pairs = decode_splits(checkpoint["splits"], config["data"]["root"])
    device = select_device(args.device)
    model = model_from_checkpoint(checkpoint, device, args.method)
    scores = evaluate(model, pairs, config, device, amp=config["train"]["amp"])
    print(json.dumps(scores, indent=2))
    return scores


if __name__ == "__main__":
    main()
