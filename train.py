"""Train any methods/<name>.py model using the same data and experiment loop."""

import argparse
import copy
import csv
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
import yaml

from methods import create_model, available_methods
from roadseg.config import load_config, validate_config, check_resume_config
from roadseg.data import RandomCropDataset, prepare_splits, encode_splits, decode_splits
from roadseg.engine import train_one_epoch, evaluate
from roadseg.losses import BCEDiceLoss
from roadseg.utils import (select_device, seed_everything, rng_state, restore_rng,
                           atomic_save, load_checkpoint)


def write_history(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML config; default configs/unet.yaml, or saved config on resume")
    parser.add_argument("--method", choices=available_methods())
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int, help="Total epoch count, including completed epochs")
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--resume", help="Path to a run's last.pt; continue in that run directory")
    args = parser.parse_args(argv)

    checkpoint = load_checkpoint(args.resume) if args.resume else None
    config = (load_config(args.config) if args.config else
              copy.deepcopy(checkpoint["config"]) if checkpoint else load_config("configs/unet.yaml"))
    if args.method:
        config["method"]["name"] = args.method
    if args.data_root:
        config["data"]["root"] = args.data_root
    for argument, key in ((args.output_dir, "output_dir"), (args.epochs, "epochs"), (args.device, "device")):
        if argument is not None:
            config["train"][key] = argument
    if checkpoint:
        resume_dir = Path(args.resume).resolve().parent
        if args.output_dir and Path(args.output_dir).resolve() != resume_dir:
            raise ValueError("Resume continues in the checkpoint's directory; omit --output-dir")
        config["train"]["output_dir"] = str(resume_dir)
        if not (resume_dir / "best.pt").is_file():
            raise FileNotFoundError("Resume needs the run directory, including best.pt and last.pt")
        check_resume_config(checkpoint["config"], config)
    validate_config(config)
    config["data"]["root"] = str(Path(config["data"]["root"]).expanduser().resolve())
    output = Path(config["train"]["output_dir"]).expanduser().resolve()
    config["train"]["output_dir"] = str(output)
    if not checkpoint and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty. Use a new output directory or --resume.")
    if checkpoint and (output / "history.csv").is_file():
        with open(output / "history.csv", newline="", encoding="utf-8") as handle:
            saved_rows = list(csv.DictReader(handle))
        if saved_rows and int(saved_rows[-1]["epoch"]) > checkpoint["epoch"]:
            raise ValueError("Checkpoint is older than this run's history. Resume from last.pt.")
    seed_everything(config["seed"])
    device = select_device(config["train"]["device"])
    train_config, data_config = config["train"], config["data"]
    amp = bool(train_config["amp"] and device.type == "cuda")

    if checkpoint:
        train_pairs, val_pairs = decode_splits(checkpoint["splits"], data_config["root"])
        splits = checkpoint["splits"]
    else:
        train_pairs, val_pairs = prepare_splits(config)
        splits = encode_splits(train_pairs, val_pairs, data_config["root"])
    dataset = RandomCropDataset(
        train_pairs, data_config["crop_size"], data_config["crops_per_image"],
        config["preprocessing"], data_config["mask_threshold"], data_config["augment"], config["seed"],
    )
    generator = torch.Generator()
    loader = DataLoader(
        dataset, batch_size=train_config["batch_size"], shuffle=True,
        num_workers=train_config["num_workers"], pin_memory=device.type == "cuda",
        generator=generator, drop_last=False, persistent_workers=False,
    )
    model_params = copy.deepcopy(config["method"]["params"])
    if checkpoint and "pretrained" in model_params:
        # Resume restores the encoder from the checkpoint; do not download ImageNet weights again.
        model_params["pretrained"] = False
    model = create_model(config["method"]["name"], **model_params).to(device)
    criterion = BCEDiceLoss(**config["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"],
                                  weight_decay=train_config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best_iou, history = 0, -1.0, []
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if amp and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best_iou, history = checkpoint["epoch"], checkpoint["best_iou"], checkpoint["history"]
        restore_rng(checkpoint["rng"])
    if start_epoch >= train_config["epochs"]:
        raise ValueError(f"Already completed {start_epoch} epochs; increase --epochs to continue")

    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / "splits.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    print(f"Method: {config['method']['name']} | device: {device} | AMP: {amp}", flush=True)
    print(f"Images: {len(train_pairs)} train / {len(val_pairs)} validation; "
          f"{len(dataset)} random crops per epoch", flush=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} | output: {output}", flush=True)

    for epoch in range(start_epoch, train_config["epochs"]):
        started = time.perf_counter()
        dataset.set_epoch(epoch)
        generator.manual_seed(config["seed"] + epoch)
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, loader, optimizer, scaler, criterion, device,
                                     amp, train_config["grad_clip_norm"])
        scores = evaluate(model, val_pairs, config, device, amp=amp)
        scheduler.step(scores["iou"])
        improved = scores["iou"] > best_iou
        best_iou = max(best_iou, scores["iou"])
        row = {"epoch": epoch + 1, "train_loss": train_loss, "lr": lr,
               **{f"val_{key}": value for key, value in scores.items()},
               "seconds": time.perf_counter() - started}
        history.append(row)
        payload = {
            "format_version": 1, "epoch": epoch + 1, "best_iou": best_iou,
            "config": config, "splits": splits, "history": history,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "rng": rng_state(), "torch_version": str(torch.__version__),
        }
        if improved:
            atomic_save(payload, output / "best.pt")
        atomic_save(payload, output / "last.pt")
        write_history(output / "history.csv", history)
        print(f"Epoch {epoch + 1}/{train_config['epochs']} | loss {train_loss:.4f} | "
              f"val IoU {scores['iou']:.4f} | Dice {scores['dice']:.4f} | "
              f"best {best_iou:.4f} | {row['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
