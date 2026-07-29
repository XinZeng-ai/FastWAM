#!/usr/bin/env python3
"""从训练日志解析 train/val 指标并上传到 wandb。

用法:
    python scripts/log_to_wandb.py <log_path> --project fast-wam --name robotwin_uncond

需要先设置 wandb:
    export WANDB_API_KEY=xxx   # 或 wandb login
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np


_TRAIN_RE = re.compile(
    r"epoch=(\d+)\s+step=(\d+)/(\d+)"
    r".*?loss=([\d.eE+-]+)"
    r".*?loss_action=([\d.eE+-]+)"
    r".*?loss_video=([\d.eE+-]+)"
    r"(?:.*?lr=([\d.eE+-]+))?",
    re.DOTALL,
)

_VAL_RE = re.compile(
    r"step=(\d+)\s+val_loss=([\d.eE+-]+)"
    r"(?:.*?infer_psnr=([\d.eE+-]+))?"
    r"(?:.*?infer_ssim=([\d.eE+-]+))?"
    r"(?:.*?action_l2=([\d.eE+-]+))?"
    r"(?:.*?action_l1=([\d.eE+-]+))?",
    re.DOTALL,
)


def parse_log(log_path):
    text = Path(log_path).read_text(errors="replace")

    train_raw = []
    for m in _TRAIN_RE.finditer(text):
        train_raw.append({
            "epoch": int(m.group(1)),
            "step": int(m.group(2)),
            "total_steps": int(m.group(3)),
            "loss": float(m.group(4)),
            "loss_action": float(m.group(5)),
            "loss_video": float(m.group(6)),
            "lr": float(m.group(7)) if m.group(7) else None,
        })

    val_raw = []
    for m in _VAL_RE.finditer(text):
        val_raw.append({
            "step": int(m.group(1)),
            "val_loss": float(m.group(2)),
            "psnr": float(m.group(3)) if m.group(3) else None,
            "ssim": float(m.group(4)) if m.group(4) else None,
            "action_l2": float(m.group(5)) if m.group(5) else None,
            "action_l1": float(m.group(6)) if m.group(6) else None,
        })

    train_dedup = {}
    for rec in train_raw:
        train_dedup[rec["step"]] = rec
    train_sorted = sorted(train_dedup.values(), key=lambda r: r["step"])

    val_dedup = {}
    for rec in val_raw:
        val_dedup[rec["step"]] = rec
    val_sorted = sorted(val_dedup.values(), key=lambda r: r["step"])

    return train_sorted, val_sorted


def main():
    parser = argparse.ArgumentParser(description="Upload train/val metrics from log to wandb")
    parser.add_argument("log_path", help="Path to train.log")
    parser.add_argument("--project", default="fast-wam", help="wandb project name")
    parser.add_argument("--name", default=None, help="wandb run name (default: log dir name)")
    parser.add_argument("--entity", default=None, help="wandb entity/team name")
    parser.add_argument("--group", default=None, help="wandb group name")
    args = parser.parse_args()

    run_name = args.name or Path(args.log_path).parent.name

    train_records, val_records = parse_log(args.log_path)
    print(f"Parsed: {len(train_records)} train records, {len(val_records)} val records")

    if not train_records and not val_records:
        print("No records found!", file=sys.stderr)
        sys.exit(1)

    import wandb

    wandb.init(
        project=args.project,
        name=run_name,
        entity=args.entity,
        group=args.group,
        config={
            "total_steps": train_records[-1]["total_steps"] if train_records else None,
            "num_train_logs": len(train_records),
            "num_val_logs": len(val_records),
            "log_path": args.log_path,
        },
        reinit=True,
    )

    # 合并 train 和 val，按 step 排序，同一 step 的 train 和 val 作为一次 log
    by_step = {}
    for rec in train_records:
        s = rec["step"]
        if s not in by_step:
            by_step[s] = {}
        by_step[s].update({
            "train/loss": rec["loss"],
            "train/loss_action": rec["loss_action"],
            "train/loss_video": rec["loss_video"],
        })
        if rec["lr"] is not None:
            by_step[s]["train/lr"] = rec["lr"]

    for rec in val_records:
        s = rec["step"]
        if s not in by_step:
            by_step[s] = {}
        by_step[s].update({
            "val/val_loss": rec["val_loss"],
        })
        if rec["psnr"] is not None:
            by_step[s]["val/infer_psnr"] = rec["psnr"]
        if rec["ssim"] is not None:
            by_step[s]["val/infer_ssim"] = rec["ssim"]
        if rec["action_l2"] is not None:
            by_step[s]["val/action_l2"] = rec["action_l2"]
        if rec["action_l1"] is not None:
            by_step[s]["val/action_l1"] = rec["action_l1"]

    for step in sorted(by_step.keys()):
        wandb.log(by_step[step], step=step)

    wandb.finish()
    print(f"Uploaded {len(by_step)} steps to wandb project={args.project} name={run_name}")


if __name__ == "__main__":
    main()
