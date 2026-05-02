#!/usr/bin/env python3
"""
Plot Phase 1 / Phase 2 distillation training loss (Supplementary Fig S1 style).

Expects JSON lines with at least train_loss or train_distill_loss per epoch.

Example:
  python scripts/plot_phase_distill_loss.py \\
    --out analysis/figS1_distill_loss.png \\
    output_ssv2/phase1/log.txt:Phase1_spatial \\
    output_ssv2/phase2_fh_lora/log.txt:Phase2_FH-LoRA
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple


def load_loss_series(path: str) -> Tuple[List[int], List[float]]:
    epochs, losses = [], []
    if not os.path.isfile(path):
        return epochs, losses
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep = d.get("epoch")
            if ep is None:
                continue
            loss = d.get("train_distill_loss", d.get("train_loss"))
            if loss is None:
                continue
            epochs.append(int(ep))
            losses.append(float(loss))
    return epochs, losses


def main() -> None:
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="analysis/figS1_distill_loss.png",
        help="Output image path",
    )
    ap.add_argument(
        "curves",
        nargs="+",
        metavar="log.txt:Label",
        help="log path (relative or absolute) and legend label",
    )
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    plt.figure(figsize=(7, 4))
    for spec in args.curves:
        if ":" not in spec:
            raise SystemExit(f"Expected log.txt:Label, got: {spec}")
        raw_path, label = spec.rsplit(":", 1)
        path = raw_path.strip()
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        ep, ls = load_loss_series(path)
        if not ep:
            print(f"Warning: no loss points in {path}")
            continue
        plt.plot(ep, ls, marker="o", markersize=3, linewidth=1.5, label=label.strip())

    plt.xlabel("Distillation epoch")
    plt.ylabel("Train loss")
    plt.title("Distillation training loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
