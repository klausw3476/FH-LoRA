#!/usr/bin/env python3
"""
Plot linear-probe Top-1 vs epoch from output_ssv2/eval_*/log.txt (Fig 5 style).

Example:
  cd FH-LoRA
  python scripts/plot_probe_convergence.py \\
    --out analysis/fig5_probe_convergence.png \\
    --root output_ssv2 \\
    eval_linear_probe_no_tdlora:NoLoRA eval_standard_lora:Standard \\
    eval_perproj_r48:Per-projection eval_fhlora_r48_hhd32:FH-LoRA
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt


def series_from_log(path: str) -> Tuple[List[int], List[float]]:
    epochs, accs = [], []
    if not os.path.isfile(path):
        return epochs, accs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in d and "test_acc1" in d:
                epochs.append(int(d["epoch"]))
                accs.append(float(d["test_acc1"]))
    return epochs, accs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="output_ssv2",
        help="Parent directory containing eval_* folders",
    )
    ap.add_argument(
        "--out",
        default="analysis/fig5_probe_convergence.png",
        help="Output image path (.png or .pdf)",
    )
    ap.add_argument(
        "runs",
        nargs="+",
        metavar="eval_dir:Label",
        help="Pairs: eval subdirectory name and legend label",
    )
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    plt.figure(figsize=(7, 4))
    for spec in args.runs:
        if ":" not in spec:
            raise SystemExit(f"Expected eval_dir:Label, got: {spec}")
        sub, label = spec.split(":", 1)
        path = os.path.join(root, sub.strip(), "log.txt")
        ep, acc = series_from_log(path)
        if not ep:
            print(f"Warning: no points in {path}")
            continue
        plt.plot(ep, acc, marker="o", markersize=3, linewidth=1.5, label=label)

    plt.xlabel("Linear probe epoch")
    plt.ylabel("Val Top-1 accuracy (%)")
    plt.title("SSv2 linear probe convergence (174-class)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
