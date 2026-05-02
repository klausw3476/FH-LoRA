"""Plot CKA summary from analyze_cka.py JSON output (e.g. analysis/cka_ssv2/)."""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_json", required=True)
    p.add_argument("--output_png", required=True)
    args = p.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    layers = data["layers"]
    order = sorted(layers.keys(), key=lambda s: int(s.split()[-1]))
    labels = [f"L{k.split()[-1]}" for k in order]
    with_l = [layers[k]["with_lora"] for k in order]
    wo_l = [layers[k]["without_lora"] for k in order]
    if "cls" in data:
        labels.append("CLS")
        with_l.append(data["cls"]["with_lora"])
        wo_l.append(data["cls"]["without_lora"])

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(x - w / 2, with_l, w, label="Student + adapters", color="#2c5f8d")
    ax.bar(x + w / 2, wo_l, w, label="Student w/o adapters", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"CKA (teacher vs.\ student)")
    ax.set_xlabel("Representation")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Linear CKA on SSv2 features (ViT-Small)", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_png) or ".", exist_ok=True)
    plt.savefig(args.output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print("Wrote", args.output_png)


if __name__ == "__main__":
    main()
