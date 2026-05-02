#!/usr/bin/env python3
"""
Create a clean student checkpoint for supervised SSv2 baselines without distillation.

Examples:
  python scripts/make_student_init_ckpt.py \
      --output output_ssv2/init_small_scratch.pth \
      --student_arch small \
      --use_tdlora false

  python scripts/make_student_init_ckpt.py \
      --output output_ssv2/init_fhlora_r48_hhd32_scratch.pth \
      --student_arch small \
      --use_tdlora true \
      --lora_type fh_lora \
      --lora_rank 48 \
      --hyper_hidden_dim 32
"""

import argparse
import os
import random

import numpy as np
import torch

from models.student_vit import (
    student_vit_small,
    student_vit_tiny,
    student_vit_nano,
    student_vit_pico,
)
from utils import utils


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--student_arch", default="small",
                   choices=["small", "tiny", "nano", "pico"])
    p.add_argument("--img_size", default=224, type=int)
    p.add_argument("--use_tdlora", default=False, type=utils.bool_flag)
    p.add_argument("--lora_type", default="fh_lora", type=str,
                   choices=["standard", "per_projection", "fh_lora"])
    p.add_argument("--lora_rank", default=48, type=int)
    p.add_argument("--time_embed_dim", default=64, type=int)
    p.add_argument("--hyper_hidden_dim", default=32, type=int)
    p.add_argument("--content_dim", default=16, type=int)
    p.add_argument("--content_dropout", default=0.3, type=float)
    p.add_argument("--seed", default=0, type=int)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    factories = {
        "small": student_vit_small,
        "tiny": student_vit_tiny,
        "nano": student_vit_nano,
        "pico": student_vit_pico,
    }

    model = factories[args.student_arch](
        img_size=args.img_size,
        use_tdlora=args.use_tdlora,
        lora_rank=args.lora_rank,
        time_embed_dim=args.time_embed_dim,
        lora_type=args.lora_type,
        hyper_hidden_dim=args.hyper_hidden_dim,
        content_dim=args.content_dim,
        content_dropout=args.content_dropout,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({"student": model.state_dict()}, args.output)

    total = sum(p.numel() for p in model.parameters())
    print(f"Saved init checkpoint: {args.output}")
    print(f"Architecture: ViT-{args.student_arch} | use_tdlora={args.use_tdlora}")
    print(f"Parameters: {total:,}")


if __name__ == "__main__":
    main()
