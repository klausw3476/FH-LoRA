#!/usr/bin/env python3
"""
Evaluate the frozen SSv2 TimeSformer teacher on the SSv2 validation split.

This uses the same repo data/config stack as the student pipelines and reports
Top-1 / Top-5 directly from the teacher's classification head.
"""

import argparse
import json
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score

from datasets import Ssv2
from models.timesformer import VisionTransformer as TeacherViT
from utils.parser import load_config
from utils import utils


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_weights", required=True, type=str)
    p.add_argument("--data_path", required=True, type=str)
    p.add_argument("--path_prefix", default="/mnt/data/ssv2_frames", type=str)
    p.add_argument("--output_dir", default="output_ssv2/eval_teacher_ssv2", type=str)
    p.add_argument("--batch_size_per_gpu", default=8, type=int)
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--cfg", dest="cfg_file", type=str,
                   default="models/configs/SSv2/TimeSformer_divST_8_224.yaml")
    p.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    return p.parse_args()


@torch.no_grad()
def evaluate(loader, teacher: TeacherViT) -> dict:
    teacher.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    all_targets, all_preds = [], []

    for inp, target, _, _ in metric_logger.log_every(loader, 50, "Test:"):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        cls_token, _ = teacher.forward_features(inp)
        logits = teacher.head(cls_token)
        loss = nn.CrossEntropyLoss()(logits, target)

        acc1, acc5 = utils.accuracy(logits, target, topk=(1, 5))
        preds = logits.argmax(dim=1)
        all_targets.extend(target.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    f1 = f1_score(all_targets, all_preds, average="micro")
    return {
        "test_loss": stats.get("loss", 0.0),
        "test_acc1": stats.get("acc1", 0.0),
        "test_acc5": stats.get("acc5", 0.0),
        "test_f1": f1,
        "num_samples": len(all_targets),
        "epoch": 0,
    }


def main() -> None:
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_config(args)
    cfg.DATA.PATH_TO_DATA_DIR = args.data_path
    cfg.DATA.PATH_PREFIX = args.path_prefix
    cfg.DATA.NO_RGB_AUG = True
    cfg.DATA.RANDOM_FLIP = False

    dataset_val = Ssv2(cfg=cfg, mode="val", num_retries=10)
    loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    ckpt = torch.load(args.teacher_weights, map_location="cpu", weights_only=False)
    if "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    elif "teacher" in ckpt:
        ckpt = ckpt["teacher"]
    state_dict = {
        k.replace("module.", "").replace("backbone.", "").replace("model.", ""): v
        for k, v in ckpt.items()
    }

    teacher_num_frames = cfg.DATA.NUM_FRAMES
    if "time_embed" in state_dict:
        teacher_num_frames = state_dict["time_embed"].shape[1]

    teacher = TeacherViT(
        img_size=cfg.DATA.TRAIN_CROP_SIZE,
        num_classes=cfg.MODEL.NUM_CLASSES,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        num_frames=teacher_num_frames,
        attention_type=cfg.TIMESFORMER.ATTENTION_TYPE,
    )
    msg = teacher.load_state_dict(state_dict, strict=False)
    print(f"Teacher loaded: {msg}")
    teacher.cuda()

    results = evaluate(loader, teacher)
    print(json.dumps(results, indent=2))
    with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
        f.write(json.dumps(results) + "\n")
    with open(os.path.join(args.output_dir, "teacher_eval.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
