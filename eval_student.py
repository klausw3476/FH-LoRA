"""
Evaluate the TD-LoRA student model on downstream endoscopy tasks.

Supports the same 4 downstream tasks as the original codebase:
  1. PolypDiag   — binary classification (F1)
  2. CVC-12k     — polyp segmentation (Dice)
  3. KUMC        — lesion detection (F1)
  4. Cholec80    — workflow recognition (accuracy)

For classification (PolypDiag / Cholec80), a linear head is appended
on top of the student CLS token.  The entire student + head is fine-tuned.

Usage:
  python eval_student.py \
      --student_weights output/phase2/checkpoint.pth \
      --data_path /path/to/polypdiag \
      --dataset polypdiag \
      --num_labels 2 \
      --epochs 20 \
      --output_dir output/eval_polypdiag
"""

import utils.checkpoint_compat  # noqa: F401 — numpy/torch checkpoint pickle compat (must be first)

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score

from models.student_vit import student_vit_small, student_vit_tiny, student_vit_nano, student_vit_pico
from datasets import Kinetics, Ssv2
from utils import utils
from utils.parser import load_config


class LinearClassifier(nn.Module):
    """Linear probe / fine-tuning head."""

    def __init__(self, dim, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        x = x.view(x.size(0), -1)
        return self.linear(x)


def get_args_parser():
    parser = argparse.ArgumentParser('Student Model Evaluation',
                                     add_help=False)
    parser.add_argument('--student_weights', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', default='output/eval', type=str)

    parser.add_argument('--dataset', default='polypdiag', type=str,
                        choices=['polypdiag', 'cholec80', 'ucf101', 'hmdb51',
                                 'kinetics400', 'ssv2'])
    parser.add_argument('--num_labels', default=2, type=int)
    parser.add_argument('--path_prefix', default=None, type=str,
                        help='Override DATA.PATH_PREFIX (e.g. path to frames)')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size_per_gpu', default=128, type=int)
    parser.add_argument('--val_freq', default=1, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--use_tdlora', default=True, type=utils.bool_flag,
                        help='Whether the checkpoint contains LoRA weights')
    parser.add_argument('--lora_rank', default=8, type=int)
    parser.add_argument('--time_embed_dim', default=64, type=int)
    parser.add_argument('--hyper_hidden_dim', default=8, type=int,
                        help='Hidden dim of hypernetwork for hyper variants')
    parser.add_argument('--content_dim', default=16, type=int,
                        help='Projection dim for content-aware gating (Proposal 1)')
    parser.add_argument('--content_dropout', default=0.3, type=float,
                        help='Dropout on content features to prevent PE being ignored')
    parser.add_argument('--lora_type', default='fh_lora', type=str,
                        choices=['standard', 'per_projection', 'fh_lora'],
                        help='LoRA family: standard (frame-blind LoRA), '
                             'per_projection (one hypernetwork per Q/K/V), '
                             'fh_lora (FH-LoRA, proposed).')
    parser.add_argument('--freeze_backbone', default=False, type=utils.bool_flag,
                        help='Only train LoRA + head (linear probe)')
    parser.add_argument('--student_arch', default='small', type=str,
                        choices=['small', 'tiny', 'nano', 'pico'],
                        help='Student architecture: small (384-dim), tiny (192-dim), or nano (128-dim)')

    parser.add_argument('--frame_shuffle', default='none', type=str,
                        choices=['none', 'reverse', 'random'],
                        help='Shuffle frame order during validation to test '
                             'temporal sensitivity')

    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_weights', default='', type=str)

    parser.add_argument(
        '--export_per_class_json', default='', type=str,
        help='If set (SSv2 only), after training save per-class accuracy JSON '
             'using weights from best_checkpoint.pth.tar in output_dir.')

    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--cfg", dest="cfg_file", type=str,
                        default="models/configs/Kinetics/"
                                "TimeSformer_divST_8x32_224.yaml")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)

    return parser


def build_datasets(args, config):
    config.DATA.PATH_TO_DATA_DIR = args.data_path
    if args.path_prefix:
        config.DATA.PATH_PREFIX = args.path_prefix
    elif not config.DATA.PATH_PREFIX:
        config.DATA.PATH_PREFIX = "."
    config.DATA.NO_RGB_AUG = True

    if args.dataset in ['ucf101', 'hmdb51']:
        config.TEST.NUM_SPATIAL_CROPS = 3
    else:
        config.TEST.NUM_SPATIAL_CROPS = 1

    if args.dataset == 'ssv2':
        config.DATA.RANDOM_FLIP = False
        dataset_train = Ssv2(cfg=config, mode="train", num_retries=10)
        dataset_val = Ssv2(cfg=config, mode="val", num_retries=10)
    elif args.dataset in ['polypdiag', 'kinetics400']:
        dataset_train = Kinetics(cfg=config, mode="train", num_retries=10)
        dataset_val = Kinetics(cfg=config, mode="val", num_retries=10)
    else:
        dataset_train = Kinetics(cfg=config, mode="train", num_retries=10)
        dataset_val = Kinetics(cfg=config, mode="val", num_retries=10)

    return dataset_train, dataset_val


def eval_student(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    utils.init_distributed_mode(args)
    cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)

    if utils.is_main_process():
        with open(f"{args.output_dir}/config.json", "w") as f:
            json.dump(vars(args), f, indent=4)

    config = load_config(args)
    dataset_train, dataset_val = build_datasets(args, config)

    num_labels = args.num_labels
    if args.dataset in ('kinetics400', 'ssv2') and num_labels == 2:
        num_labels = config.MODEL.NUM_CLASSES
        if utils.is_main_process():
            print(f"Dataset {args.dataset}: using num_labels={num_labels} from config")

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, sampler=train_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers, pin_memory=True, shuffle=False)

    print(f"Data loaded: {len(dataset_train)} train, "
          f"{len(dataset_val)} val samples.")

    # ---- Build student model ----
    _student_factories = {'small': student_vit_small, 'tiny': student_vit_tiny, 'nano': student_vit_nano, 'pico': student_vit_pico}
    model = _student_factories[args.student_arch](
        img_size=config.DATA.TRAIN_CROP_SIZE,
        use_tdlora=args.use_tdlora,
        lora_rank=args.lora_rank,
        time_embed_dim=args.time_embed_dim,
        lora_type=args.lora_type,
        hyper_hidden_dim=args.hyper_hidden_dim,
        content_dim=args.content_dim,
        content_dropout=args.content_dropout)

    ckpt = torch.load(args.student_weights, map_location='cpu', weights_only=False)
    if 'student' in ckpt:
        state_dict = ckpt['student']
    else:
        state_dict = ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    if not args.use_tdlora:
        state_dict = {k: v for k, v in state_dict.items()
                      if 'lora_' not in k and 'gating' not in k}

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Student loaded: {msg}")
    model.cuda()

    if args.freeze_backbone:
        model.freeze_backbone()
        print("Backbone frozen; training only TD-LoRA + classifier head")

    model_embed_dim = model.embed_dim

    has_trainable = any(p.requires_grad for p in model.parameters())
    if has_trainable:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
    else:
        print("No trainable model params; skipping DDP for backbone")

    classifier = LinearClassifier(model_embed_dim, num_labels=num_labels)
    classifier.cuda()
    classifier = nn.parallel.DistributedDataParallel(
        classifier, device_ids=[args.gpu])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_cls = sum(p.numel() for p in classifier.parameters())
    print(f"Trainable model params: {n_params:,} | Classifier: {n_cls:,}")

    # ---- Test-only mode ----
    if args.test:
        if not args.test_weights:
            raise ValueError(
                '--test requires --test_weights (path to checkpoint)')
        test_ckpt = torch.load(args.test_weights, map_location='cpu')
        if 'backbone_state_dict' in test_ckpt:
            state = test_ckpt['backbone_state_dict']
            has_module_prefix = (
                state and next(iter(state.keys()), "").startswith("module.")
            )
            model_inner = model.module if hasattr(model, 'module') else model
            if has_module_prefix:
                state = {k.replace("module.", ""): v for k, v in state.items()}
            model_inner.load_state_dict(state, strict=False)
        if 'state_dict' in test_ckpt:
            state = test_ckpt['state_dict']
            has_module_prefix = (
                state and next(iter(state.keys()), "").startswith("module.")
            )
            cls_inner = classifier.module if hasattr(classifier, 'module') else classifier
            if has_module_prefix:
                state = {k.replace("module.", ""): v for k, v in state.items()}
            cls_inner.load_state_dict(state, strict=False)
        test_stats, f1, _ = validate(val_loader, model, classifier, num_labels,
                                     frame_shuffle=args.frame_shuffle)
        acc1 = test_stats.get('acc1', 0)
        acc5 = test_stats.get('acc5', 0)
        print(f"Test Top-1: {acc1:.1f}%  Top-5: {acc5:.1f}%  F1: {f1 * 100:.1f}%")
        return

    # ---- Optimizer ----
    scaled_lr = args.lr * (args.batch_size_per_gpu *
                           utils.get_world_size()) / 256.
    optimizer = torch.optim.AdamW(
        [{'params': model.parameters(), 'lr': scaled_lr},
         {'params': classifier.parameters(), 'lr': scaled_lr}],
        weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=0)

    best_f1 = 0.0
    start_epoch = 0

    # ---- Resume from checkpoint ----
    resume_path = os.path.join(args.output_dir, "checkpoint.pth.tar")
    if os.path.isfile(resume_path):
        resume_ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        start_epoch = resume_ckpt.get("epoch", 0)
        best_f1 = resume_ckpt.get("best_f1", 0.0)
        model_inner = model.module if hasattr(model, 'module') else model
        cls_inner = classifier.module if hasattr(classifier, 'module') else classifier
        if 'backbone_state_dict' in resume_ckpt:
            state = resume_ckpt['backbone_state_dict']
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model_inner.load_state_dict(state, strict=False)
        if 'state_dict' in resume_ckpt:
            state = resume_ckpt['state_dict']
            state = {k.replace("module.", ""): v for k, v in state.items()}
            cls_inner.load_state_dict(state, strict=False)
        if 'optimizer' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer'])
        if 'scheduler' in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt['scheduler'])
        print(f"Resumed from epoch {start_epoch} (best_f1={best_f1*100:.1f}%)")

    for epoch in range(start_epoch, args.epochs):
        prev_best = best_f1
        train_loader.sampler.set_epoch(epoch)
        train_stats = train(model, classifier, optimizer, train_loader, epoch)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats, f1, _ = validate(val_loader, model, classifier,
                                        num_labels,
                                        frame_shuffle=args.frame_shuffle)
            acc1 = test_stats.get('acc1', 0)
            acc5 = test_stats.get('acc5', 0)
            print(f"Epoch {epoch} — Top-1: {acc1:.1f}%  Top-5: {acc5:.1f}%  F1: {f1 * 100:.1f}%")
            best_f1 = max(best_f1, f1)
            print(f"Best F1 so far: {best_f1 * 100:.1f}%")
            log_stats.update({f'test_{k}': v for k, v in test_stats.items()})
            log_stats['test_f1'] = f1

        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "backbone_state_dict": model.state_dict(),
                "state_dict": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_f1": best_f1,
            }
            torch.save(save_dict,
                      os.path.join(args.output_dir, "checkpoint.pth.tar"))
            # Save best checkpoint by F1 when validation ran and F1 improved
            if (epoch % args.val_freq == 0 or epoch == args.epochs - 1) and best_f1 > prev_best:
                torch.save(save_dict,
                          os.path.join(args.output_dir, "best_checkpoint.pth.tar"))

    print(f"\nFinal best F1: {best_f1 * 100:.1f}%")
    print("See log.txt for per-epoch Top-1, Top-5, and F1 scores.")

    if (args.export_per_class_json and args.dataset == 'ssv2'
            and utils.is_main_process()):
        best_path = os.path.join(args.output_dir, "best_checkpoint.pth.tar")
        if not os.path.isfile(best_path):
            best_path = os.path.join(args.output_dir, "checkpoint.pth.tar")
        if os.path.isfile(best_path):
            ck = torch.load(best_path, map_location='cpu', weights_only=False)
            model_inner = model.module if hasattr(model, 'module') else model
            cls_inner = (classifier.module if hasattr(classifier, 'module')
                         else classifier)
            if 'backbone_state_dict' in ck:
                sd = {k.replace("module.", ""): v for k, v in
                      ck['backbone_state_dict'].items()}
                model_inner.load_state_dict(sd, strict=False)
            if 'state_dict' in ck:
                sd = {k.replace("module.", ""): v for k, v in
                      ck['state_dict'].items()}
                cls_inner.load_state_dict(sd, strict=False)
            _, _, per_class = validate(
                val_loader, model, classifier, num_labels,
                frame_shuffle=args.frame_shuffle, return_per_class=True)
            out_path = args.export_per_class_json
            _od = os.path.dirname(out_path)
            if _od:
                os.makedirs(_od, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(per_class, f, indent=2)
            print(f"Wrote per-class accuracy to {out_path}")
        else:
            print("export_per_class_json: no checkpoint found; skipping.")


def train(model, classifier, optimizer, loader, epoch):
    model.train()
    classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    for inp, target, _, _ in metric_logger.log_every(loader, 50, header):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(inp)
        output = classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _shuffle_frames(x, mode):
    """Permute frames along T dimension. x shape: (B, C, T, H, W)."""
    if mode == 'none':
        return x
    T = x.shape[2]
    if mode == 'reverse':
        idx = torch.arange(T - 1, -1, -1, device=x.device)
    elif mode == 'random':
        idx = torch.randperm(T, device=x.device)
    else:
        return x
    return x[:, :, idx, :, :]


def _per_class_accuracy_json(all_targets, all_preds, num_labels):
    """Per-class top-1 accuracy on the validation set (SSv2 many-class)."""
    y_t = np.array(all_targets, dtype=np.int64)
    y_p = np.array(all_preds, dtype=np.int64)
    out = {}
    for c in range(num_labels):
        m = y_t == c
        n = int(m.sum())
        if n == 0:
            out[str(c)] = {"support": 0, "accuracy": None}
        else:
            out[str(c)] = {
                "support": n,
                "accuracy": float((y_p[m] == c).mean()),
            }
    return out


@torch.no_grad()
def validate(val_loader, model, classifier, num_labels, frame_shuffle='none',
             return_per_class=False):
    model.eval()
    classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    all_targets, all_preds = [], []

    for inp, target, _, _ in metric_logger.log_every(val_loader, 50, 'Test:'):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        inp = _shuffle_frames(inp, frame_shuffle)

        output = model(inp)
        output = classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        topk = (1, 5) if num_labels >= 5 else (1,)
        accs = utils.accuracy(output, target, topk=topk)
        all_targets.extend(target.cpu().numpy())
        all_preds.extend(output.argmax(dim=1).cpu().numpy())

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(accs[0].item(), n=batch_size)
        if len(accs) > 1:
            metric_logger.meters['acc5'].update(accs[1].item(), n=batch_size)

    avg = 'binary' if num_labels == 2 else 'micro'
    f1 = f1_score(all_targets, all_preds, average=avg)

    stats = {k: meter.global_avg for k, meter in
             metric_logger.meters.items()}
    if return_per_class:
        pc = _per_class_accuracy_json(all_targets, all_preds, num_labels)
        return stats, f1, pc
    return stats, f1, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Student Evaluation',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    eval_student(args)
