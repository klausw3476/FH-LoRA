"""
Phase 1 — Spatial backbone distillation.

Trains the ViT-Small student (no TD-LoRA yet) to mimic the teacher's
spatial feature representations on individual video frames. The teacher
is the frozen TimeSformer ViT-Base with divided space-time attention.

Distillation losses:
  * Feature MSE at intermediate layers {3, 6, 9, 12} (0-indexed: 2, 5, 8, 11)
    after a learnable linear projection from student dim -> teacher dim.
  * CLS-token cosine similarity loss between student and teacher.

Usage:
  python distill_spatial.py \
      --teacher_weights checkpoints/TimeSformer_divST_8_224_SSv2.pyth \
      --data_path /path/to/video_data \
      --output_dir output/phase1 \
      --epochs 50 --batch_size_per_gpu 16 --lr 1e-4
"""

import argparse
import json
import math
import os
import sys
import time
import datetime
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from models.timesformer import VisionTransformer as TeacherViT
from models.student_vit import student_vit_small, student_vit_tiny, student_vit_nano, student_vit_pico
from datasets import Kinetics, Ssv2
from utils import utils
from utils.parser import load_config


class FeatureProjector(nn.Module):
    """Project student features to teacher dimension for distillation."""

    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, x):
        return self.proj(x)


class SpatialDistillationLoss(nn.Module):
    """
    Combined feature-matching + CLS-alignment loss.

    For intermediate layers the student features (dim=384) are projected up
    to the teacher dimension (dim=768) via learnable linear projections, then
    matched with MSE.
    """

    def __init__(self, student_dim=384, teacher_dim=768, n_layers=4,
                 alpha_feat=1.0, alpha_cls=1.0):
        super().__init__()
        self.projectors = nn.ModuleList(
            [FeatureProjector(student_dim, teacher_dim)
             for _ in range(n_layers)])
        self.cls_projector = FeatureProjector(student_dim, teacher_dim)
        self.alpha_feat = alpha_feat
        self.alpha_cls = alpha_cls

    def forward(self, student_intermediates, teacher_intermediates,
                student_cls, teacher_cls):
        """
        Args:
            student_intermediates: list of (BT, N+1, D_s) from layers {2,5,8,11}
            teacher_intermediates: list of (BT, N+1, D_t) from same layers
            student_cls: (B, D_s) averaged CLS token
            teacher_cls: (B, D_t) averaged CLS token
        """
        feat_loss = 0.0
        for proj, s_feat, t_feat in zip(self.projectors,
                                        student_intermediates,
                                        teacher_intermediates):
            s_proj = proj(s_feat)
            feat_loss = feat_loss + F.mse_loss(s_proj, t_feat.detach())

        cls_proj = self.cls_projector(student_cls)
        cls_loss = 1.0 - F.cosine_similarity(
            cls_proj, teacher_cls.detach(), dim=-1).mean()

        return self.alpha_feat * feat_loss + self.alpha_cls * cls_loss


class TeacherWrapper(nn.Module):
    """Wraps the teacher ViT-Base to return intermediate features."""

    def __init__(self, teacher: TeacherViT):
        super().__init__()
        self.teacher = teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        self.distill_layers = {2, 5, 8, 11}

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) video tensor.
        Returns:
            cls_token: (B, D_t)
            intermediates: list of (BT, N+1, D_t) at distill layers.
        """
        B = x.shape[0]
        x_in, T, W = self.teacher.patch_embed(x)
        cls_tokens = self.teacher.cls_token.expand(x_in.size(0), -1, -1)
        x_in = torch.cat((cls_tokens, x_in), dim=1)

        if x_in.size(1) != self.teacher.pos_embed.size(1):
            pos_embed = self.teacher.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x_in.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x_in.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W),
                                          mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2).transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x_in = x_in + new_pos_embed
        else:
            x_in = x_in + self.teacher.pos_embed
        x_in = self.teacher.pos_drop(x_in)

        if hasattr(self.teacher, 'time_embed'):
            cls_tok = x_in[:B, 0, :].unsqueeze(1)
            x_in = x_in[:, 1:]
            from einops import rearrange
            x_in = rearrange(x_in, '(b t) n m -> (b n) t m', b=B, t=T)
            if T != self.teacher.time_embed.size(1):
                time_embed = self.teacher.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T,),
                                               mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x_in = x_in + new_time_embed
            else:
                x_in = x_in + self.teacher.time_embed
            x_in = self.teacher.time_drop(x_in)
            x_in = rearrange(x_in, '(b n) t m -> b (n t) m', b=B, t=T)
            x_in = torch.cat((cls_tok, x_in), dim=1)

        intermediates = []
        for i, blk in enumerate(self.teacher.blocks):
            x_in = blk(x_in, B, T, W)
            if i in self.distill_layers:
                if self.teacher.attention_type == 'divided_space_time':
                    from einops import rearrange
                    cls_t = x_in[:, 0:1, :]
                    patch_t = x_in[:, 1:, :]
                    patch_t = rearrange(patch_t,
                                        'b (n t) d -> (b t) n d',
                                        t=T)
                    cls_t_expanded = cls_t.unsqueeze(1).expand(
                        -1, T, -1, -1).reshape(B * T, 1, -1)
                    intermediates.append(
                        torch.cat([cls_t_expanded, patch_t], dim=1))
                else:
                    intermediates.append(x_in)

        x_in = self.teacher.norm(x_in)
        cls_token = x_in[:, 0]

        return cls_token, intermediates


def get_args_parser():
    parser = argparse.ArgumentParser('Phase 1: Spatial Distillation',
                                     add_help=False)
    parser.add_argument('--teacher_weights', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', default='output/phase1', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size_per_gpu', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--clip_grad', default=3.0, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--saveckp_freq', default=10, type=int)
    parser.add_argument('--alpha_feat', default=1.0, type=float)
    parser.add_argument('--alpha_cls', default=1.0, type=float)
    parser.add_argument('--student_dim', default=384, type=int)
    parser.add_argument('--student_arch', default='small', type=str,
                        choices=['small', 'tiny', 'nano', 'pico'],
                        help='Student architecture: small (384-dim), tiny (192-dim), or nano (128-dim)')
    parser.add_argument('--use_fp16', default=True, type=utils.bool_flag)
    parser.add_argument('--dataset', default='kinetics', type=str,
                        choices=['kinetics', 'ssv2'],
                        help='Dataset to use for distillation')
    parser.add_argument('--path_prefix', default=None, type=str,
                        help='Override DATA.PATH_PREFIX (e.g. path to frames)')

    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--local_rank", default=0, type=int)

    parser.add_argument("--cfg", dest="cfg_file", type=str,
                        default="models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)

    return parser


def train_phase1(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if utils.is_main_process():
        with open(Path(args.output_dir) / "config.json", "w") as f:
            json.dump(vars(args), f, indent=4)

    config = load_config(args)
    config.DATA.PATH_TO_DATA_DIR = args.data_path
    if args.path_prefix:
        config.DATA.PATH_PREFIX = args.path_prefix
    elif not config.DATA.PATH_PREFIX:
        config.DATA.PATH_PREFIX = "."
    config.DATA.NO_RGB_AUG = True

    if args.dataset == 'ssv2':
        config.DATA.RANDOM_FLIP = False
        dataset = Ssv2(cfg=config, mode="train", num_retries=10)
    else:
        dataset = Kinetics(cfg=config, mode="train", num_retries=10)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print(f"Training data loaded: {len(dataset)} videos.")

    # ---- Build teacher (frozen) ----
    ckpt = torch.load(args.teacher_weights, map_location='cpu', weights_only=False)
    if "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    elif "teacher" in ckpt:
        ckpt = ckpt["teacher"]
    state_dict = {k.replace("module.", "").replace("backbone.", "").replace("model.", ""): v
                  for k, v in ckpt.items()}

    teacher_num_frames = config.DATA.NUM_FRAMES
    if "time_embed" in state_dict:
        teacher_num_frames = state_dict["time_embed"].shape[1]
        print(f"Teacher checkpoint uses {teacher_num_frames} frames")

    teacher_vit = TeacherViT(
        img_size=config.DATA.TRAIN_CROP_SIZE,
        num_classes=config.MODEL.NUM_CLASSES,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        num_frames=teacher_num_frames,
        attention_type=config.TIMESFORMER.ATTENTION_TYPE)

    msg = teacher_vit.load_state_dict(state_dict, strict=False)
    print(f"Teacher loaded: {msg}")

    teacher = TeacherWrapper(teacher_vit).cuda()
    teacher.eval()

    # ---- Build student ----
    _student_factories = {'small': student_vit_small, 'tiny': student_vit_tiny, 'nano': student_vit_nano, 'pico': student_vit_pico}
    student = _student_factories[args.student_arch](
        img_size=config.DATA.TRAIN_CROP_SIZE,
        use_tdlora=False).cuda()
    args.student_dim = student.embed_dim

    student = nn.parallel.DistributedDataParallel(
        student, device_ids=[args.gpu], find_unused_parameters=False)
    print(f"Student built: ViT-{args.student_arch} (embed_dim={args.student_dim})")

    # ---- Loss ----
    criterion = SpatialDistillationLoss(
        student_dim=args.student_dim, teacher_dim=768,
        alpha_feat=args.alpha_feat, alpha_cls=args.alpha_cls).cuda()

    # ---- Optimizer ----
    param_groups = [
        {'params': list(student.parameters()), 'lr': args.lr},
        {'params': list(criterion.parameters()), 'lr': args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    lr_schedule = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs)

    # ---- Resume ----
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student, optimizer=optimizer,
        fp16_scaler=fp16_scaler, criterion=criterion)
    start_epoch = to_restore["epoch"]

    # ---- Training loop ----
    best_loss = float('inf')
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            student, teacher, criterion, data_loader, optimizer,
            lr_schedule, epoch, fp16_scaler, args)

        save_dict = {
            'student': student.state_dict(),
            'criterion': criterion.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict,
                             os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and (epoch + 1) % args.saveckp_freq == 0:
            utils.save_on_master(
                save_dict,
                os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        epoch_loss = train_stats.get('loss', float('inf'))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            utils.save_on_master(save_dict,
                                 os.path.join(args.output_dir, 'checkpoint_best.pth'))
            print(f"  => New best loss: {best_loss:.6f} (epoch {epoch})")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'best_loss': best_loss}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = str(datetime.timedelta(
        seconds=int(time.time() - start_time)))
    print(f'Phase 1 training complete in {total_time}')


def train_one_epoch(student, teacher, criterion, data_loader, optimizer,
                    lr_schedule, epoch, fp16_scaler, args):
    student.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Phase1 Epoch [{epoch}/{args.epochs}]'

    for it, (videos, _, _, _) in enumerate(
            metric_logger.log_every(data_loader, 50, header)):
        global_it = len(data_loader) * epoch + it
        for pg in optimizer.param_groups:
            pg["lr"] = lr_schedule[global_it]

        videos = videos.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_cls, teacher_intermediates = teacher(videos)
            student_cls, _, student_intermediates = \
                student(videos, return_intermediate=True)

            loss = criterion(student_intermediates, teacher_intermediates,
                             student_cls, teacher_cls)

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training", flush=True)
            sys.exit(1)

        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                for model in [student, criterion]:
                    utils.clip_gradients(model, args.clip_grad)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for model in [student, criterion]:
                    utils.clip_gradients(model, args.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Phase 1 Distillation',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    train_phase1(args)
