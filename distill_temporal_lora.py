"""
Phase 2 — Temporal LoRA distillation.

Loads the Phase-1 pre-trained ViT-Small backbone (spatial knowledge already
distilled), freezes it, inserts TD-LoRA adapters, and trains ONLY the LoRA
parameters + gating MLPs to recover temporal information from the teacher.

The distillation target is the teacher's block outputs after both temporal
and spatial attention, so the TD-LoRA learns to compensate for the removed
temporal attention.

Optionally supports a Phase 3 joint fine-tuning stage where the entire
student (backbone + TD-LoRA) is unfrozen and trained end-to-end on a
downstream task with combined distillation + task loss.

Temporal distillation losses (Proposal 2 from tdlora_analysis.md):
  - Temporal Order Sensitivity: MSE(student_diff, teacher_diff) for reversed frames
  - Frame Importance Matching: KL(student_gating_importance, teacher_attn_importance)
  - Frame Similarity Matching: MSE(student_gating_sim, teacher_attn_sim)

Usage:
  # Phase 2 only:
  python distill_temporal_lora.py \\
      --teacher_weights checkpoints/TimeSformer_divST_8_224_SSv2.pyth \\
      --student_weights output/phase1/checkpoint_best.pth \\
      --data_path /path/to/video_data \\
      --output_dir output/phase2 \\
      --epochs 30 --lr 5e-4 --lora_rank 8

  # Phase 3 joint fine-tuning:
  python distill_temporal_lora.py \\
      --teacher_weights checkpoints/TimeSformer_divST_8_224_SSv2.pyth \\
      --student_weights output/phase2/checkpoint_best.pth \\
      --data_path /path/to/video_data \\
      --output_dir output/phase3 \\
      --phase3 --task_num_classes 2 \\
      --epochs 20 --lr 1e-5 --beta_distill 0.5
"""

import utils.checkpoint_compat  # noqa: F401 — numpy/torch checkpoint pickle compat (must be first)

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
from einops import rearrange

from utils import utils
from utils.parser import load_config


def _proposal2_teacher_targets(teacher_temporal_attn):
    """Single pass over (B, T, T) attention for importance + similarity losses."""
    t = teacher_temporal_attn.detach()
    importance = F.softmax(t.sum(dim=-2), dim=-1)
    sym = (t + t.transpose(-1, -2)) * 0.5
    sim_target = sym / sym.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return importance, sim_target


class FeatureProjector(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, x):
        return self.proj(x)


# -----------------------------------------------------------------------
# Proposal 2A: Temporal Order Sensitivity Loss
# -----------------------------------------------------------------------

class TemporalOrderLoss(nn.Module):
    """
    Match student's sensitivity to temporal reversal with teacher's.

    L_order = MSE(student_fwd - student_rev, teacher_fwd - teacher_rev)

    The teacher's response difference encodes which actions are
    order-sensitive vs order-invariant. The student should reproduce
    this sensitivity pattern, not just push forward/reversed apart.
    """

    def forward(self, student_cls_fwd: torch.Tensor,
                student_cls_rev: torch.Tensor,
                teacher_cls_fwd: torch.Tensor,
                teacher_cls_rev: torch.Tensor) -> torch.Tensor:
        teacher_diff = teacher_cls_fwd.detach() - teacher_cls_rev.detach()
        student_diff = student_cls_fwd - student_cls_rev
        return F.mse_loss(student_diff, teacher_diff)


# -----------------------------------------------------------------------
# Proposal 2B: Teacher Temporal Attention Distillation
# -----------------------------------------------------------------------

class FrameImportanceMatchingLoss(nn.Module):
    """
    Align student gating importance with teacher temporal attention importance.

    Teacher frame importance: column-sum of temporal attention matrix
        (how much each frame is attended to by all other frames)
    Student frame importance: mean gating magnitude per frame
        (how much each frame's LoRA adaptation contributes)
    Loss: KL divergence between importance distributions.
    Guard: only apply when teacher importance is non-uniform.
    """

    def forward(self, student_gates: torch.Tensor,
                teacher_temporal_attn: torch.Tensor = None,
                teacher_importance_probs: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            student_gates: (B, T, rank) student's s(t) gating per frame
            teacher_temporal_attn: (B, T, T) raw teacher temporal attention
            teacher_importance_probs: (B, T) teacher importance distribution
                (softmax over frames); if set, teacher_temporal_attn is ignored
        """
        if teacher_importance_probs is None:
            if teacher_temporal_attn is None:
                raise ValueError(
                    'FrameImportanceMatchingLoss needs teacher_temporal_attn '
                    'or teacher_importance_probs')
            teacher_importance = F.softmax(
                teacher_temporal_attn.sum(dim=-2), dim=-1)
        else:
            teacher_importance = teacher_importance_probs

        student_importance = student_gates.mean(dim=-1)
        student_importance = F.softmax(student_importance, dim=-1)

        teacher_entropy = -(teacher_importance *
                            teacher_importance.log().clamp(min=-100)).sum(dim=-1)
        max_entropy = math.log(teacher_importance.shape[-1])
        entropy_mask = (teacher_entropy < 0.9 * max_entropy).float()

        if entropy_mask.sum() == 0:
            return student_gates.new_zeros(())

        loss = F.kl_div(
            student_importance.clamp(min=1e-8).log(), teacher_importance,
            reduction='none'
        ).sum(dim=-1)
        loss = (loss * entropy_mask).sum() / entropy_mask.sum().clamp(min=1)
        return loss


class FrameSimilarityMatchingLoss(nn.Module):
    """
    Match frame-frame relationship structure.

    If the teacher says frames 2 and 5 are related (high mutual attention),
    the student's gating for frames 2 and 5 should be similar.

    Teacher: symmetrized attention matrix
    Student: cosine similarity in gating space
    Loss: MSE between similarity matrices
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_gates: torch.Tensor,
                teacher_temporal_attn: torch.Tensor = None,
                teacher_sim_target: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            student_gates: (B, T, rank)
            teacher_temporal_attn: (B, T, T) raw attention (optional if
                teacher_sim_target is passed)
            teacher_sim_target: (B, T, T) row-normalized symmetrized target
        """
        if teacher_sim_target is None:
            if teacher_temporal_attn is None:
                raise ValueError(
                    'FrameSimilarityMatchingLoss needs teacher_temporal_attn '
                    'or teacher_sim_target')
            teacher_sim = (teacher_temporal_attn +
                           teacher_temporal_attn.transpose(-1, -2)) * 0.5
            teacher_sim = teacher_sim / teacher_sim.sum(
                dim=-1, keepdim=True).clamp(min=1e-8)
            teacher_sim = teacher_sim.detach()
        else:
            teacher_sim = teacher_sim_target

        s_norm = F.normalize(student_gates, dim=-1)
        student_sim = torch.bmm(s_norm, s_norm.transpose(-1, -2))
        student_sim = F.softmax(student_sim / self.temperature, dim=-1)

        return F.mse_loss(student_sim, teacher_sim)


# -----------------------------------------------------------------------
# Proposal 2C: Temporal Contrastive Loss
# -----------------------------------------------------------------------

class TemporalContrastiveLoss(nn.Module):
    """
    Contrastive loss: temporally transformed videos (reversed, shuffled)
    should produce different representations from the original.

    Anchor:    student CLS from forward pass
    Negatives: student CLS from reversed frames, shuffled frames
               (extracted with stop-gradients on backbone)

    Uses InfoNCE loss with a learned projection head.
    """

    def __init__(self, feat_dim: int = 384, proj_dim: int = 128,
                 temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, anchor_cls: torch.Tensor,
                neg_cls_list: list) -> torch.Tensor:
        """
        Args:
            anchor_cls: (B, D) student CLS from original video
            neg_cls_list: list of (B, D) student CLS from transformed videos
                          (reversed, shuffled, etc). Detached from backbone.
        """
        n_neg = len(neg_cls_list)
        if n_neg == 0:
            return anchor_cls.new_zeros(())

        pieces = [anchor_cls] + [c.detach() for c in neg_cls_list]
        flat = torch.cat(pieces, dim=0)
        z_flat = F.normalize(self.projector(flat), dim=-1)
        z = z_flat.view(n_neg + 1, anchor_cls.shape[0], -1)
        z_anchor = z[0]
        neg = z[1:].transpose(0, 1).contiguous()
        logits_neg = torch.bmm(neg, z_anchor.unsqueeze(-1)).squeeze(-1)
        logits_pos = (z_anchor * z_anchor).sum(dim=-1, keepdim=True)
        logits = torch.cat([logits_pos, logits_neg], dim=-1) / self.temperature
        labels = torch.zeros(anchor_cls.shape[0], dtype=torch.long,
                             device=anchor_cls.device)

        return F.cross_entropy(logits, labels)


# -----------------------------------------------------------------------
# Combined Distillation Loss
# -----------------------------------------------------------------------

class TemporalDistillationLoss(nn.Module):
    """
    Feature-matching loss for Phase 2.

    Matches TD-LoRA-enhanced student features against teacher features
    that have been processed through both temporal and spatial attention.

    Optionally includes temporal distillation losses (Proposal 2):
    - order: MSE(student_diff, teacher_diff) for reversed frames
    - importance: KL between frame importance distributions
    - similarity: MSE between frame similarity matrices
    """

    def __init__(self, student_dim=384, teacher_dim=768, n_layers=4,
                 alpha_feat=1.0, alpha_cls=1.0,
                 use_temporal_order_loss=False, alpha_order=0.5,
                 use_importance_loss=False, alpha_importance=1.0,
                 use_similarity_loss=False, alpha_similarity=1.0,
                 sim_temperature=0.1,
                 use_contrastive_loss=False, alpha_contrastive=0.3,
                 contrastive_proj_dim=128):
        super().__init__()
        self.projectors = nn.ModuleList(
            [FeatureProjector(student_dim, teacher_dim)
             for _ in range(n_layers)])
        self.cls_projector = FeatureProjector(student_dim, teacher_dim)
        self.alpha_feat = alpha_feat
        self.alpha_cls = alpha_cls

        self.use_temporal_order_loss = use_temporal_order_loss
        self.alpha_order = alpha_order
        if use_temporal_order_loss:
            self.order_loss = TemporalOrderLoss()

        self.use_importance_loss = use_importance_loss
        self.alpha_importance = alpha_importance
        if use_importance_loss:
            self.importance_loss = FrameImportanceMatchingLoss()

        self.use_similarity_loss = use_similarity_loss
        self.alpha_similarity = alpha_similarity
        if use_similarity_loss:
            self.similarity_loss = FrameSimilarityMatchingLoss(
                temperature=sim_temperature)

        self.use_contrastive_loss = use_contrastive_loss
        self.alpha_contrastive = alpha_contrastive
        if use_contrastive_loss:
            self.contrastive_loss = TemporalContrastiveLoss(
                feat_dim=student_dim, proj_dim=contrastive_proj_dim)

    def forward(self, student_intermediates, teacher_intermediates,
                student_cls, teacher_cls,
                student_cls_rev=None, teacher_cls_rev=None,
                student_gates=None, teacher_temporal_attn=None,
                contrastive_neg_cls=None,
                imp_sim_ramp=1.0):
        feat_loss = 0.0
        for proj, s_feat, t_feat in zip(self.projectors,
                                        student_intermediates,
                                        teacher_intermediates):
            s_proj = proj(s_feat)
            feat_loss = feat_loss + F.mse_loss(s_proj, t_feat.detach())

        cls_proj = self.cls_projector(student_cls)
        cls_loss = 1.0 - F.cosine_similarity(
            cls_proj, teacher_cls.detach(), dim=-1).mean()

        total = self.alpha_feat * feat_loss + self.alpha_cls * cls_loss

        if (self.use_temporal_order_loss
                and student_cls_rev is not None
                and teacher_cls_rev is not None):
            student_cls_proj_rev = self.cls_projector(student_cls_rev)
            total = total + self.alpha_order * self.order_loss(
                cls_proj, student_cls_proj_rev,
                teacher_cls, teacher_cls_rev)

        t_imp_probs = None
        t_sim_tgt = None
        if (student_gates is not None and teacher_temporal_attn is not None
                and (self.use_importance_loss or self.use_similarity_loss)):
            if self.use_importance_loss and self.use_similarity_loss:
                t_imp_probs, t_sim_tgt = _proposal2_teacher_targets(
                    teacher_temporal_attn)
            elif self.use_importance_loss:
                t = teacher_temporal_attn.detach()
                t_imp_probs = F.softmax(t.sum(dim=-2), dim=-1)
            else:
                t = teacher_temporal_attn.detach()
                sym = (t + t.transpose(-1, -2)) * 0.5
                t_sim_tgt = sym / sym.sum(dim=-1, keepdim=True).clamp(
                    min=1e-8)

        if (self.use_importance_loss
                and student_gates is not None
                and teacher_temporal_attn is not None):
            total = total + (
                imp_sim_ramp * self.alpha_importance
                * self.importance_loss(
                    student_gates,
                    teacher_importance_probs=t_imp_probs,
                    teacher_temporal_attn=(
                        None if t_imp_probs is not None
                        else teacher_temporal_attn)))

        if (self.use_similarity_loss
                and student_gates is not None
                and teacher_temporal_attn is not None):
            total = total + (
                imp_sim_ramp * self.alpha_similarity
                * self.similarity_loss(
                    student_gates,
                    teacher_sim_target=t_sim_tgt,
                    teacher_temporal_attn=(
                        None if t_sim_tgt is not None
                        else teacher_temporal_attn)))

        if (self.use_contrastive_loss
                and contrastive_neg_cls is not None
                and len(contrastive_neg_cls) > 0):
            total = total + self.alpha_contrastive * self.contrastive_loss(
                student_cls, contrastive_neg_cls)

        return total


class TeacherWrapper(nn.Module):
    """Frozen teacher returning intermediate features (same as Phase 1)."""

    def __init__(self, teacher: TeacherViT, extract_temporal_attn=False):
        super().__init__()
        self.teacher = teacher
        self.extract_temporal_attn = extract_temporal_attn
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        self.distill_layers = {2, 5, 8, 11}

    @torch.no_grad()
    def forward(self, x):
        B = x.shape[0]
        temporal_attn_maps = [] if self.extract_temporal_attn else None

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
            cap = temporal_attn_maps if (
                self.extract_temporal_attn and i in self.distill_layers
                and getattr(blk, 'attention_type', None) == 'divided_space_time'
            ) else None
            x_in = blk(x_in, B, T, W, capture_temporal_attn=cap)
            if i in self.distill_layers:
                if self.teacher.attention_type == 'divided_space_time':
                    cls_t = x_in[:, 0:1, :]
                    patch_t = x_in[:, 1:, :]
                    patch_t = rearrange(patch_t, 'b (n t) d -> (b t) n d',
                                        t=T)
                    cls_t_expanded = cls_t.unsqueeze(1).expand(
                        -1, T, -1, -1).reshape(B * T, 1, -1)
                    intermediates.append(
                        torch.cat([cls_t_expanded, patch_t], dim=1))
                else:
                    intermediates.append(x_in)

        x_in = self.teacher.norm(x_in)
        cls_token = x_in[:, 0]

        temporal_attn = None
        if temporal_attn_maps:
            stacked = torch.stack(temporal_attn_maps, dim=0)
            temporal_attn = stacked.mean(dim=0)
            temporal_attn = rearrange(temporal_attn, '(b n) t1 t2 -> b n t1 t2',
                                      b=B)
            temporal_attn = temporal_attn.mean(dim=1)

        return cls_token, intermediates, temporal_attn


def get_args_parser():
    parser = argparse.ArgumentParser('Phase 2/3: Temporal LoRA Distillation',
                                     add_help=False)
    parser.add_argument('--teacher_weights', type=str, required=True)
    parser.add_argument('--student_weights', type=str, required=True,
                        help='Path to Phase-1 student checkpoint')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', default='output/phase2', type=str)

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

    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size_per_gpu', default=16, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_epochs', default=3, type=int)
    parser.add_argument('--clip_grad', default=3.0, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--saveckp_freq', default=5, type=int)
    parser.add_argument('--alpha_feat', default=1.0, type=float)
    parser.add_argument('--alpha_cls', default=1.0, type=float)

    # Proposal 2A: Temporal Order Sensitivity Loss
    parser.add_argument('--use_temporal_order_loss', action='store_true',
                        help='Match teacher response difference for reversed frames')
    parser.add_argument('--alpha_order', default=0.5, type=float,
                        help='Weight for temporal order loss')
    # Proposal 2B: Frame Importance Matching Loss
    parser.add_argument('--use_importance_loss', action='store_true',
                        help='Match teacher frame importance via gating values')
    parser.add_argument('--alpha_importance', default=1.0, type=float,
                        help='Weight for frame importance matching loss')
    # Proposal 2B: Frame Similarity Matching Loss
    parser.add_argument('--use_similarity_loss', action='store_true',
                        help='Match teacher frame similarity via gating cosine sim')
    parser.add_argument('--alpha_similarity', default=1.0, type=float,
                        help='Weight for frame similarity matching loss')
    parser.add_argument('--sim_temperature', default=0.1, type=float,
                        help='Temperature for softmax on student frame sim '
                             '(Proposal 2B similarity loss)')
    parser.add_argument('--imp_sim_warmup_epochs', default=0, type=int,
                        help='Linearly ramp importance+similarity loss from 0 '
                             'to full over this many epochs (0 = off)')
    # Proposal 2C: Temporal Contrastive Loss
    parser.add_argument('--use_contrastive_loss', action='store_true',
                        help='InfoNCE on original vs temporal transforms')
    parser.add_argument('--alpha_contrastive', default=0.3, type=float,
                        help='Weight for temporal contrastive loss')
    parser.add_argument('--contrastive_proj_dim', default=128, type=int,
                        help='Projection dim for contrastive head')

    parser.add_argument('--student_arch', default='small', type=str,
                        choices=['small', 'tiny', 'nano', 'pico'],
                        help='Student architecture: small (384-dim), tiny (192-dim), or nano (128-dim)')
    parser.add_argument('--use_fp16', default=True, type=utils.bool_flag)
    parser.add_argument('--dataset', default='kinetics', type=str,
                        choices=['kinetics', 'ssv2'],
                        help='Dataset to use for distillation')
    parser.add_argument('--path_prefix', default=None, type=str,
                        help='Override DATA.PATH_PREFIX (e.g. path to frames)')

    # Phase 3 options
    parser.add_argument('--phase3', action='store_true',
                        help='Enable Phase 3 joint fine-tuning')
    parser.add_argument('--task_num_classes', default=0, type=int,
                        help='Task head classes; 0 = use config.MODEL.NUM_CLASSES (Kinetics)')
    parser.add_argument('--beta_distill', default=0.5, type=float,
                        help='Weight of distillation loss in Phase 3')
    parser.add_argument('--backbone_lr_scale', default=0.1, type=float,
                        help='LR scale for backbone params in Phase 3')

    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--local_rank", default=0, type=int)

    parser.add_argument("--cfg", dest="cfg_file", type=str,
                        default="models/configs/Kinetics/"
                                "TimeSformer_divST_8x32_224.yaml")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)

    return parser


def train_phase2(args):
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

    extract_temporal_attn = args.use_importance_loss or args.use_similarity_loss
    teacher = TeacherWrapper(
        teacher_vit,
        extract_temporal_attn=extract_temporal_attn).cuda()

    # ---- Build student with TD-LoRA ----
    _student_factories = {'small': student_vit_small, 'tiny': student_vit_tiny,
                          'nano': student_vit_nano, 'pico': student_vit_pico}
    student = _student_factories[args.student_arch](
        img_size=config.DATA.TRAIN_CROP_SIZE,
        use_tdlora=False).cuda()

    phase1_ckpt = torch.load(args.student_weights, map_location='cpu')
    if 'student' in phase1_ckpt:
        student_state = phase1_ckpt['student']
    else:
        student_state = phase1_ckpt
    student_state = {k.replace("module.", ""): v
                     for k, v in student_state.items()}
    has_lora_in_ckpt = any(
        'lora_' in k or 'gating' in k for k in student_state.keys())
    if has_lora_in_ckpt:
        student.enable_tdlora(rank=args.lora_rank,
                              time_embed_dim=args.time_embed_dim,
                              lora_type=args.lora_type,
                              hyper_hidden_dim=args.hyper_hidden_dim,
                              content_dim=args.content_dim,
                              content_dropout=args.content_dropout)
    msg = student.load_state_dict(student_state, strict=False)
    print(f"Phase-1 student loaded: {msg}")
    if not has_lora_in_ckpt:
        student.enable_tdlora(rank=args.lora_rank,
                              time_embed_dim=args.time_embed_dim,
                              lora_type=args.lora_type,
                              hyper_hidden_dim=args.hyper_hidden_dim,
                              content_dim=args.content_dim,
                              content_dropout=args.content_dropout)

    if args.phase3:
        student.unfreeze_all()
        print("Phase 3: all parameters unfrozen for joint fine-tuning")
    else:
        student.freeze_backbone()
        print("Phase 2: backbone frozen, only TD-LoRA parameters trainable")

    n_lora = sum(p.numel() for p in student.lora_parameters())
    n_total = sum(p.numel() for p in student.parameters())
    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"LoRA params: {n_lora:,} | Total: {n_total:,} | "
          f"Trainable: {n_trainable:,}")

    student_embed_dim = student.embed_dim

    student = nn.parallel.DistributedDataParallel(
        student, device_ids=[args.gpu], find_unused_parameters=True)

    # ---- Loss ----
    criterion = TemporalDistillationLoss(
        student_dim=student_embed_dim, teacher_dim=768,
        alpha_feat=args.alpha_feat, alpha_cls=args.alpha_cls,
        use_temporal_order_loss=args.use_temporal_order_loss,
        alpha_order=args.alpha_order,
        use_importance_loss=args.use_importance_loss,
        alpha_importance=args.alpha_importance,
        use_similarity_loss=args.use_similarity_loss,
        alpha_similarity=args.alpha_similarity,
        sim_temperature=args.sim_temperature,
        use_contrastive_loss=args.use_contrastive_loss,
        alpha_contrastive=args.alpha_contrastive,
        contrastive_proj_dim=args.contrastive_proj_dim).cuda()

    task_head = None
    task_criterion = None
    if args.phase3:
        task_num_classes = (
            args.task_num_classes
            if args.task_num_classes > 0
            else config.MODEL.NUM_CLASSES
        )
        if task_num_classes <= 0 and utils.is_main_process():
            print("Warning: Phase 3 task_num_classes is 0 (config.MODEL.NUM_CLASSES). "
                  "No task head; only distillation loss will be used.")
        if task_num_classes > 0:
            task_head = nn.Linear(student_embed_dim, task_num_classes).cuda()
            task_head = nn.parallel.DistributedDataParallel(
                task_head, device_ids=[args.gpu])
            task_criterion = nn.CrossEntropyLoss()

    # ---- Optimizer ----
    param_list = []
    if args.phase3:
        param_list.append({
            'params': list(student.module.backbone_parameters()),
            'lr': args.lr * args.backbone_lr_scale,
            '_lr_scale': args.backbone_lr_scale,
        })
        param_list.append({
            'params': list(student.module.lora_parameters()),
            'lr': args.lr,
            '_lr_scale': 1.0,
        })
    else:
        param_list.append({
            'params': list(student.module.lora_parameters()),
            'lr': args.lr,
            '_lr_scale': 1.0,
        })
    param_list.append({
        'params': list(criterion.parameters()),
        'lr': args.lr,
        '_lr_scale': 1.0,
    })
    if task_head is not None:
        param_list.append({
            'params': list(task_head.parameters()),
            'lr': args.lr,
            '_lr_scale': 1.0,
        })

    optimizer = torch.optim.AdamW(param_list,
                                  weight_decay=args.weight_decay)
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    lr_schedule = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs)

    to_restore = {"epoch": 0}
    resume_path = os.path.join(args.output_dir, 'checkpoint.pth')
    utils.restart_from_checkpoint(
        resume_path,
        run_variables=to_restore,
        student=student,
        criterion=criterion,
        optimizer=optimizer,
        **({"fp16_scaler": fp16_scaler} if fp16_scaler is not None else {}),
        **({"task_head": task_head} if task_head is not None else {}),
    )
    start_epoch = to_restore["epoch"]

    # ---- Training loop ----
    best_loss = float('inf')
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            student, teacher, criterion, data_loader, optimizer,
            lr_schedule, epoch, fp16_scaler, args,
            task_head=task_head, task_criterion=task_criterion)

        save_dict = {
            'student': student.state_dict(),
            'criterion': criterion.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }
        if task_head is not None:
            save_dict['task_head'] = task_head.state_dict()
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
    phase = "Phase 3" if args.phase3 else "Phase 2"
    print(f'{phase} training complete in {total_time}')


def train_one_epoch(student, teacher, criterion, data_loader, optimizer,
                    lr_schedule, epoch, fp16_scaler, args,
                    task_head=None, task_criterion=None):
    student.train()
    if task_head is not None:
        task_head.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    phase = "Phase3" if args.phase3 else "Phase2"
    header = f'{phase} Epoch [{epoch}/{args.epochs}]'

    need_reversed = args.use_temporal_order_loss or args.use_contrastive_loss
    need_gates = args.use_importance_loss or args.use_similarity_loss
    need_contrastive = args.use_contrastive_loss

    for it, batch in enumerate(
            metric_logger.log_every(data_loader, 50, header)):
        videos = batch[0] if isinstance(batch, (list, tuple)) else batch
        global_it = len(data_loader) * epoch + it
        for pg in optimizer.param_groups:
            pg["lr"] = lr_schedule[global_it] * pg.get('_lr_scale', 1.0)

        videos = videos.cuda(non_blocking=True)
        T_vid = videos.shape[2]

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # 1. Forward passes (original order)
            teacher_cls, teacher_intermediates, teacher_temporal_attn = teacher(videos)
            student_cls, _, student_intermediates = \
                student(videos, return_intermediate=True)

            # 2. Collect gating values BEFORE reversed pass overwrites them
            student_gates = None
            if need_gates:
                B_vid = videos.shape[0]
                student_gates = student.module.collect_gating_values(B_vid, T_vid)

            # 3. Reversed passes (Proposal 2A: order sensitivity)
            student_cls_rev = None
            teacher_cls_rev = None
            if need_reversed:
                videos_rev = videos.flip(dims=[2])
                teacher_cls_rev, _, _ = teacher(videos_rev)
                t_rev = torch.arange(
                    T_vid - 1, -1, -1,
                    device=videos.device, dtype=torch.float32)
                student_cls_rev, _ = student.module.forward_features(
                    videos_rev, t_override=t_rev)

            # 4. Contrastive negatives (Proposal 2C)
            contrastive_neg_cls = None
            if need_contrastive:
                contrastive_neg_cls = []
                if student_cls_rev is not None:
                    contrastive_neg_cls.append(student_cls_rev)
                perm = torch.randperm(T_vid, device=videos.device)
                videos_shuf = videos[:, :, perm, :, :]
                with torch.no_grad():
                    t_shuf = perm.float()
                    cls_shuf, _ = student.module.forward_features(
                        videos_shuf, t_override=t_shuf)
                contrastive_neg_cls.append(cls_shuf)

            # 5. Compute loss (optional linear ramp for imp+sim only)
            wu = getattr(args, 'imp_sim_warmup_epochs', 0) or 0
            if wu > 0:
                imp_sim_ramp = min(1.0, float(epoch + 1) / float(wu))
            else:
                imp_sim_ramp = 1.0

            distill_loss = criterion(
                student_intermediates, teacher_intermediates,
                student_cls, teacher_cls,
                student_cls_rev=student_cls_rev,
                teacher_cls_rev=teacher_cls_rev,
                student_gates=student_gates,
                teacher_temporal_attn=teacher_temporal_attn,
                contrastive_neg_cls=contrastive_neg_cls,
                imp_sim_ramp=imp_sim_ramp)

            loss = distill_loss

            if args.phase3 and task_head is not None and task_criterion is not None:
                labels = batch[1].cuda(non_blocking=True)
                logits = task_head(student_cls)
                task_loss = task_criterion(logits, labels)
                loss = task_loss + args.beta_distill * distill_loss
                metric_logger.update(task_loss=task_loss.item())

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training", flush=True)
            sys.exit(1)

        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                for model in [student, criterion] + ([task_head] if task_head is not None else []):
                    utils.clip_gradients(model, args.clip_grad)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for model in [student, criterion] + ([task_head] if task_head is not None else []):
                    utils.clip_gradients(model, args.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(distill_loss=distill_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Phase 2/3 TD-LoRA Distillation',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    train_phase2(args)
