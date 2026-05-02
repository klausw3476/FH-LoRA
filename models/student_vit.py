"""
Spatial-only ViT-Small student model with optional TD-LoRA adapters.

The student processes each video frame independently through spatial-only
attention (no temporal MHSA), producing per-frame features.  When TD-LoRAs
are enabled, the Q/K/V projections receive time-dependent residuals so that
spatial attention implicitly adapts as a function of the frame index.

Architecture choices match the teacher's conventions (PatchEmbed layout,
CLS token, positional embedding interpolation) to simplify distillation.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.vit_utils import DropPath, trunc_normal_
from models.td_lora import StandardLoRA, PerProjLoRA, FHLoRA, FHTrunk


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TDLoRAAttention(nn.Module):
    """
    Multi-head self-attention with optional TD-LoRA on Q, K, V projections.

    When `use_tdlora=False` this is a plain MHSA identical to the teacher's
    spatial attention — useful for Phase-1 distillation before LoRA insertion.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.,
                 use_tdlora=False, lora_rank=8, time_embed_dim=64,
                 lora_type='standard', block_index=0, total_blocks=12,
                 hyper_hidden_dim=8, content_dim=16, content_dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_tdlora = use_tdlora
        self.lora_type = lora_type
        if use_tdlora:
            self._init_lora(dim, lora_rank, time_embed_dim, lora_type,
                            hyper_hidden_dim=hyper_hidden_dim)

    def _init_lora(self, dim, lora_rank, time_embed_dim, lora_type,
                   hyper_hidden_dim=8):
        if lora_type == 'fh_lora':
            self.shared_trunk = FHTrunk(
                rank=lora_rank, time_embed_dim=time_embed_dim,
                trunk_dim=hyper_hidden_dim)
            self.lora_q = FHLoRA(
                dim, dim, rank=lora_rank, time_embed_dim=time_embed_dim,
                proj_name='q')
            self.lora_k = FHLoRA(
                dim, dim, rank=lora_rank, time_embed_dim=time_embed_dim,
                proj_name='k')
            self.lora_v = FHLoRA(
                dim, dim, rank=lora_rank, time_embed_dim=time_embed_dim,
                proj_name='v')
            self.lora_q.set_trunk(self.shared_trunk)
            self.lora_k.set_trunk(self.shared_trunk)
            self.lora_v.set_trunk(self.shared_trunk)
            return

        if lora_type == 'standard':
            LoRAClass, extra_kwargs = StandardLoRA, {}
        elif lora_type == 'per_projection':
            LoRAClass = PerProjLoRA
            extra_kwargs = {'hyper_hidden_dim': hyper_hidden_dim}
        else:
            raise ValueError(
                f"Unknown lora_type={lora_type!r}; expected one of "
                "'standard', 'per_projection', 'fh_lora'."
            )

        self.lora_q = LoRAClass(dim, dim, rank=lora_rank,
                                time_embed_dim=time_embed_dim, **extra_kwargs)
        self.lora_k = LoRAClass(dim, dim, rank=lora_rank,
                                time_embed_dim=time_embed_dim, **extra_kwargs)
        self.lora_v = LoRAClass(dim, dim, rank=lora_rank,
                                time_embed_dim=time_embed_dim, **extra_kwargs)

    def forward(self, x, t=None, prev_cls=None):
        """
        Args:
            x: (BT, N, D) — all frames stacked on the batch axis.
            t: (T,) frame indices.  Required only when use_tdlora=True.
            prev_cls: (B, T, D) CLS from previous block, for content-aware variants.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_tdlora and t is not None:
            dq = self.lora_q(x, t, prev_cls=prev_cls).reshape(
                B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            dk = self.lora_k(x, t, prev_cls=prev_cls).reshape(
                B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            dv = self.lora_v(x, t, prev_cls=prev_cls).reshape(
                B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = q + dq
            k = k + dk
            v = v + dv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class StudentBlock(nn.Module):
    """Spatial-only transformer block with optional TD-LoRA."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_tdlora=False, lora_rank=8, time_embed_dim=64,
                 lora_type='standard', block_index=0, total_blocks=12,
                 hyper_hidden_dim=8, content_dim=16, content_dropout=0.3):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TDLoRAAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
            use_tdlora=use_tdlora, lora_rank=lora_rank,
            time_embed_dim=time_embed_dim, lora_type=lora_type,
            block_index=block_index, total_blocks=total_blocks,
            hyper_hidden_dim=hyper_hidden_dim,
            content_dim=content_dim, content_dropout=content_dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, t=None, prev_cls=None):
        x = x + self.drop_path(self.attn(self.norm1(x), t=t, prev_cls=prev_cls))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding — mirrors the teacher's implementation."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W_patches = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W_patches


class StudentViT(nn.Module):
    """
    Spatial-only ViT student for video, with optional TD-LoRA adapters.

    Key differences from the teacher (TimeSformer ViT-Base):
      - No temporal attention blocks
      - Smaller embed_dim (384 vs 768)
      - Fewer heads (6 vs 12)
      - Optional TD-LoRA on Q/K/V for implicit temporal modelling

    The model can return intermediate features for distillation via
    `return_intermediate_layers`.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=None,
        use_tdlora=False,
        lora_rank=8,
        time_embed_dim=64,
        lora_type='fh_lora',
        hyper_hidden_dim=8,
        content_dim=16,
        content_dropout=0.3,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.depth = depth
        self.use_tdlora = use_tdlora
        self.lora_type = lora_type

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            StudentBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer,
                use_tdlora=use_tdlora, lora_rank=lora_rank,
                time_embed_dim=time_embed_dim, lora_type=lora_type,
                block_index=i, total_blocks=depth,
                hyper_hidden_dim=hyper_hidden_dim,
                content_dim=content_dim, content_dropout=content_dropout)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        if use_tdlora:
            self._reinit_lora_weights()

    def _reinit_lora_weights(self):
        """Re-apply LoRA-specific initializations after _init_weights."""
        import math as _math
        for blk in self.blocks:
            attn = blk.attn
            if not attn.use_tdlora:
                continue
            for lora in [attn.lora_q, attn.lora_k, attn.lora_v]:
                nn.init.kaiming_uniform_(lora.lora_A.weight, a=_math.sqrt(5))
                nn.init.zeros_(lora.lora_B.weight)
                if hasattr(lora, 'hyper_net'):
                    nn.init.zeros_(lora.hyper_net[-1].weight)
                    with torch.no_grad():
                        r = lora.rank
                        lora.hyper_net[-1].bias.copy_(
                            torch.eye(r).flatten())
                if hasattr(lora, 'gating') and hasattr(lora.gating, 'net_u'):
                    nn.init.zeros_(lora.gating.net_u[-1].weight)
                    nn.init.zeros_(lora.gating.net_u[-1].bias)
                    nn.init.zeros_(lora.gating.net_v[-1].weight)
                    nn.init.zeros_(lora.gating.net_v[-1].bias)
            if hasattr(attn, 'shared_trunk'):
                for head in attn.shared_trunk.heads.values():
                    nn.init.zeros_(head.weight)
                    with torch.no_grad():
                        r = attn.shared_trunk.rank
                        head.bias.copy_(torch.eye(r).flatten())

    @property
    def _needs_prev_cls(self):
        return self.lora_type in ('content', 'content_hyper')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def _interpolate_pos(self, x, H_patches, W_patches):
        """Interpolate positional embeddings if spatial resolution differs."""
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N:
            return self.pos_embed
        cls_pe = self.pos_embed[:, :1]
        patch_pe = self.pos_embed[:, 1:]
        dim = patch_pe.shape[-1]
        P = int(N ** 0.5)
        patch_pe = patch_pe.reshape(1, P, P, dim).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(
            patch_pe, size=(H_patches, W_patches), mode='bilinear',
            align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat([cls_pe, patch_pe], dim=1)

    def forward_features(self, x, return_intermediate=False, t_override=None):
        """
        Args:
            x: (B, C, T, H, W) video tensor.
            return_intermediate: if True, also return features from blocks
                                 at indices {2, 5, 8, 11} for distillation.
            t_override: (T,) optional frame indices override (e.g. reversed).
        Returns:
            cls_token: (B, D) global CLS features averaged over T frames.
            patch_tokens: (B, N, D) patch features averaged over T frames.
            intermediates: (optional) list of (B*T, N+1, D) at selected layers.
        """
        B_orig = x.shape[0]
        x, T, W_patches = self.patch_embed(x)
        BT = x.shape[0]
        H_patches = x.shape[1] // W_patches

        cls_tokens = self.cls_token.expand(BT, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self._interpolate_pos(x, H_patches, W_patches)
        x = x + pos_embed
        x = self.pos_drop(x)

        if t_override is not None:
            t = t_override
        elif self.use_tdlora:
            t = torch.arange(T, device=x.device, dtype=torch.float32)
        else:
            t = None

        intermediates = []
        n = self.depth
        distill_layers = {n*1//4-1, n*2//4-1, n*3//4-1, n-1}

        prev_cls = None
        for i, blk in enumerate(self.blocks):
            x = blk(x, t=t, prev_cls=prev_cls)
            if self._needs_prev_cls and t is not None:
                cls_per_frame = x[:, 0]
                prev_cls = rearrange(cls_per_frame, '(b t) d -> b t d',
                                     b=B_orig, t=T)
            if return_intermediate and i in distill_layers:
                intermediates.append(x)

        x = self.norm(x)

        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]

        cls_token = rearrange(cls_token, '(b t) d -> b t d',
                              b=B_orig, t=T)
        cls_token = cls_token.mean(dim=1)

        patch_tokens = rearrange(patch_tokens, '(b t) n d -> b t n d',
                                 b=B_orig, t=T)
        patch_tokens = patch_tokens.mean(dim=1)

        if return_intermediate:
            return cls_token, patch_tokens, intermediates
        return cls_token, patch_tokens

    def forward_per_frame(self, x):
        """Return per-frame CLS features *without* temporal averaging.

        Useful for downstream tasks that need per-frame representations
        (e.g. Cholec80 surgical phase recognition with LSTM).

        Args:
            x: (B, C, T, H, W) video tensor.
        Returns:
            per_frame_cls: (B, T, D) CLS token for each frame.
        """
        B_orig = x.shape[0]
        x, T, W_patches = self.patch_embed(x)
        BT = x.shape[0]
        H_patches = x.shape[1] // W_patches

        cls_tokens = self.cls_token.expand(BT, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self._interpolate_pos(x, H_patches, W_patches)
        x = x + pos_embed
        x = self.pos_drop(x)

        t = (torch.arange(T, device=x.device, dtype=torch.float32)
             if self.use_tdlora else None)

        prev_cls = None
        for blk in self.blocks:
            x = blk(x, t=t, prev_cls=prev_cls)
            if self._needs_prev_cls and t is not None:
                cls_per_frame = x[:, 0]
                prev_cls = rearrange(cls_per_frame, '(b t) d -> b t d',
                                     b=B_orig, t=T)

        x = self.norm(x)
        cls_token = x[:, 0]
        return rearrange(cls_token, '(b t) d -> b t d', b=B_orig, t=T)

    def collect_gating_values(self, B, T):
        """Collect and average gating values from all blocks for loss computation.

        Returns:
            gates: (B, T, rank) averaged gating values, or None if unavailable.
        """
        all_gates = []
        for blk in self.blocks:
            attn = blk.attn
            if not attn.use_tdlora:
                continue
            block_gates = []
            for lora in [attn.lora_q, attn.lora_k, attn.lora_v]:
                g = getattr(lora, '_last_gates', None)
                if g is not None:
                    if g.dim() == 2:
                        g = g.unsqueeze(0).expand(B, -1, -1)
                    block_gates.append(g)
            if block_gates:
                all_gates.append(torch.stack(block_gates).mean(0))
        if not all_gates:
            return None
        return torch.stack(all_gates).mean(0)

    def forward(self, x, use_head=False, return_intermediate=False):
        if return_intermediate:
            cls_token, patch_tokens, intermediates = self.forward_features(
                x, return_intermediate=True)
            if use_head:
                return self.head(cls_token), patch_tokens, intermediates
            return cls_token, patch_tokens, intermediates
        cls_token, patch_tokens = self.forward_features(x)
        if use_head:
            return self.head(cls_token)
        return cls_token, patch_tokens

    # ------------------------------------------------------------------
    # TD-LoRA management helpers
    # ------------------------------------------------------------------

    def enable_tdlora(self, rank=8, time_embed_dim=64,
                      lora_type='fh_lora',
                      hyper_hidden_dim=8, content_dim=16, content_dropout=0.3):
        """Insert LoRA adapters into every attention block (post-init)."""
        self.use_tdlora = True
        self.lora_type = lora_type
        for i, blk in enumerate(self.blocks):
            attn = blk.attn
            if attn.use_tdlora:
                continue
            attn.use_tdlora = True
            attn.lora_type = lora_type
            dim = attn.qkv.in_features
            dev = next(attn.parameters()).device
            attn._init_lora(dim, rank, time_embed_dim, lora_type,
                            block_index=i, total_blocks=self.depth,
                            hyper_hidden_dim=hyper_hidden_dim,
                            content_dim=content_dim,
                            content_dropout=content_dropout)
            attn.lora_q.to(dev)
            attn.lora_k.to(dev)
            attn.lora_v.to(dev)
            if hasattr(attn, 'shared_trunk'):
                attn.shared_trunk.to(dev)
                attn.lora_q.set_trunk(attn.shared_trunk)
                attn.lora_k.set_trunk(attn.shared_trunk)
                attn.lora_v.set_trunk(attn.shared_trunk)

    def freeze_backbone(self):
        """Freeze everything except TD-LoRA parameters."""
        lora_keys = ('lora_', 'gating', 'gate', 'hyper_net',
                     'time_embed', 'feat_proj', 'temporal_module',
                     'net_diag', 'net_u', 'net_v',
                     'shared_trunk', 'trunk', 'heads')
        for name, param in self.named_parameters():
            if any(k in name for k in lora_keys):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters for joint fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def _is_lora_param(self, name: str) -> bool:
        lora_keys = ('lora_', 'gating', 'gate', 'hyper_net',
                     'time_embed', 'feat_proj', 'temporal_module',
                     'net_diag', 'net_u', 'net_v',
                     'shared_trunk', 'trunk', 'heads')
        return any(k in name for k in lora_keys)

    def lora_parameters(self):
        """Yield only TD-LoRA parameters (for optimizer)."""
        for name, param in self.named_parameters():
            if self._is_lora_param(name):
                yield param

    def backbone_parameters(self):
        """Yield non-LoRA parameters."""
        for name, param in self.named_parameters():
            if not self._is_lora_param(name):
                yield param


# -------------------------------------------------------------------
# Factory functions
# -------------------------------------------------------------------

def student_vit_small(img_size=224, patch_size=16, use_tdlora=False,
                      lora_rank=8, lora_type='fh_lora', **kwargs):
    return StudentViT(
        img_size=img_size, patch_size=patch_size,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_tdlora=use_tdlora, lora_rank=lora_rank, lora_type=lora_type,
        **kwargs)


def student_vit_tiny(img_size=224, patch_size=16, use_tdlora=False,
                     lora_rank=8, lora_type='fh_lora', **kwargs):
    return StudentViT(
        img_size=img_size, patch_size=patch_size,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_tdlora=use_tdlora, lora_rank=lora_rank, lora_type=lora_type,
        **kwargs)


def student_vit_nano(img_size=224, patch_size=16, use_tdlora=False,
                     lora_rank=8, lora_type='fh_lora', **kwargs):
    return StudentViT(
        img_size=img_size, patch_size=patch_size,
        embed_dim=96, depth=10, num_heads=3, mlp_ratio=4.,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_tdlora=use_tdlora, lora_rank=lora_rank, lora_type=lora_type,
        **kwargs)


def student_vit_pico(img_size=224, patch_size=16, use_tdlora=False,
                     lora_rank=4, lora_type='fh_lora', **kwargs):
    """~1/100 of nano (~12K params). For rapid TD-LoRA prototyping only."""
    return StudentViT(
        img_size=img_size, patch_size=patch_size,
        embed_dim=10, depth=3, num_heads=1, mlp_ratio=2.,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_tdlora=use_tdlora, lora_rank=lora_rank, lora_type=lora_type,
        **kwargs)
