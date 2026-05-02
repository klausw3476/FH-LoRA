"""
Frame-conditioned LoRA adapters used in the paper.

Three variants are provided, all sharing the LoRA factorisation
``Delta h = (alpha / r) * B @ M(t) @ A @ x`` with optional frame-dependent
modulator ``M(t) in R^{r x r}``:

* ``StandardLoRA``  - frame-blind baseline (``M(t) == I``).
* ``PerProjLoRA``   - "Per-projection" baseline: each of the three
                      projections (q, k, v) owns its own hypernetwork
                      that produces ``M(t)``.
* ``FHLoRA``        - "FH-LoRA" (proposed): a single per-block trunk
                      (``FHTrunk``) on the frame index produces a hidden
                      state, and three projection-specific heads read out
                      ``M_q(t)``, ``M_k(t)``, ``M_v(t)``.

The CLI flag ``--lora_type`` in ``distill_temporal_lora.py`` selects
between them via the keys ``standard``, ``per_projection``, and
``fh_lora`` respectively.
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding of a scalar temporal index."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32)
            / half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        args = t.unsqueeze(-1) * self.freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ---------------------------------------------------------------------------
# LoRA variants
# ---------------------------------------------------------------------------


class StandardLoRA(nn.Module):
    """Frame-blind LoRA baseline: ``Delta h = (alpha / r) * B A x``."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 time_embed_dim: int = 64, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self._last_gates = None

    def forward(self, x: torch.Tensor, t: torch.Tensor = None,
                prev_cls: torch.Tensor = None) -> torch.Tensor:
        self._last_gates = None
        lo = self.lora_A(x)
        lo = self.lora_B(lo)
        return lo * self.scaling


class PerProjLoRA(nn.Module):
    """Per-projection hypernetwork baseline.

    A small MLP maps the sinusoidal embedding of the frame index to a full
    ``r x r`` matrix ``M(t)`` per projection. ``M(t)`` is initialised to
    the identity by zeroing the final linear weight and setting the bias
    to ``vec(I_r)``.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 time_embed_dim: int = 64, hyper_hidden_dim: int = 8,
                 alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        self.time_embed = SinusoidalPositionalEncoding(time_embed_dim)
        self.hyper_net = nn.Sequential(
            nn.Linear(time_embed_dim, hyper_hidden_dim),
            nn.GELU(),
            nn.Linear(hyper_hidden_dim, rank * rank),
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        nn.init.zeros_(self.hyper_net[-1].weight)
        with torch.no_grad():
            self.hyper_net[-1].bias.copy_(torch.eye(rank).flatten())
        self._last_gates = None

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                prev_cls: torch.Tensor = None) -> torch.Tensor:
        BT, N, _ = x.shape
        T = t.shape[0]
        B = BT // T

        t_emb = self.time_embed(t)
        M = self.hyper_net(t_emb).view(T, self.rank, self.rank)
        self._last_gates = torch.diagonal(M, dim1=-2, dim2=-1)

        lo = self.lora_A(x)
        lo = lo.view(B, T, N, self.rank)
        lo = torch.einsum('btnr,trs->btns', lo, M)
        lo = lo.reshape(BT, N, self.rank)
        lo = self.lora_B(lo)
        return lo * self.scaling


# ---------------------------------------------------------------------------
# FH-LoRA (proposed): shared trunk + projection-specific heads
# ---------------------------------------------------------------------------


class FHTrunk(nn.Module):
    """Shared per-block trunk on the frame index with q/k/v heads.

    A single ``Linear -> GELU`` trunk processes the sinusoidal embedding of
    ``t`` to a hidden state of width ``trunk_dim``; three lightweight
    linear heads then produce ``M_q(t)``, ``M_k(t)``, ``M_v(t)``. Heads are
    initialised so that all three matrices start at the identity.
    """

    def __init__(self, rank: int = 8, time_embed_dim: int = 64,
                 trunk_dim: int = 32):
        super().__init__()
        self.rank = rank
        self.pe = SinusoidalPositionalEncoding(dim=time_embed_dim)

        self.trunk = nn.Sequential(
            nn.Linear(time_embed_dim, trunk_dim),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict({
            proj: nn.Linear(trunk_dim, rank * rank)
            for proj in ('q', 'k', 'v')
        })

        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            with torch.no_grad():
                head.bias.copy_(torch.eye(rank).flatten())

    def forward(self, t: torch.Tensor, proj_name: str) -> torch.Tensor:
        pe = self.pe(t)
        h = self.trunk(pe)
        M = self.heads[proj_name](h).reshape(-1, self.rank, self.rank)
        return M


class FHLoRA(nn.Module):
    """FH-LoRA adapter for one projection.

    The shared trunk (``FHTrunk``) lives on the parent attention module
    (so it is instantiated once per block and shared across q/k/v); each
    ``FHLoRA`` instance keeps its own ``A`` and ``B`` and a ``proj_name``
    (one of ``q``, ``k``, ``v``) used to fetch the right head from the
    trunk at forward time.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 time_embed_dim: int = 64, alpha: float = 1.0,
                 proj_name: str = 'q'):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.proj_name = proj_name

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self._last_gates = None

    def set_trunk(self, trunk: FHTrunk) -> None:
        """Store a reference to the shared trunk without registering it
        as a sub-module (so it is not duplicated in the state dict)."""
        object.__setattr__(self, '_trunk_ref', trunk)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                prev_cls: torch.Tensor = None) -> torch.Tensor:
        BT, N, _ = x.shape
        T = t.shape[0]
        B = BT // T

        M = self._trunk_ref(t, self.proj_name)
        self._last_gates = torch.diagonal(M, dim1=-2, dim2=-1)

        lo = self.lora_A(x)
        lo = lo.view(B, T, N, self.rank)
        lo = torch.einsum('btnr,trs->btns', lo, M)
        lo = lo.reshape(BT, N, self.rank)
        lo = self.lora_B(lo)
        return lo * self.scaling
