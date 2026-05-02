#!/usr/bin/env python3
"""
Regenerate fig1 (distillation schematic) and fig2 (four adapter panels) for neurips_2026.tex
with a clean TC-LoRA–style aesthetic: rounded panels, context bar, hypernetwork grid, legend.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle
import numpy as np

# --- Palette (soft, paper-friendly) ---
C_BG = "#f1f5f9"
C_PANEL = "#ffffff"
C_PANEL_EDGE = "#cbd5e1"
C_BLUE = "#bfdbfe"
C_BLUE_D = "#3b82f6"
C_ORANGE = "#fed7aa"
C_ORANGE_D = "#ea580c"
C_GREEN = "#86efac"
C_GREEN_D = "#16a34a"
C_PURPLE = "#e9d5ff"
C_MUTED = "#64748b"
C_TEXT = "#0f172a"
C_YELLOW = "#fef08a"
C_PINK = "#fbcfe8"


def _rounded_box(ax, x, y, w, h, text, fc, ec=None, lw=1.2, fontsize=9, weight="normal", color=C_TEXT):
    """Draw a rounded rectangle with text guaranteed centered inside (tight bbox around text)."""
    ec = ec or C_PANEL_EDGE
    rs = float(min(0.05, w * 0.08, h * 0.25))
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={rs}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=2,
        clip_on=False,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color=color,
        zorder=3,
        linespacing=1.15,
        clip_on=False,
    )
    return p


def _badge(ax, x, y, text, fc=C_PURPLE, fontsize=6.5):
    bw = max(len(text) * 0.028 + 0.06, 0.14)
    bh = 0.055
    p = FancyBboxPatch(
        (x, y),
        bw,
        bh,
        boxstyle="round,pad=0.008,rounding_size=0.03",
        facecolor=fc,
        edgecolor="#a855f7",
        linewidth=0.8,
        zorder=4,
    )
    ax.add_patch(p)
    ax.text(
        x + bw / 2,
        y + bh / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight="bold",
        color="#5b21b6",
        zorder=5,
    )


def _arrow(ax, x1, y1, x2, y2, text=None):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.4,
        color=C_MUTED,
        zorder=1,
    )
    ax.add_patch(arr)
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, text, ha="center", fontsize=7, color=C_MUTED)


def _snowflake_icon(ax, x, y, s=0.035):
    """Simple 6-spoke snowflake marker for 'frozen'."""
    for k in range(6):
        ang = np.pi / 3 * k
        x0, y0 = x + 0.012 * np.cos(ang), y + 0.012 * np.sin(ang)
        x1, y1 = x + s * np.cos(ang), y + s * np.sin(ang)
        ax.plot([x0, x1], [y0, y1], color=C_BLUE_D, lw=1.4, zorder=6)


def _fire_icon(ax, x, y, s=0.028):
    verts = np.array(
        [
            [x, y + s],
            [x + s * 0.35, y + s * 0.2],
            [x + s * 0.2, y - s * 0.1],
            [x + s * 0.5, y - s * 0.35],
            [x + s * 0.15, y - s * 0.5],
            [x - s * 0.15, y - s * 0.35],
            [x - s * 0.35, y + s * 0.1],
        ]
    )
    ax.add_patch(Polygon(verts, closed=True, facecolor=C_ORANGE_D, edgecolor="#c2410c", lw=0.8, zorder=6))


def _hypernet_grid(ax, cx, cy, n=3, cell=0.045):
    """Small grid of nodes like the reference hypernetwork."""
    off = (n - 1) * cell / 2
    for i in range(n):
        for j in range(n):
            xi, yj = cx - off + i * cell, cy - off + j * cell
            ax.add_patch(Circle((xi, yj), cell * 0.32, facecolor=C_GREEN_D, edgecolor="#14532d", lw=0.6, zorder=5))


def _context_bar(ax, x, y, w, h, segments: list[tuple[str, str]]):
    """Horizontal segmented bar: (label, color)."""
    n = len(segments)
    sw = w / n
    for i, (lab, col) in enumerate(segments):
        bx = x + i * sw
        rect = FancyBboxPatch(
            (bx, y),
            sw * 0.97,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.02",
            facecolor=col,
            edgecolor="#94a3b8",
            linewidth=0.6,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            bx + sw / 2,
            y + h / 2,
            lab,
            ha="center",
            va="center",
            fontsize=5.8,
            weight="bold",
            color=C_TEXT,
            linespacing=1.0,
        )
    ax.text(x + w / 2, y + h + 0.028, "Context", ha="center", fontsize=7, weight="bold", color=C_MUTED)


def render_fig1(path: str) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 9,
        }
    )
    fig = plt.figure(figsize=(10.8, 5.2), facecolor="white")
    fig.patch.set_facecolor("white")

    fig.text(
        0.5,
        0.95,
        "Temporal behavior distilled from TimeSformer into a spatial student",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
        color="black",
    )

    ax_l = fig.add_axes([0.05, 0.10, 0.42, 0.78])
    ax_r = fig.add_axes([0.53, 0.10, 0.42, 0.78])
    for ax in (ax_l, ax_r):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.10, 1.0)
        ax.axis("off")

    def rect(ax, x, y, w, h, text, fs=8.5, weight="normal"):
        p = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.005,rounding_size=0.005",
            facecolor="#f7f7f7",
            edgecolor="black",
            linewidth=1.1,
            clip_on=False,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, weight=weight, color="black")
        return p

    def plus(ax, x, y, r=0.025):
        c = Circle((x, y), r, facecolor="white", edgecolor="black", linewidth=1.0)
        c.set_clip_on(False)
        ax.add_patch(c)
        ax.text(x, y - 0.001, "+", ha="center", va="center", fontsize=12, color="black")

    def v_arrow(ax, x0, y0, x1, y1):
        arr = FancyArrowPatch((x0, y0), (x1, y1),
                              arrowstyle="-|>", mutation_scale=8,
                              linewidth=1.0, color="black")
        arr.set_clip_on(False)
        ax.add_patch(arr)

    def residual(ax, x_right, y_top, y_bottom, bulge=0.12):
        verts = [(x_right, y_top), (x_right + bulge, (y_top + y_bottom) / 2), (x_right, y_bottom)]
        path = mpl.path.Path(verts, [mpl.path.Path.MOVETO, mpl.path.Path.CURVE3, mpl.path.Path.CURVE3])
        patch = mpl.patches.PathPatch(path, facecolor="none", edgecolor="black", linewidth=1.2)
        patch.set_clip_on(False)
        ax.add_patch(patch)

    # Left panel: teacher block in the style of the reference image.
    ax_l.text(0.50, 0.985, "(a) TimeSformer teacher block", fontsize=9.6, weight="bold", va="top", ha="center")
    rect(ax_l, 0.36, 0.82, 0.26, 0.07, r"$z^{(\ell-1)}$", fs=11)
    v_arrow(ax_l, 0.49, 0.82, 0.49, 0.79)
    rect(ax_l, 0.36, 0.68, 0.26, 0.08, "Spatial Att.", fs=9)
    v_arrow(ax_l, 0.49, 0.68, 0.49, 0.615)
    plus(ax_l, 0.49, 0.61)
    residual(ax_l, 0.62, 0.875, 0.67)
    v_arrow(ax_l, 0.49, 0.585, 0.49, 0.55)
    rect(ax_l, 0.32, 0.46, 0.34, 0.09, "Time Att.", fs=9)
    v_arrow(ax_l, 0.49, 0.46, 0.49, 0.395)
    plus(ax_l, 0.49, 0.39)
    residual(ax_l, 0.66, 0.67, 0.42)
    v_arrow(ax_l, 0.49, 0.365, 0.49, 0.30)
    rect(ax_l, 0.39, 0.21, 0.20, 0.08, "MLP", fs=9)
    v_arrow(ax_l, 0.49, 0.21, 0.49, 0.145)
    plus(ax_l, 0.49, 0.14)
    residual(ax_l, 0.59, 0.42, 0.14)
    v_arrow(ax_l, 0.49, 0.115, 0.49, 0.10)
    rect(ax_l, 0.38, 0.03, 0.22, 0.07, r"$z^{(\ell)}$", fs=11)
    ax_l.text(0.49, -0.055, "Divided space-time attention (teacher)", ha="center", fontsize=9)

    # Right panel: student block, keeping the same visual grammar.
    ax_r.text(0.50, 0.985, "(b) Spatial student + FH-LoRA", fontsize=9.4, weight="bold", va="top", ha="center")
    rect(ax_r, 0.36, 0.82, 0.26, 0.07, r"$z^{(\ell-1)}$", fs=11)
    v_arrow(ax_r, 0.49, 0.82, 0.49, 0.79)
    rect(ax_r, 0.36, 0.68, 0.26, 0.08, "Spatial Att.", fs=9)
    v_arrow(ax_r, 0.49, 0.68, 0.49, 0.615)
    plus(ax_r, 0.49, 0.61)
    residual(ax_r, 0.62, 0.875, 0.67)

    # Side hypernetwork / LoRA injection block.
    rect(ax_r, 0.70, 0.74, 0.25, 0.06, r"$t,\ \mathrm{PE}(t)$", fs=8)
    v_arrow(ax_r, 0.825, 0.74, 0.825, 0.67)
    rect(ax_r, 0.68, 0.59, 0.29, 0.08, "Shared trunk", fs=8)
    v_arrow(ax_r, 0.825, 0.59, 0.825, 0.50)
    rect(ax_r, 0.66, 0.43, 0.31, 0.11, r"heads $\rightarrow$ $M_q, M_k, M_v$", fs=7.6)
    arr = FancyArrowPatch((0.66, 0.485), (0.60, 0.485),
                          arrowstyle="-|>", mutation_scale=8,
                          linewidth=1.0, color="black")
    arr.set_clip_on(False)
    ax_r.add_patch(arr)
    ax_r.text(0.605, 0.515, r"$\Delta h = B\,M^{(\mathrm{proj})}(t)\,A\,x$", fontsize=6.8, ha="left")

    v_arrow(ax_r, 0.49, 0.585, 0.49, 0.55)
    rect(ax_r, 0.39, 0.46, 0.20, 0.09, "MLP", fs=9)
    v_arrow(ax_r, 0.49, 0.46, 0.49, 0.395)
    plus(ax_r, 0.49, 0.39)
    residual(ax_r, 0.59, 0.67, 0.42)
    v_arrow(ax_r, 0.49, 0.365, 0.49, 0.30)
    rect(ax_r, 0.38, 0.21, 0.22, 0.08, r"LoRA on Q/K/V", fs=8)
    v_arrow(ax_r, 0.49, 0.21, 0.49, 0.145)
    plus(ax_r, 0.49, 0.14)
    residual(ax_r, 0.60, 0.42, 0.14)
    v_arrow(ax_r, 0.49, 0.115, 0.49, 0.10)
    rect(ax_r, 0.38, 0.03, 0.22, 0.07, r"$z^{(\ell)}$", fs=11)
    ax_r.text(0.49, -0.055, "Spatial-only trunk with temporal adapters", ha="center", fontsize=9)

    fig.text(
        0.5,
        0.03,
        "Student removes temporal MHSA from the trunk; temporal behavior is reintroduced through FH-LoRA on Q/K/V.",
        ha="center",
        fontsize=7.5,
        color="#444444",
    )

    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)


def render_fig2(path: str) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 9,
        }
    )
    fig = plt.figure(figsize=(11.2, 4.4), facecolor="white")
    fig.patch.set_facecolor("white")
    fig.text(
        0.5,
        0.94,
        "Adapter structures compared in this work",
        ha="center",
        fontsize=11.5,
        weight="bold",
        color="black",
    )
    fig.text(
        0.5,
        0.86,
        "All variants use the same ViT-Small backbone; only the adapter pathway differs.",
        ha="center",
        fontsize=8.2,
        color="#555555",
    )

    cards = [
        (
            "Standard LoRA",
            "Time-invariant adapter",
            "Input\n|\nLoRA A/B\n|\nOutput",
            "Single low-rank path shared across all frames.",
            False,
        ),
        (
            "Per-projection",
            "Independent hypernetworks",
            "frame index  $t$\n|\nPE(t)\n|\nthree hypernets\n|\nQ / K / V heads",
            "Separate temporal networks generate projection-wise modulation.",
            False,
        ),
        (
            "FH-LoRA",
            "Shared trunk + heads",
            "frame index  $t$\n|\nPE(t)\n|\nshared trunk\n|\nQ / K / V heads",
            "One shared temporal state feeds projection-specific heads.",
            True,
        ),
    ]

    n = len(cards)
    margin = 0.06
    gap = 0.04
    cw = (1 - 2 * margin - (n - 1) * gap) / n
    y0 = 0.18
    h = 0.58

    for i, (label, head, formula, cap, highlight) in enumerate(cards):
        x0 = margin + i * (cw + gap)
        ax = fig.add_axes([x0, y0, cw, h])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ec = "black"
        lw = 1.5 if highlight else 1.1
        bg = "white"
        outer = FancyBboxPatch(
            (0.02, 0.02),
            0.96,
            0.96,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor=bg,
            edgecolor=ec,
            linewidth=lw,
            zorder=0,
        )
        ax.add_patch(outer)
        ax.text(0.5, 0.90, label, ha="center", fontsize=9.2, weight="bold", color="black")
        ax.text(0.5, 0.76, head, ha="center", fontsize=8.5, weight="normal", color="black")
        inner = FancyBboxPatch(
            (0.18, 0.26),
            0.64,
            0.38,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#f7f7f7",
            edgecolor="black",
            linewidth=0.9,
        )
        ax.add_patch(inner)
        fsize = 8.0
        ax.text(
            0.5,
            0.45,
            formula,
            ha="center",
            va="center",
            fontsize=fsize,
            color="black",
            linespacing=1.25,
        )
        ax.text(
            0.5,
            0.11,
            cap,
            ha="center",
            va="center",
            fontsize=7.0,
            color="#444444",
            linespacing=1.2,
        )

    fig.text(
        0.5,
        0.08,
        "FH-LoRA differs from the per-projection variant only in replacing three independent temporal networks with one shared trunk and projection-specific heads.",
        ha="center",
        fontsize=7.4,
        color="#555555",
    )

    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)


def main():
    import os

    root = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    render_fig1(os.path.join(root, "fig1_st_hyper_schematic.png"))
    render_fig2(os.path.join(root, "fig2_modulation_four_panel.png"))
    print("Wrote:", root)


if __name__ == "__main__":
    main()
