"""Nord-inspired plotting utilities with white background.

Copied from the original analysis/lib/plotting.py to keep export runnable.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd

NORD_PALETTE = {
    "polar1": "#2E3440",
    "polar2": "#3B4252",
    "polar3": "#434C5E",
    "polar4": "#4C566A",
    "frost1": "#8FBCBB",
    "frost2": "#88C0D0",
    "frost3": "#81A1C1",
    "frost4": "#5E81AC",
    "aurora1": "#BF616A",
    "aurora2": "#D08770",
    "aurora3": "#EBCB8B",
    "aurora4": "#A3BE8C",
    "aurora5": "#B48EAD",
}


@dataclass
class NordWhiteTheme:
    background: str = "#FFFFFF"
    axis: str = NORD_PALETTE["polar3"]
    title: str = NORD_PALETTE["polar4"]
    grid: str = "#D8DEE9"
    palette: tuple[str, ...] = (
        NORD_PALETTE["frost4"],
        NORD_PALETTE["aurora2"],
        NORD_PALETTE["aurora4"],
        NORD_PALETTE["aurora1"],
    )


def apply_theme(ax: plt.Axes, theme: NordWhiteTheme | None = None) -> None:
    theme = theme or NordWhiteTheme()
    ax.figure.set_facecolor(theme.background)
    ax.set_facecolor(theme.background)
    ax.tick_params(colors=theme.axis, labelsize=11)
    ax.xaxis.label.set_color(theme.axis)
    ax.yaxis.label.set_color(theme.axis)
    ax.title.set_color(theme.title)
    for spine in ax.spines.values():
        spine.set_color(theme.axis)
    ax.grid(color=theme.grid, alpha=0.4)


def lineplot(ax: plt.Axes, data: pd.DataFrame, x: str, y: str, hue: str | None = None, theme: NordWhiteTheme | None = None) -> plt.Axes:
    theme = theme or NordWhiteTheme()
    palette_cycle = list(theme.palette)
    if hue:
        for idx, (label, grp) in enumerate(data.groupby(hue)):
            color = palette_cycle[idx % len(palette_cycle)]
            ax.plot(grp[x], grp[y], marker="o", color=color, label=label)
    else:
        ax.plot(data[x], data[y], color=palette_cycle[0])
    apply_theme(ax, theme)
    if hue:
        ax.legend(frameon=False)
    return ax


def heatmap(ax: plt.Axes, matrix: pd.DataFrame, theme: NordWhiteTheme | None = None) -> plt.Axes:
    theme = theme or NordWhiteTheme()
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=matrix.min().min(), vmax=matrix.max().max())
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", color=theme.axis)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, color=theme.axis)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", color=theme.background if val > matrix.mean().mean() else theme.axis)
    apply_theme(ax, theme)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=theme.axis)
    for label in cbar.ax.get_yticklabels():
        label.set_color(theme.axis)
    return ax


__all__ = ["NordWhiteTheme", "apply_theme", "lineplot", "heatmap", "NORD_PALETTE"]

