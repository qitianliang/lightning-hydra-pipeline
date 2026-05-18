# src/utils/visualization.py
"""论文风格参数敏感性绘图模块 — 纯函数, 无 workflow 依赖。"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# ── 通用论文风格 ────────────────────────────────────────────────────────────
_PAPER_RC = {
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _apply_paper_style():
    plt.rcParams.update(_PAPER_RC)


# ── 1D 敏感性绘图 ──────────────────────────────────────────────────────────
def plot_sensitivity_1d(
    param_name: str,
    param_values: list,
    metric_means: list,
    metric_stds: list,
    xlabel: str,
    ylabel: str,
    output_dir: Path,
    figure_width: float = 3.5,
    figure_dpi: int = 300,
    title: str = "",
) -> Tuple[Path, Path]:
    """1D 参数敏感性折线图 (带误差棒)。

    Args:
        title: 图表标题 (通常为 group_name), 空则不显示

    Returns:
        (png_path, pdf_path)
    """
    _apply_paper_style()

    fig, ax = plt.subplots(figsize=(figure_width, figure_width * 0.65))

    x = np.array(param_values, dtype=float)
    means = np.array(metric_means, dtype=float)
    stds = np.array(metric_stds, dtype=float)

    ax.errorbar(x, means, yerr=stds, marker="o", capsize=3,
                linewidth=1.2, markersize=4, color="#2c7bb6")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        fig.suptitle(title, fontsize=8, y=0.98)
    fig.tight_layout(pad=0.3)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = param_name.replace(".", "_")
    png_path = output_dir / f"sensitivity_1d_{safe_name}.png"
    pdf_path = output_dir / f"sensitivity_1d_{safe_name}.pdf"

    fig.savefig(png_path, dpi=figure_dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)

    log.info(f"📊 1D sensitivity plot saved: {png_path.resolve()}, {pdf_path.resolve()}")
    return png_path.resolve(), pdf_path.resolve()


# ── 2D 敏感性绘图 ──────────────────────────────────────────────────────────
def plot_sensitivity_2d(
    param_x_name: str,
    param_y_name: str,
    param_x_values: list,
    param_y_values: list,
    metric_matrix: np.ndarray,
    xlabel: str,
    ylabel: str,
    output_dir: Path,
    figure_width: float = 3.5,
    figure_dpi: int = 300,
    title: str = "",
) -> Tuple[Path, Path]:
    """2D 参数敏感性热力图。

    Args:
        metric_matrix: shape (len_y, len_x), 行=y, 列=x
        title: 图表标题 (通常为 group_name), 空则不显示

    Returns:
        (png_path, pdf_path)
    """
    _apply_paper_style()

    fig, ax = plt.subplots(figsize=(figure_width, figure_width * 0.8))

    sns.heatmap(
        metric_matrix,
        xticklabels=[str(v) for v in param_x_values],
        yticklabels=[str(v) for v in param_y_values],
        annot=True,
        fmt=".4f",
        cmap="viridis",
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        fig.suptitle(title, fontsize=8, y=1.02)
    fig.tight_layout(pad=0.3)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_x = param_x_name.replace(".", "_")
    safe_y = param_y_name.replace(".", "_")
    png_path = output_dir / f"sensitivity_2d_{safe_x}_{safe_y}.png"
    pdf_path = output_dir / f"sensitivity_2d_{safe_x}_{safe_y}.pdf"

    fig.savefig(png_path, dpi=figure_dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)

    log.info(f"📊 2D sensitivity plot saved: {png_path.resolve()}, {pdf_path.resolve()}")
    return png_path.resolve(), pdf_path.resolve()
