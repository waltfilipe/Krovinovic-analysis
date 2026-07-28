"""Maps for the xP study tab."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mplsoccer import Pitch

from xp_study_engine import (
    FIELD_X,
    FIELD_Y,
    QUADRANT_LABELS,
    QUADRANT_X_SPLIT,
    QUADRANT_Y_SPLIT,
    XP_GRID_COLS,
    XP_GRID_ROWS,
    XP_PASS_MAX,
)

FIG_W, FIG_H = 8.4, 5.6
FIG_DPI = 220
ARROW_WIDTH = 0.9
ARROW_HEADWIDTH = 1.3
ARROW_HEADLENGTH = 1.3
CMAP_XP = plt.cm.plasma
# Low xP = cinza, alto xP = vermelho forte.
CMAP_XP_GRAY_RED = LinearSegmentedColormap.from_list(
    "xp_gray_red", ["#6b7280", "#9ca3af", "#f87171", "#ef4444", "#b91c1c"]
)
# Residual: abaixo do esperado (azul) → neutro → acima do esperado (verde).
CMAP_RESIDUAL = LinearSegmentedColormap.from_list(
    "residual_div", ["#3b82f6", "#cbd5e1", "#22c55e"]
)


def _base_pitch(*, figsize: tuple[float, float] = (FIG_W, FIG_H), dpi: int = FIG_DPI):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=figsize)
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(dpi)
    return fig, ax, pitch


def _delicate_arrows(pitch, ax, x1, y1, x2, y2, color, *, alpha: float, lw_scale: float = 1.0) -> None:
    pitch.arrows(
        x1, y1, x2, y2,
        color=color,
        width=ARROW_WIDTH * lw_scale,
        headwidth=ARROW_HEADWIDTH * lw_scale,
        headlength=ARROW_HEADLENGTH * lw_scale,
        ax=ax,
        zorder=4,
        alpha=alpha,
    )


def draw_xp_destination_surface(
    xp_grid: np.ndarray,
    count_grid: np.ndarray,
    *,
    title: str,
    dest_cols: int = XP_GRID_COLS,
    dest_rows: int = XP_GRID_ROWS,
):
    """Background heatmap of destination-cell xP weights for the match."""
    fig, ax, pitch = _base_pitch()
    rows, cols = xp_grid.shape
    dest_rows = rows
    dest_cols = cols
    x_bins = np.linspace(0.0, FIELD_X, dest_cols + 1)
    y_bins = np.linspace(0.0, FIELD_Y, dest_rows + 1)
    norm = Normalize(vmin=0.0, vmax=XP_PASS_MAX)
    for iy in range(dest_rows):
        for ix in range(dest_cols):
            if count_grid[iy, ix] <= 0:
                continue
            rect = Rectangle(
                (x_bins[ix], y_bins[iy]),
                x_bins[ix + 1] - x_bins[ix],
                y_bins[iy + 1] - y_bins[iy],
                facecolor=CMAP_XP(norm(float(xp_grid[iy, ix]))),
                edgecolor="#334155",
                linewidth=0.6,
                alpha=0.82,
                zorder=1,
            )
            ax.add_patch(rect)
    sm = ScalarMappable(norm=norm, cmap=CMAP_XP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("xP destino (raridade)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    ax.set_title(title, color="white", fontsize=10, pad=8)
    return fig


def draw_top_xp_passes_map(
    top_passes,
    *,
    player_name: str,
    match_label: str,
    xp_grid: np.ndarray | None = None,
    dest_cols: int = XP_GRID_COLS,
    dest_rows: int = XP_GRID_ROWS,
):
    """Top-N xP passes for one player, color-coded by xP value."""
    fig, ax, pitch = _base_pitch()

    if xp_grid is not None:
        dest_rows, dest_cols = xp_grid.shape
        x_bins = np.linspace(0.0, FIELD_X, dest_cols + 1)
        y_bins = np.linspace(0.0, FIELD_Y, dest_rows + 1)
        norm = Normalize(vmin=0.0, vmax=XP_PASS_MAX)
        for iy in range(dest_rows):
            for ix in range(dest_cols):
                rect = Rectangle(
                    (x_bins[ix], y_bins[iy]),
                    x_bins[ix + 1] - x_bins[ix],
                    y_bins[iy + 1] - y_bins[iy],
                    facecolor=CMAP_XP(norm(float(xp_grid[iy, ix]))),
                    edgecolor="none",
                    alpha=0.18,
                    zorder=1,
                )
                ax.add_patch(rect)

    if top_passes is None or top_passes.empty:
        ax.text(60, 40, "No passes with xP", ha="center", va="center", color="white", fontsize=10)
        ax.set_title(f"{player_name}\nTop passes xP · {match_label}", color="white", fontsize=10, pad=8)
        return fig

    values = top_passes["xp_value"].to_numpy(dtype=float)
    norm = Normalize(vmin=0.0, vmax=XP_PASS_MAX)

    for rank, row in enumerate(top_passes.itertuples(index=False), start=1):
        color = CMAP_XP(norm(float(row.xp_value)))
        lw_scale = 0.85 + 0.35 * (float(row.xp_value) / XP_PASS_MAX)
        _delicate_arrows(
            pitch, ax,
            row.x_start, row.y_start, row.x_end, row.y_end,
            color, alpha=0.92, lw_scale=lw_scale,
        )
        pitch.scatter(
            row.x_start, row.y_start,
            s=16, marker="o", color=color, edgecolors="white", linewidths=0.4, ax=ax, zorder=5,
        )
        pitch.scatter(
            row.x_end, row.y_end,
            s=20, marker="s", color=color, edgecolors="white", linewidths=0.4, ax=ax, zorder=6,
        )
        ax.text(
            row.x_end, row.y_end + 2.5,
            f"#{rank} {row.xp_value:.2f}",
            ha="center", va="bottom", color="white", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#475569", alpha=0.9),
            zorder=7,
        )

    sm = ScalarMappable(norm=norm, cmap=CMAP_XP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("xP do passe", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    legend_handles = [
        Line2D([0], [0], color=CMAP_XP(0.9), lw=2.0, label="Maior xP"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#94a3b8", markersize=5, linestyle="None", label="Origem"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#94a3b8", markersize=5, linestyle="None", label="Destino"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        fontsize=7,
    )
    for text in leg.get_texts():
        text.set_color("white")

    ax.set_title(
        f"{player_name}\nTop {len(top_passes)} passes xP · {match_label}",
        color="white", fontsize=10, pad=8,
    )
    return fig


def draw_top_residual_passes_map(
    top_passes,
    *,
    player_name: str,
    season_label: str = "temporada",
    residual_col: str = "xp_residual",
    highlight_index: int | None = None,
    show_labels: bool = True,
):
    """Top-N passes by xP residual, color-coded by surprise (actual − expected)."""
    fig, ax, pitch = _base_pitch()

    if top_passes is None or top_passes.empty:
        ax.text(
            60, 40, "No passes with residual",
            ha="center", va="center", color="white", fontsize=10,
        )
        ax.set_title(
            f"{player_name}\nTop Residual · {season_label}",
            color="white", fontsize=10, pad=8,
        )
        return fig

    work = top_passes.copy()
    if residual_col not in work.columns and {"xp_m4", "xp_expected"}.issubset(work.columns):
        work[residual_col] = work["xp_m4"].astype(float) - work["xp_expected"].astype(float)

    values = work[residual_col].to_numpy(dtype=float)
    abs_max = max(float(np.max(np.abs(values))), 0.02)
    norm = Normalize(vmin=-abs_max, vmax=abs_max)

    _draw_passes_on_pitch(
        ax,
        pitch,
        work,
        xp_col=residual_col,
        highlight_index=highlight_index,
        show_labels=show_labels,
        cmap=CMAP_RESIDUAL,
    )

    sm = ScalarMappable(norm=norm, cmap=CMAP_RESIDUAL)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Δ residual (xP real − esperado)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    legend_handles = [
        Line2D([0], [0], color=CMAP_RESIDUAL(0.95), lw=2.0, label="Acima do esperado"),
        Line2D([0], [0], color=CMAP_RESIDUAL(0.05), lw=2.0, label="Abaixo do esperado"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#94a3b8", markersize=5, linestyle="None", label="Origem"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#94a3b8", markersize=5, linestyle="None", label="Destino"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        fontsize=7,
    )
    for text in leg.get_texts():
        text.set_color("white")

    ax.set_title(
        f"{player_name}\nTop {len(work)} Residual · {season_label}",
        color="white", fontsize=10, pad=8,
    )
    return fig


def _draw_passes_on_pitch(
    ax,
    pitch: Pitch,
    work,
    *,
    xp_col: str | None = "xp_m4",
    highlight_index: int | None = None,
    show_labels: bool = True,
    cmap=CMAP_XP,
) -> None:
    """Draw mplsoccer arrows and origin/destination markers on an existing pitch axis."""
    color_by_xp = xp_col is not None and xp_col in work.columns
    if color_by_xp:
        values = work[xp_col].to_numpy(dtype=float)
        if float(np.min(values)) < 0.0:
            abs_max = max(float(np.max(np.abs(values))), 0.02)
            norm = Normalize(vmin=-abs_max, vmax=abs_max)
            diverging = True
        else:
            vmax = max(float(np.max(values)), 0.05)
            norm = Normalize(vmin=0.0, vmax=min(vmax, XP_PASS_MAX))
            diverging = False
    else:
        norm = None
        diverging = False

    rows = list(work.itertuples(index=False))
    draw_order = [i for i in range(len(rows)) if i != highlight_index]
    if highlight_index is not None and 0 <= highlight_index < len(rows):
        draw_order.append(highlight_index)

    for i in draw_order:
        row = rows[i]
        is_highlight = i == highlight_index
        if color_by_xp and not is_highlight:
            xp_value = float(getattr(row, xp_col))
            color = cmap(norm(xp_value))
            if diverging:
                abs_max = max(abs(float(norm.vmin)), abs(float(norm.vmax)), 0.02)
                lw_scale = 0.85 + 0.35 * min(abs(xp_value) / abs_max, 1.0)
            else:
                lw_scale = 0.85 + 0.35 * min(xp_value / XP_PASS_MAX, 1.0)
        elif is_highlight:
            color = "#fbbf24"
            lw_scale = 1.35
        else:
            color = "#60a5fa"
            lw_scale = 1.0

        alpha = 0.98 if is_highlight else 0.9
        origin_size = 34 if is_highlight else 16
        dest_size = 42 if is_highlight else 20
        z_mark = 10 if is_highlight else 5

        _delicate_arrows(
            pitch,
            ax,
            row.x_start,
            row.y_start,
            row.x_end,
            row.y_end,
            color,
            alpha=alpha,
            lw_scale=lw_scale,
        )
        pitch.scatter(
            row.x_start,
            row.y_start,
            s=origin_size,
            marker="o",
            color=color,
            edgecolors="#f8fafc",
            linewidths=1.2 if is_highlight else 0.5,
            ax=ax,
            zorder=z_mark,
        )
        pitch.scatter(
            row.x_end,
            row.y_end,
            s=dest_size,
            marker="s",
            color=color,
            edgecolors="#f8fafc",
            linewidths=1.2 if is_highlight else 0.5,
            ax=ax,
            zorder=z_mark + 1,
        )
        if show_labels:
            ax.text(
                row.x_start,
                row.y_start - 2.2,
                str(i + 1),
                ha="center",
                va="top",
                color="#f8fafc" if is_highlight else "#cbd5e1",
                fontsize=8.5 if is_highlight else 7.5,
                fontweight="bold" if is_highlight else "normal",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="#0f172a" if is_highlight else "#1e293b",
                    edgecolor="#fbbf24" if is_highlight else "#475569",
                    alpha=0.92,
                ),
                zorder=z_mark + 2,
            )


def draw_special_passes_season_map(
    passes,
    *,
    player_name: str,
    season_label: str = "temporada",
    category_label: str = "Special pass",
    xp_col: str | None = "xp_m4",
    threat_col: str | None = None,
    highlight_index: int | None = None,
    show_labels: bool = True,
    cmap=CMAP_XP,
):
    """Season map of passes for one special-pass category."""
    fig, ax, pitch = _base_pitch()

    if passes is None or passes.empty:
        ax.text(
            60, 40, "No passes for this filter",
            ha="center", va="center", color="white", fontsize=10,
        )
        ax.set_title(
            f"{player_name}\n{category_label} · {season_label}",
            color="white", fontsize=10, pad=8,
        )
        return fig

    work = passes.copy()
    color_by_xp = xp_col is not None and xp_col in work.columns
    _draw_passes_on_pitch(
        ax,
        pitch,
        work,
        xp_col=xp_col,
        highlight_index=highlight_index,
        show_labels=show_labels,
        cmap=cmap,
    )

    if color_by_xp:
        values = work[xp_col].to_numpy(dtype=float)
        vmax = max(float(np.max(values)), 0.05)
        norm = Normalize(vmin=0.0, vmax=min(vmax, XP_PASS_MAX))
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("xP do passe", color="white", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    legend_handles = [
        Line2D([0], [0], color=cmap(0.9) if color_by_xp else "#60a5fa", lw=2.0, label="Passe"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#94a3b8", markersize=5, linestyle="None", label="Origem"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#94a3b8", markersize=5, linestyle="None", label="Destino"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        fontsize=7,
    )
    for text in leg.get_texts():
        text.set_color("white")

    ax.set_title(
        f"{player_name}\n{len(work)} passes · {category_label} · {season_label}",
        color="white", fontsize=10, pad=8,
    )
    return fig


def draw_passes_destination_heatmap(
    passes,
    *,
    player_name: str,
    season_label: str = "temporada",
    category_label: str = "Special pass",
    cols: int = 12,
    rows: int = 8,
    cmap=CMAP_XP_GRAY_RED,
):
    """Heatmap of pass destination cells for the currently filtered passes."""
    fig, ax, pitch = _base_pitch()

    if passes is None or passes.empty or "x_end" not in passes.columns:
        ax.text(
            60, 40, "No passes for this filter",
            ha="center", va="center", color="white", fontsize=10,
        )
        ax.set_title(
            f"{player_name}\nDestino · {category_label} · {season_label}",
            color="white", fontsize=10, pad=8,
        )
        return fig

    work = passes.dropna(subset=["x_end", "y_end"]).copy()
    x_bins = np.linspace(0.0, FIELD_X, cols + 1)
    y_bins = np.linspace(0.0, FIELD_Y, rows + 1)
    grid = np.zeros((rows, cols), dtype=float)
    if not work.empty:
        x_idx = np.clip(
            np.digitize(work["x_end"].to_numpy(dtype=float), x_bins, right=True) - 1,
            0, cols - 1,
        )
        y_idx = np.clip(
            np.digitize(work["y_end"].to_numpy(dtype=float), y_bins, right=True) - 1,
            0, rows - 1,
        )
        for ix, iy in zip(x_idx, y_idx):
            grid[iy, ix] += 1.0

    vmax = max(float(grid.max()), 1.0)
    norm = Normalize(vmin=0.0, vmax=vmax)
    for iy in range(rows):
        for ix in range(cols):
            value = float(grid[iy, ix])
            if value <= 0:
                continue
            ax.add_patch(
                Rectangle(
                    (x_bins[ix], y_bins[iy]),
                    x_bins[ix + 1] - x_bins[ix],
                    y_bins[iy + 1] - y_bins[iy],
                    facecolor=cmap(norm(value)),
                    edgecolor=(1, 1, 1, 0.10),
                    linewidth=0.3,
                    alpha=0.9,
                    zorder=2,
                )
            )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Passes no destino", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    ax.set_title(
        f"{player_name}\nDestino dos passes · {category_label} · {season_label}",
        color="white", fontsize=10, pad=8,
    )
    return fig


CMAP_FREQ_GREEN = LinearSegmentedColormap.from_list(
    "freq_green", ["#1e293b", "#166534", "#22c55e", "#bbf7d0"]
)


def _draw_pitch_quadrants(ax) -> None:
    """Light quadrant guides: defensive/attacking halves and left/right lanes."""
    ax.axvline(x=QUADRANT_X_SPLIT, color="#cbd5e1", lw=1.4, alpha=0.55, zorder=3)
    ax.axhline(y=QUADRANT_Y_SPLIT, color="#cbd5e1", lw=1.4, alpha=0.55, zorder=3)
    ax.axvline(x=FIELD_X / 3.0, color="#475569", lw=0.8, alpha=0.28, zorder=3)
    ax.axvline(x=2.0 * FIELD_X / 3.0, color="#475569", lw=0.8, alpha=0.28, zorder=3)

    label_specs = (
        (FIELD_X * 0.25, FIELD_Y * 0.25, QUADRANT_LABELS["def_left"]),
        (FIELD_X * 0.25, FIELD_Y * 0.75, QUADRANT_LABELS["def_right"]),
        (FIELD_X * 0.75, FIELD_Y * 0.25, QUADRANT_LABELS["att_left"]),
        (FIELD_X * 0.75, FIELD_Y * 0.75, QUADRANT_LABELS["att_right"]),
    )
    for x_pos, y_pos, label in label_specs:
        ax.text(
            x_pos,
            y_pos,
            label,
            ha="center",
            va="center",
            color="#e2e8f0",
            fontsize=7.5,
            fontweight="700",
            alpha=0.82,
            zorder=4,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="#0f172a",
                edgecolor="#334155",
                alpha=0.72,
            ),
        )


def _draw_destination_grid_map(
    grid: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    cmap,
    vmin: float = 0.0,
    vmax: float | None = None,
    dest_cols: int = 8,
    dest_rows: int = 6,
):
    fig, ax, _pitch = _base_pitch()
    rows, cols = grid.shape
    dest_rows, dest_cols = rows, cols
    x_bins = np.linspace(0.0, FIELD_X, dest_cols + 1)
    y_bins = np.linspace(0.0, FIELD_Y, dest_rows + 1)
    values = grid.astype(float)
    positive = values[values > 0]
    if vmax is None:
        vmax = max(float(positive.max()), 1e-6) if positive.size else 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    for iy in range(dest_rows):
        for ix in range(dest_cols):
            value = float(values[iy, ix])
            if value <= 0:
                continue
            ax.add_patch(
                Rectangle(
                    (x_bins[ix], y_bins[iy]),
                    x_bins[ix + 1] - x_bins[ix],
                    y_bins[iy + 1] - y_bins[iy],
                    facecolor=cmap(norm(value)),
                    edgecolor=(1, 1, 1, 0.12),
                    linewidth=0.35,
                    alpha=0.92,
                    zorder=2,
                )
            )

    _draw_pitch_quadrants(ax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label, color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    ax.set_title(title, color="white", fontsize=10, pad=8)
    return fig


def draw_midfielder_common_passes_map(
    count_grid: np.ndarray,
    *,
    title: str,
    dest_cols: int = 8,
    dest_rows: int = 6,
):
    """Heatmap of where midfielder passes most often end (volume = common)."""
    return _draw_destination_grid_map(
        count_grid,
        title=title,
        cbar_label="Passes no destino",
        cmap=CMAP_FREQ_GREEN,
        dest_cols=dest_cols,
        dest_rows=dest_rows,
    )


def draw_midfielder_rare_passes_map(
    mean_xp_grid: np.ndarray,
    *,
    title: str,
    dest_cols: int = 8,
    dest_rows: int = 6,
):
    """Heatmap of mean xP by destination cell (higher = rarer passes)."""
    return _draw_destination_grid_map(
        mean_xp_grid,
        title=title,
        cbar_label="xP médio no destino",
        cmap=CMAP_XP_GRAY_RED,
        vmax=XP_PASS_MAX,
        dest_cols=dest_cols,
        dest_rows=dest_rows,
    )


def draw_xp_threat_passes_season_map(
    passes,
    *,
    player_name: str,
    season_label: str = "temporada",
    distance_label: str = "all distances",
    xp_col: str = "xp_m4",
):
    """Backward-compatible alias for threat-only season maps."""
    return draw_special_passes_season_map(
        passes,
        player_name=player_name,
        season_label=season_label,
        category_label=distance_label,
        xp_col=xp_col,
    )
