"""Match-level xP study: per-team destination rarity with hybrid league models."""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np
import pandas as pd

import passes_engine as pe

STUDY_MATCH_EVENT_ID = 15526003
XP_SMOOTHING = 1.0
XP_BLEND_ALPHA = 0.0
XP_REFERENCE_GLOBAL_WEIGHT = 0.65
XP_REFERENCE_LEAGUE_WEIGHT = 0.35
XP_PASS_MAX = 1.0

FIELD_X = pe.FIELD_X
FIELD_Y = pe.FIELD_Y
FIRST_THIRD_LINE_X = FIELD_X / 3.0
FIRST_THIRD_BLEND_END_X = 52.0
XP_FIRST_THIRD_MIN_MULT = 0.12

# Progress toward goal (StatsBomb x↑): penalize backward/lateral passes after rarity scoring.
# Calibrated on league completed passes — backward ratio bin mean xP ~0.28 vs forward ~0.38.
XP_PROGRESS_FLOOR_MULT = 0.30
XP_PROGRESS_LOGISTIC_K = 2.2

# Default grid (backward-compatible aliases).
XP_GRID_COLS = 12
XP_GRID_ROWS = 8
OD_GRID_COLS = 6
OD_GRID_ROWS = 4

XP_MODEL_HIER_DEST = "hierarchical_dest"
XP_MODEL_HIER_OD = "hierarchical_od"

XP_MODEL_LABELS: dict[str, str] = {
    XP_MODEL_HIER_DEST: "3 — Suavização hierárquica (destino 12×8)",
    XP_MODEL_HIER_OD: "4 — Suavização hierárquica (origem 12×8 → destino 12×8)",
}

XP_MODEL_COLUMNS: dict[str, str] = {
    XP_MODEL_HIER_DEST: "xp_hier_dest",
    XP_MODEL_HIER_OD: "xp_hier_od",
}

STUDY_MODELS: tuple[str, ...] = (XP_MODEL_HIER_DEST, XP_MODEL_HIER_OD)

THREAT_XP_THRESHOLDS: dict[str, float] = {
    XP_MODEL_HIER_DEST: 0.25,
    XP_MODEL_HIER_OD: 0.50,
}

DISTANCE_BAND_ORDER: tuple[str, ...] = ("short", "long")
XP_DISTANCE_BAND_MAX_SHORT_M = 30.0
DISTANCE_BAND_LABELS: dict[str, str] = {
    "short": "≤30 m",
    "long": ">30 m",
}


class GridConfig(NamedTuple):
    dest_cols: int
    dest_rows: int
    od_origin_cols: int
    od_origin_rows: int
    od_dest_cols: int
    od_dest_rows: int
    key: str
    label: str

    @property
    def od_cols(self) -> int:
        return self.od_origin_cols

    @property
    def od_rows(self) -> int:
        return self.od_origin_rows


STUDY_GRID = GridConfig(
    12, 8,
    12, 8,
    12, 8,
    "study_m3_m4",
    "Estudo — tudo 12×8",
)

GRID_PRESETS: dict[str, GridConfig] = {
    "default": GridConfig(12, 8, 6, 4, 12, 8, "default", "Destino 12×8 · OD origem 6×4"),
    "dest_8x6": GridConfig(8, 6, 6, 4, 8, 6, "dest_8x6", "Destino 8×6 · OD origem 6×4"),
    "all_8x6": GridConfig(8, 6, 8, 6, 8, 6, "all_8x6", "Tudo 8×6"),
    "all_12x8": GridConfig(12, 8, 12, 8, 12, 8, "all_12x8", "Tudo 12×8"),
    STUDY_GRID.key: STUDY_GRID,
}
DEFAULT_GRID_PRESET = STUDY_GRID.key


def normalize_grid_preset(preset: str | None) -> str:
    key = str(preset or DEFAULT_GRID_PRESET).strip().lower()
    return key if key in GRID_PRESETS else DEFAULT_GRID_PRESET


def get_grid_config(preset: str | None = None) -> GridConfig:
    return GRID_PRESETS[normalize_grid_preset(preset)]


def list_grid_presets() -> tuple[GridConfig, ...]:
    """Ordered grid presets for UI selectors."""
    return tuple(GRID_PRESETS[k] for k in GRID_PRESETS)


def _parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "successful"})


def _cell_indices(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cols: int,
    rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_bins = np.linspace(0.0, FIELD_X, cols + 1)
    y_bins = np.linspace(0.0, FIELD_Y, rows + 1)
    x_idx = np.clip(np.digitize(x, x_bins, right=True) - 1, 0, cols - 1)
    y_idx = np.clip(np.digitize(y, y_bins, right=True) - 1, 0, rows - 1)
    return x_idx, y_idx


def _progress_ratio_series(frame: pd.DataFrame) -> np.ndarray:
    """Signed progress toward opponent goal, normalized by pass length ∈ [-1, 1]."""
    dist = np.maximum(frame["pass_distance"].to_numpy(dtype=float), 0.5)
    dx = frame["progress_m"].to_numpy(dtype=float)
    return np.clip(dx / dist, -1.0, 1.0)


def progress_toward_goal_multiplier(progress_ratio: np.ndarray | pd.Series) -> np.ndarray:
    """Smooth monotone map: backward passes ↓ xP, forward passes ≈ 1.

    Logistic centered at lateral (ratio=0), floor at fully backward, cap at 1.0.
    Parameters calibrated against league marginal xP by progress-ratio bin.
    """
    ratio = np.asarray(progress_ratio, dtype=float)
    logistic = 1.0 / (1.0 + np.exp(-XP_PROGRESS_LOGISTIC_K * ratio))
    return XP_PROGRESS_FLOOR_MULT + (1.0 - XP_PROGRESS_FLOOR_MULT) * logistic


def _first_third_multiplier_vec(x_end: np.ndarray) -> np.ndarray:
    x = np.asarray(x_end, dtype=float)
    mult = np.ones(len(x), dtype=float)
    deep = x <= FIRST_THIRD_LINE_X
    mult[deep] = XP_FIRST_THIRD_MIN_MULT
    blend = (x > FIRST_THIRD_LINE_X) & (x < FIRST_THIRD_BLEND_END_X)
    if blend.any():
        t = (x[blend] - FIRST_THIRD_LINE_X) / (FIRST_THIRD_BLEND_END_X - FIRST_THIRD_LINE_X)
        mult[blend] = XP_FIRST_THIRD_MIN_MULT + (1.0 - XP_FIRST_THIRD_MIN_MULT) * t
    return mult


def _scale_grid_max_one(xp_grid: np.ndarray) -> np.ndarray:
    """Scale so the rarest cell equals XP_PASS_MAX (1.0); other cells are proportional."""
    mx = float(np.max(xp_grid))
    if mx <= 0:
        return xp_grid
    return xp_grid * (XP_PASS_MAX / mx)


def _enrich_match_passes(frame: pd.DataFrame) -> pd.DataFrame:
    sx, sy = pe._wyscout_to_sb(frame["start_x"], frame["start_y"])
    has_end = frame["end_x"].notna() & frame["end_y"].notna()
    ex = np.full(len(frame), np.nan)
    ey = np.full(len(frame), np.nan)
    if has_end.any():
        ex[has_end.to_numpy()], ey[has_end.to_numpy()] = pe._wyscout_to_sb(
            frame.loc[has_end, "end_x"], frame.loc[has_end, "end_y"]
        )

    out = pd.DataFrame({
        "player_id": frame["player_id"].astype(str),
        "player_name": frame["player_name"].astype(str),
        "position": frame["position"].astype(str).str.strip().str.upper() if "position" in frame.columns else "CM",
        "team": np.where(
            _parse_bool_series(frame["isHome"]),
            frame["home_team"].astype(str),
            frame["away_team"].astype(str),
        ),
        "is_success": _parse_bool_series(frame["outcome"]) if "outcome" in frame.columns else False,
        "action_type": frame["eventActionType"].astype(str).str.strip().str.lower(),
        "x_start": sx,
        "y_start": sy,
        "x_end": ex,
        "y_end": ey,
        "has_end": has_end.to_numpy(),
        "event_id": frame["event_id"].astype(int),
        "home_team": frame["home_team"].astype(str),
        "away_team": frame["away_team"].astype(str),
        "match_date": frame["match_date"].astype(str),
    })
    out["is_won"] = out["is_success"].astype(bool)
    out["pass_distance"] = np.where(
        out["has_end"],
        np.sqrt((out["x_end"] - out["x_start"]) ** 2 + (out["y_end"] - out["y_start"]) ** 2),
        0.0,
    )
    out["progress_m"] = np.where(
        out["has_end"],
        out["x_end"].to_numpy(dtype=float) - out["x_start"].to_numpy(dtype=float),
        0.0,
    )
    out["progress_ratio"] = _progress_ratio_series(out)
    out["is_restart"] = pe._restart_pass_mask(
        out["x_start"].to_numpy(dtype=float),
        out["y_start"].to_numpy(dtype=float),
        out["action_type"].astype(str).to_numpy(),
    )
    return out


def _count_destination_grid(
    passes: pd.DataFrame,
    grid: GridConfig,
) -> np.ndarray:
    count_grid = np.zeros((grid.dest_rows, grid.dest_cols), dtype=float)
    if passes is None or passes.empty:
        return count_grid

    completed = passes[passes["is_won"] & passes["has_end"]]
    if completed.empty:
        return count_grid

    x_idx, y_idx = _cell_indices(
        completed["x_end"].to_numpy(dtype=float),
        completed["y_end"].to_numpy(dtype=float),
        cols=grid.dest_cols,
        rows=grid.dest_rows,
    )
    for ix, iy in zip(x_idx, y_idx):
        count_grid[iy, ix] += 1.0
    return count_grid


def _count_od_tensor(passes: pd.DataFrame, grid: GridConfig) -> np.ndarray:
    o_rows, o_cols = grid.od_origin_rows, grid.od_origin_cols
    d_rows, d_cols = grid.od_dest_rows, grid.od_dest_cols
    tensor = np.zeros((o_rows, o_cols, d_rows, d_cols), dtype=float)
    if passes is None or passes.empty:
        return tensor

    completed = passes[passes["is_won"] & passes["has_end"]]
    if completed.empty:
        return tensor

    ox, oy = _cell_indices(
        completed["x_start"].to_numpy(dtype=float),
        completed["y_start"].to_numpy(dtype=float),
        cols=grid.od_origin_cols,
        rows=grid.od_origin_rows,
    )
    dx, dy = _cell_indices(
        completed["x_end"].to_numpy(dtype=float),
        completed["y_end"].to_numpy(dtype=float),
        cols=grid.od_dest_cols,
        rows=grid.od_dest_rows,
    )
    for oxi, oyi, dxi, dyi in zip(ox, oy, dx, dy):
        tensor[oyi, oxi, dyi, dxi] += 1.0
    return tensor


def _counts_to_xp_grid(
    count_grid: np.ndarray,
    *,
    smoothing: float = XP_SMOOTHING,
) -> np.ndarray:
    rows, cols = count_grid.shape
    xp_grid = np.ones((rows, cols), dtype=float)
    total = float(count_grid.sum())
    num_cells = rows * cols
    denom = total + smoothing * num_cells
    for iy in range(rows):
        for ix in range(cols):
            smoothed_count = count_grid[iy, ix] + smoothing
            freq = smoothed_count / denom
            xp_grid[iy, ix] = 1.0 / freq
    return _scale_grid_max_one(xp_grid)


def _blend_count_grid(
    match_count: np.ndarray,
    league_count_per_match: np.ndarray,
    *,
    alpha: float = XP_BLEND_ALPHA,
) -> np.ndarray:
    return alpha * match_count + (1.0 - alpha) * league_count_per_match


def _blend_od_tensor(
    match_tensor: np.ndarray,
    league_tensor_per_match: np.ndarray,
    *,
    alpha: float = XP_BLEND_ALPHA,
) -> np.ndarray:
    return alpha * match_tensor + (1.0 - alpha) * league_tensor_per_match


def _od_counts_to_lookup(tensor: np.ndarray) -> np.ndarray:
    o_rows, o_cols = tensor.shape[0], tensor.shape[1]
    d_rows, d_cols = tensor.shape[2], tensor.shape[3]
    total = float(tensor.sum())
    num_cells = tensor.size
    denom = total + XP_SMOOTHING * num_cells
    lookup = np.ones_like(tensor, dtype=float)
    for oyi in range(o_rows):
        for oxi in range(o_cols):
            for dyi in range(d_rows):
                for dxi in range(d_cols):
                    smoothed = tensor[oyi, oxi, dyi, dxi] + XP_SMOOTHING
                    lookup[oyi, oxi, dyi, dxi] = 1.0 / (smoothed / denom)
    return _scale_grid_max_one(lookup)


def build_destination_xp_grid(passes: pd.DataFrame, grid: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    count_grid = _count_destination_grid(passes, grid)
    return _counts_to_xp_grid(count_grid), count_grid


def build_team_xp_surfaces(
    passes: pd.DataFrame,
    grid: GridConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    xp_grids: dict[str, np.ndarray] = {}
    count_grids: dict[str, np.ndarray] = {}
    if passes is None or passes.empty:
        return xp_grids, count_grids

    for team, grp in passes.groupby("team", sort=False):
        team_name = str(team)
        xp_grids[team_name], count_grids[team_name] = build_destination_xp_grid(grp, grid)
    return xp_grids, count_grids


@functools.lru_cache(maxsize=1)
def _load_combined_league_pass_frame() -> pd.DataFrame:
    """Copa do Mundo (+ optional club leagues when CSVs are present) for the global xP reference pool."""
    frames: list[pd.DataFrame] = []
    world_cup = pe._load_season_pass_frame()
    if not world_cup.empty:
        wc = world_cup.copy()
        wc["league_source"] = "world_cup"
        frames.append(wc)
    serie_a = pe._load_br_pass_frame()
    if not serie_a.empty:
        sa = serie_a.copy()
        sa["league_source"] = "serie_a"
        frames.append(sa)
    premier_league = pe._load_pl_pass_frame()
    if not premier_league.empty:
        pl = premier_league.copy()
        pl["league_source"] = "premier_league"
        frames.append(pl)
    italia_seriea = pe._load_italia_seriea_pass_frame()
    if not italia_seriea.empty:
        it = italia_seriea.copy()
        it["league_source"] = "italia_seriea"
        frames.append(it)
    laliga = pe._load_laliga_pass_frame()
    if not laliga.empty:
        ll = laliga.copy()
        ll["league_source"] = "laliga"
        frames.append(ll)
    bundesliga = pe._load_bundesliga_pass_frame()
    if not bundesliga.empty:
        bl = bundesliga.copy()
        bl["league_source"] = "bundesliga"
        frames.append(bl)
    ligue1 = pe._load_ligue1_pass_frame()
    if not ligue1.empty:
        lg = ligue1.copy()
        lg["league_source"] = "ligue1"
        frames.append(lg)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@functools.lru_cache(maxsize=1)
def _league_completed_passes() -> pd.DataFrame:
    frame = _load_combined_league_pass_frame()
    if frame.empty:
        return pd.DataFrame()
    league_source = frame["league_source"].copy() if "league_source" in frame.columns else None
    passes = _enrich_match_passes(frame)
    if league_source is not None and len(league_source) == len(passes):
        passes = passes.copy()
        passes["league_source"] = league_source.to_numpy()
    passes = pe.filter_live_ball_passes(passes)
    if passes is None or passes.empty:
        return pd.DataFrame()
    completed = passes[passes["is_won"] & passes["has_end"]].copy()
    return completed


def _grid_config(
    dest_cols: int,
    dest_rows: int,
    od_origin_cols: int,
    od_origin_rows: int,
    od_dest_cols: int,
    od_dest_rows: int,
) -> GridConfig:
    return GridConfig(
        dest_cols, dest_rows,
        od_origin_cols, od_origin_rows,
        od_dest_cols, od_dest_rows,
        "", "",
    )


def _empty_reference_surfaces(
    dest_cols: int,
    dest_rows: int,
    od_origin_cols: int,
    od_origin_rows: int,
    od_dest_cols: int,
    od_dest_rows: int,
) -> dict[str, np.ndarray | float | int]:
    empty_dest = np.zeros((dest_rows, dest_cols), dtype=float)
    empty_od = np.zeros((od_origin_rows, od_origin_cols, od_dest_rows, od_dest_cols), dtype=float)
    return {
        "dest_count": empty_dest,
        "dest_count_per_match": empty_dest,
        "dest_xp": _counts_to_xp_grid(empty_dest),
        "od_count": empty_od,
        "od_count_per_match": empty_od,
        "num_matches": 1,
        "league_passes": 0,
    }


def _surfaces_from_completed(
    completed: pd.DataFrame,
    grid: GridConfig,
) -> dict[str, np.ndarray | float | int]:
    if completed.empty:
        return _empty_reference_surfaces(
            grid.dest_cols, grid.dest_rows,
            grid.od_origin_cols, grid.od_origin_rows,
            grid.od_dest_cols, grid.od_dest_rows,
        )
    dest_count = _count_destination_grid(completed, grid)
    od_count = _count_od_tensor(completed, grid)
    num_matches = max(int(completed["event_id"].nunique()), 1)
    dest_per_match = dest_count / num_matches
    od_per_match = od_count / num_matches
    return {
        "dest_count": dest_count,
        "dest_count_per_match": dest_per_match,
        "dest_xp": _counts_to_xp_grid(dest_count),
        "od_count": od_count,
        "od_count_per_match": od_per_match,
        "num_matches": num_matches,
        "league_passes": int(len(completed)),
    }


@functools.lru_cache(maxsize=16)
def _league_reference_surfaces(
    dest_cols: int,
    dest_rows: int,
    od_origin_cols: int,
    od_origin_rows: int,
    od_dest_cols: int,
    od_dest_rows: int,
) -> dict[str, np.ndarray | float | int]:
    grid = _grid_config(
        dest_cols, dest_rows,
        od_origin_cols, od_origin_rows,
        od_dest_cols, od_dest_rows,
    )
    completed = _league_completed_passes()
    surfaces = _surfaces_from_completed(completed, grid)
    if completed.empty:
        return surfaces

    if "league_source" in completed.columns:
        matches_by_league = completed.groupby("league_source")["event_id"].nunique()
        num_matches_world_cup = int(matches_by_league.get("world_cup", 0))
        num_matches_serie_a = int(matches_by_league.get("serie_a", 0))
        num_matches_premier_league = int(matches_by_league.get("premier_league", 0))
        num_matches_italia_seriea = int(matches_by_league.get("italia_seriea", 0))
        num_matches_laliga = int(matches_by_league.get("laliga", 0))
        num_matches_bundesliga = int(matches_by_league.get("bundesliga", 0))
        num_matches_ligue1 = int(matches_by_league.get("ligue1", 0))
        num_matches = max(
            num_matches_world_cup
            + num_matches_serie_a
            + num_matches_premier_league
            + num_matches_italia_seriea
            + num_matches_laliga
            + num_matches_bundesliga
            + num_matches_ligue1,
            1,
        )
    else:
        num_matches_world_cup = max(int(completed["event_id"].nunique()), 0)
        num_matches_serie_a = 0
        num_matches_premier_league = 0
        num_matches_italia_seriea = 0
        num_matches_laliga = 0
        num_matches_bundesliga = 0
        num_matches_ligue1 = 0
        num_matches = max(num_matches_world_cup, 1)

    surfaces.update({
        "num_matches": num_matches,
        "num_matches_world_cup": num_matches_world_cup,
        "num_matches_serie_a": num_matches_serie_a,
        "num_matches_premier_league": num_matches_premier_league,
        "num_matches_italia_seriea": num_matches_italia_seriea,
        "num_matches_laliga": num_matches_laliga,
        "num_matches_bundesliga": num_matches_bundesliga,
        "num_matches_ligue1": num_matches_ligue1,
        "reference_global_weight": XP_REFERENCE_GLOBAL_WEIGHT,
        "reference_league_weight": XP_REFERENCE_LEAGUE_WEIGHT,
    })
    return surfaces


@functools.lru_cache(maxsize=32)
def _league_source_reference_surfaces(
    league_source: str,
    dest_cols: int,
    dest_rows: int,
    od_origin_cols: int,
    od_origin_rows: int,
    od_dest_cols: int,
    od_dest_rows: int,
) -> dict[str, np.ndarray | float | int] | None:
    completed = _league_completed_passes()
    if completed.empty or "league_source" not in completed.columns:
        return None
    sub = completed[completed["league_source"].astype(str) == str(league_source)]
    if sub.empty:
        return None
    grid = _grid_config(
        dest_cols, dest_rows,
        od_origin_cols, od_origin_rows,
        od_dest_cols, od_dest_rows,
    )
    return _surfaces_from_completed(sub, grid)


def blend_global_league_reference(
    global_ref: dict[str, np.ndarray | float | int],
    league_ref: dict[str, np.ndarray | float | int] | None,
    *,
    global_weight: float = XP_REFERENCE_GLOBAL_WEIGHT,
    league_weight: float = XP_REFERENCE_LEAGUE_WEIGHT,
) -> dict[str, np.ndarray | float | int]:
    """Blend global and league-specific xP reference surfaces (65% / 35% by default)."""
    if not league_ref:
        return global_ref
    blended = dict(global_ref)
    blended["dest_count_per_match"] = (
        float(global_weight) * global_ref["dest_count_per_match"]
        + float(league_weight) * league_ref["dest_count_per_match"]
    )
    blended["od_count_per_match"] = (
        float(global_weight) * global_ref["od_count_per_match"]
        + float(league_weight) * league_ref["od_count_per_match"]
    )
    blended["reference_global_weight"] = float(global_weight)
    blended["reference_league_weight"] = float(league_weight)
    blended["reference_league_source"] = league_ref.get("reference_league_source")
    return blended


def reference_surfaces_for_league_source(
    league_source: str,
    dest_cols: int,
    dest_rows: int,
    od_origin_cols: int,
    od_origin_rows: int,
    od_dest_cols: int,
    od_dest_rows: int,
) -> dict[str, np.ndarray | float | int]:
    global_ref = _league_reference_surfaces(
        dest_cols, dest_rows,
        od_origin_cols, od_origin_rows,
        od_dest_cols, od_dest_rows,
    )
    league_ref = _league_source_reference_surfaces(
        league_source,
        dest_cols, dest_rows,
        od_origin_cols, od_origin_rows,
        od_dest_cols, od_dest_rows,
    )
    if league_ref is not None:
        league_ref = dict(league_ref)
        league_ref["reference_league_source"] = str(league_source)
    return blend_global_league_reference(global_ref, league_ref)


def _cap_pass_xp(values: np.ndarray, zone_mult: np.ndarray) -> np.ndarray:
    return np.minimum(values * zone_mult, XP_PASS_MAX)


def _assign_study_xp_models(
    passes: pd.DataFrame,
    *,
    grid: GridConfig,
    count_grids_by_team: dict[str, np.ndarray],
    league: dict[str, np.ndarray | float | int],
    alpha: float = XP_BLEND_ALPHA,
    team_season_od: dict[str, np.ndarray] | None = None,
    team_n_matches: dict[str, int] | None = None,
) -> pd.DataFrame:
    out = passes.copy()
    for col in XP_MODEL_COLUMNS.values():
        out[col] = 0.0
    out["xp_zone_mult"] = 0.0
    out["dest_ix"] = -1
    out["dest_iy"] = -1

    mask = out["is_won"] & out["has_end"]
    if not mask.any():
        return out

    sub = out.loc[mask]
    x_idx, y_idx = _cell_indices(
        sub["x_end"].to_numpy(dtype=float),
        sub["y_end"].to_numpy(dtype=float),
        cols=grid.dest_cols,
        rows=grid.dest_rows,
    )
    ox, oy = _cell_indices(
        sub["x_start"].to_numpy(dtype=float),
        sub["y_start"].to_numpy(dtype=float),
        cols=grid.od_origin_cols,
        rows=grid.od_origin_rows,
    )
    odx, ody = _cell_indices(
        sub["x_end"].to_numpy(dtype=float),
        sub["y_end"].to_numpy(dtype=float),
        cols=grid.od_dest_cols,
        rows=grid.od_dest_rows,
    )
    zone_mult = _first_third_multiplier_vec(sub["x_end"].to_numpy(dtype=float))

    league_dest_per_match = league["dest_count_per_match"]  # type: ignore[index]
    league_od_per_match = league["od_count_per_match"]  # type: ignore[index]

    hier_dest_xp_by_team: dict[str, np.ndarray] = {}
    hier_od_lookup_by_team: dict[str, np.ndarray] = {}
    for team, count_grid in count_grids_by_team.items():
        blended_dest = _blend_count_grid(count_grid, league_dest_per_match, alpha=alpha)
        hier_dest_xp_by_team[team] = _counts_to_xp_grid(blended_dest)
        team_key = str(team)
        season_od = team_season_od.get(team_key) if team_season_od is not None else None
        if season_od is not None:
            n_matches = max(int((team_n_matches or {}).get(team_key, 1)), 1)
            team_od = season_od / n_matches
        else:
            team_od = _count_od_tensor(passes[passes["team"].astype(str) == team_key], grid)
        blended_od = _blend_od_tensor(team_od, league_od_per_match, alpha=alpha)
        hier_od_lookup_by_team[team] = _od_counts_to_lookup(blended_od)

    hier_dest_vals = np.zeros(len(sub), dtype=float)
    hier_od_vals = np.zeros(len(sub), dtype=float)

    for i, (team, iy, ix, oyi, oxi, dyi, dxi) in enumerate(
        zip(sub["team"].astype(str), y_idx, x_idx, oy, ox, ody, odx)
    ):
        hier_grid = hier_dest_xp_by_team.get(team)
        hier_dest_vals[i] = float(hier_grid[iy, ix]) if hier_grid is not None else 0.0

        od_lookup = hier_od_lookup_by_team.get(team)
        hier_od_vals[i] = float(od_lookup[oyi, oxi, dyi, dxi]) if od_lookup is not None else 0.0

    out.loc[mask, "dest_ix"] = x_idx
    out.loc[mask, "dest_iy"] = y_idx
    out.loc[mask, "xp_zone_mult"] = zone_mult
    out.loc[mask, "xp_hier_dest"] = _cap_pass_xp(hier_dest_vals, zone_mult)
    out.loc[mask, "xp_hier_od"] = _cap_pass_xp(hier_od_vals, zone_mult)
    return out


def rank_players_by_xp(
    passes: pd.DataFrame,
    *,
    model: str = XP_MODEL_HIER_DEST,
) -> pd.DataFrame:
    col = XP_MODEL_COLUMNS.get(model, "xp_hier_dest")
    if passes is None or passes.empty or col not in passes.columns:
        return pd.DataFrame()

    scored = passes[passes["is_won"] & passes["has_end"]].copy()
    if scored.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for pid, grp in scored.groupby("player_id", sort=False):
        rows.append({
            "player_id": str(pid),
            "player_name": str(grp["player_name"].iloc[0]),
            "position": str(grp["position"].iloc[0]),
            "team": str(grp["team"].mode().iloc[0] if not grp["team"].mode().empty else grp["team"].iloc[0]),
            "passes_completed": int(len(grp)),
            "xp_total": float(grp[col].sum()),
            "xp_per_pass": float(grp[col].mean()),
            "xp_max_pass": float(grp[col].max()),
        })

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    ranking = ranking.sort_values(["xp_total", "xp_per_pass"], ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def build_model34_comparison_table(passes: pd.DataFrame) -> pd.DataFrame:
    """Compare xP totals and ranks for models 3 and 4 only."""
    if passes is None or passes.empty:
        return pd.DataFrame()

    scored = passes[passes["is_won"] & passes["has_end"]].copy()
    if scored.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for pid, grp in scored.groupby("player_id", sort=False):
        row = {
            "player_id": str(pid),
            "player_name": str(grp["player_name"].iloc[0]),
            "position": str(grp["position"].iloc[0]),
            "team": str(grp["team"].mode().iloc[0] if not grp["team"].mode().empty else grp["team"].iloc[0]),
            "passes_completed": int(len(grp)),
        }
        for model, col in XP_MODEL_COLUMNS.items():
            row[f"xp_{model}"] = float(grp[col].sum())
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    for model in XP_MODEL_COLUMNS:
        col = f"xp_{model}"
        table[f"rank_{model}"] = table[col].rank(ascending=False, method="min").astype(int)

    dest_col = f"xp_{XP_MODEL_HIER_DEST}"
    return table.sort_values(dest_col, ascending=False).reset_index(drop=True)


def top_xp_passes_for_player(
    passes: pd.DataFrame,
    player_id: str,
    *,
    n: int = 5,
    model: str = XP_MODEL_HIER_DEST,
) -> pd.DataFrame:
    col = XP_MODEL_COLUMNS.get(model, "xp_hier_dest")
    subset = passes[
        (passes["player_id"].astype(str) == str(player_id))
        & passes["is_won"]
        & passes["has_end"]
    ].copy()
    if subset.empty or col not in subset.columns:
        return subset
    subset = subset.assign(xp_value=subset[col])
    return subset.sort_values("xp_value", ascending=False).head(n).reset_index(drop=True)


def match_label(meta: dict) -> str:
    return f"{meta['home_team']} vs {meta['away_team']}"


def team_surface_for_player(
    xp_grids_by_team: dict[str, np.ndarray],
    count_grids_by_team: dict[str, np.ndarray],
    *,
    team: str,
    dest_rows: int,
    dest_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    empty_xp = np.ones((dest_rows, dest_cols), dtype=float) * 0.0
    empty_count = np.zeros((dest_rows, dest_cols), dtype=float)
    return (
        xp_grids_by_team.get(team, empty_xp),
        count_grids_by_team.get(team, empty_count),
    )


def grid_occupancy_stats(count_grids_by_team: dict[str, np.ndarray], grid: GridConfig) -> dict[str, float | int]:
    total_dest_cells = grid.dest_cols * grid.dest_rows
    total_od_cells = (
        grid.od_origin_cols * grid.od_origin_rows
        * grid.od_dest_cols * grid.od_dest_rows
    )
    dest_used = sum(int((cg > 0).sum()) for cg in count_grids_by_team.values())
    return {
        "dest_cells": total_dest_cells,
        "od_cells": total_od_cells,
        "dest_used": dest_used,
    }


def normalize_xp_model(model: str | None) -> str:
    key = str(model or XP_MODEL_HIER_DEST).strip().lower()
    return key if key in XP_MODEL_COLUMNS else XP_MODEL_HIER_DEST


def _distance_band_series(distances: pd.Series | np.ndarray) -> pd.Series:
    dist = np.asarray(distances, dtype=float)
    bands = np.where(dist <= XP_DISTANCE_BAND_MAX_SHORT_M, "short", "long")
    return pd.Series(bands, index=getattr(distances, "index", None))


def build_distance_threat_study(passes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize mean xP and threat-pass counts by distance band (models 3 and 4)."""
    col_m3 = XP_MODEL_COLUMNS[XP_MODEL_HIER_DEST]
    col_m4 = XP_MODEL_COLUMNS[XP_MODEL_HIER_OD]
    thr_m3 = THREAT_XP_THRESHOLDS[XP_MODEL_HIER_DEST]
    thr_m4 = THREAT_XP_THRESHOLDS[XP_MODEL_HIER_OD]

    empty_band = pd.DataFrame(columns=[
        "distance_band", "band_label", "passes",
        "mean_xp_m3", "mean_xp_m4",
        "threat_m3", "threat_m4", "pct_threat_m3", "pct_threat_m4",
    ])
    empty_player = pd.DataFrame(columns=[
        "player_id", "player_name", "team", "distance_band", "band_label", "passes",
        "mean_xp_m3", "mean_xp_m4", "threat_m3", "threat_m4",
    ])

    if passes is None or passes.empty:
        return empty_band, empty_player

    scored = passes[passes["is_won"] & passes["has_end"]].copy()
    if scored.empty:
        return empty_band, empty_player

    scored["distance_band"] = _distance_band_series(scored["pass_distance"])
    scored["band_label"] = scored["distance_band"].map(DISTANCE_BAND_LABELS)
    scored["is_threat_m3"] = scored[col_m3] > thr_m3
    scored["is_threat_m4"] = scored[col_m4] > thr_m4

    band_rows: list[dict] = []
    for band in DISTANCE_BAND_ORDER:
        grp = scored[scored["distance_band"] == band]
        if grp.empty:
            continue
        n = int(len(grp))
        band_rows.append({
            "distance_band": band,
            "band_label": DISTANCE_BAND_LABELS[band],
            "passes": n,
            "mean_xp_m3": float(grp[col_m3].mean()),
            "mean_xp_m4": float(grp[col_m4].mean()),
            "threat_m3": int(grp["is_threat_m3"].sum()),
            "threat_m4": int(grp["is_threat_m4"].sum()),
            "pct_threat_m3": float(grp["is_threat_m3"].mean() * 100.0),
            "pct_threat_m4": float(grp["is_threat_m4"].mean() * 100.0),
        })
    band_summary = pd.DataFrame(band_rows)

    player_rows: list[dict] = []
    for (pid, band), grp in scored.groupby(["player_id", "distance_band"], sort=False):
        player_rows.append({
            "player_id": str(pid),
            "player_name": str(grp["player_name"].iloc[0]),
            "team": str(grp["team"].mode().iloc[0] if not grp["team"].mode().empty else grp["team"].iloc[0]),
            "distance_band": str(band),
            "band_label": DISTANCE_BAND_LABELS.get(str(band), str(band)),
            "passes": int(len(grp)),
            "mean_xp_m3": float(grp[col_m3].mean()),
            "mean_xp_m4": float(grp[col_m4].mean()),
            "threat_m3": int(grp["is_threat_m3"].sum()),
            "threat_m4": int(grp["is_threat_m4"].sum()),
        })
    player_summary = pd.DataFrame(player_rows)
    if not player_summary.empty:
        player_summary = player_summary.sort_values(
            ["distance_band", "threat_m3", "threat_m4", "mean_xp_m3"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)

    return band_summary, player_summary


@functools.lru_cache(maxsize=4)
def load_study_match_bundle(
    event_id: int = STUDY_MATCH_EVENT_ID,
    grid_preset: str = DEFAULT_GRID_PRESET,
) -> dict:
    grid = get_grid_config(grid_preset)
    empty = {
        "passes": pd.DataFrame(),
        "xp_grids_by_team": {},
        "count_grids_by_team": {},
        "league": {},
        "rankings_by_model": {},
        "comparison": pd.DataFrame(),
        "distance_study": pd.DataFrame(),
        "distance_study_by_player": pd.DataFrame(),
        "meta": {},
        "grid": grid,
    }
    frame = pe._load_season_pass_frame()
    if frame.empty:
        return empty

    match_frame = frame[frame["event_id"].astype(int) == int(event_id)].copy()
    if match_frame.empty:
        return empty

    passes = _enrich_match_passes(match_frame)
    passes = pe.filter_live_ball_passes(passes)
    if passes is None:
        passes = pd.DataFrame()

    league = _league_reference_surfaces(
        grid.dest_cols, grid.dest_rows,
        grid.od_origin_cols, grid.od_origin_rows,
        grid.od_dest_cols, grid.od_dest_rows,
    )
    xp_grids_by_team, count_grids_by_team = build_team_xp_surfaces(passes, grid)
    if not passes.empty:
        passes = _assign_study_xp_models(
            passes,
            grid=grid,
            count_grids_by_team=count_grids_by_team,
            league=league,
        )

    first = match_frame.iloc[0]
    home_team = str(first["home_team"])
    away_team = str(first["away_team"])
    meta = {
        "event_id": int(event_id),
        "home_team": home_team,
        "away_team": away_team,
        "match_date": str(first["match_date"])[:10],
        "pass_events": int(len(match_frame)),
        "live_ball_passes": int(len(passes)),
        "completed_passes": int((passes["is_won"] & passes["has_end"]).sum()) if not passes.empty else 0,
        "players": int(passes["player_id"].nunique()) if not passes.empty else 0,
        "home_completed": int(
            (passes["is_won"] & passes["has_end"] & (passes["team"] == home_team)).sum()
        ) if not passes.empty else 0,
        "away_completed": int(
            (passes["is_won"] & passes["has_end"] & (passes["team"] == away_team)).sum()
        ) if not passes.empty else 0,
        "league_matches": int(league.get("num_matches", 0)),
        "league_matches_world_cup": int(league.get("num_matches_world_cup", 0)),
        "league_matches_serie_a": int(league.get("num_matches_serie_a", 0)),
        "league_matches_premier_league": int(league.get("num_matches_premier_league", 0)),
        "league_matches_italia_seriea": int(league.get("num_matches_italia_seriea", 0)),
        "league_matches_laliga": int(league.get("num_matches_laliga", 0)),
        "league_matches_bundesliga": int(league.get("num_matches_bundesliga", 0)),
        "league_matches_ligue1": int(league.get("num_matches_ligue1", 0)),
        "league_passes": int(league.get("league_passes", 0)),
        "blend_alpha": XP_BLEND_ALPHA,
        "reference_global_weight": XP_REFERENCE_GLOBAL_WEIGHT,
        "reference_league_weight": XP_REFERENCE_LEAGUE_WEIGHT,
        "xp_pass_max": XP_PASS_MAX,
        "grid_preset": grid.key,
        "grid_label": grid.label,
        "dest_cols": grid.dest_cols,
        "dest_rows": grid.dest_rows,
        "od_origin_cols": grid.od_origin_cols,
        "od_origin_rows": grid.od_origin_rows,
        "od_dest_cols": grid.od_dest_cols,
        "od_dest_rows": grid.od_dest_rows,
    }

    rankings_by_model = {
        model: rank_players_by_xp(passes, model=model)
        for model in STUDY_MODELS
    }
    comparison = build_model34_comparison_table(passes)
    distance_study, distance_study_by_player = build_distance_threat_study(passes)

    return {
        "passes": passes,
        "xp_grids_by_team": xp_grids_by_team,
        "count_grids_by_team": count_grids_by_team,
        "league": league,
        "rankings_by_model": rankings_by_model,
        "comparison": comparison,
        "distance_study": distance_study,
        "distance_study_by_player": distance_study_by_player,
        "meta": meta,
        "grid": grid,
    }


QUADRANT_X_SPLIT = FIELD_X / 2.0
QUADRANT_Y_SPLIT = FIELD_Y / 2.0
QUADRANT_LABELS: dict[str, str] = {
    "def_left": "Defensivo · esquerda",
    "def_right": "Defensivo · direita",
    "att_left": "Ofensivo · esquerda",
    "att_right": "Ofensivo · direita",
}


QUADRANT_ORDER: tuple[str, ...] = ("def_left", "def_right", "att_left", "att_right")


def quadrant_key_for_point(x: float, y: float) -> str:
    """Split the pitch into four quadrants: defensive/attacking half × left/right lane."""
    if float(x) < QUADRANT_X_SPLIT:
        return "def_left" if float(y) < QUADRANT_Y_SPLIT else "def_right"
    return "att_left" if float(y) < QUADRANT_Y_SPLIT else "att_right"


def quadrant_bounds(key: str) -> tuple[float, float, float, float]:
    """Rectangle (x0, y0, x1, y1) covering the quadrant."""
    x0, x1 = (0.0, QUADRANT_X_SPLIT) if str(key).startswith("def") else (QUADRANT_X_SPLIT, FIELD_X)
    y0, y1 = (0.0, QUADRANT_Y_SPLIT) if str(key).endswith("left") else (QUADRANT_Y_SPLIT, FIELD_Y)
    return x0, y0, x1, y1


def destination_quadrant_key(x_end: float, y_end: float) -> str:
    """Split the pitch into four quadrants by destination coordinates."""
    return quadrant_key_for_point(x_end, y_end)


def summarize_destination_quadrants(passes: pd.DataFrame) -> list[dict]:
    """Pass-volume share by destination quadrant for completed passes."""
    if passes is None or passes.empty:
        return []
    work = passes[passes["is_won"] & passes["has_end"]].dropna(subset=["x_end", "y_end"])
    if work.empty:
        return []

    counts: dict[str, int] = {key: 0 for key in QUADRANT_LABELS}
    for x_end, y_end in zip(
        work["x_end"].to_numpy(dtype=float),
        work["y_end"].to_numpy(dtype=float),
    ):
        counts[destination_quadrant_key(x_end, y_end)] += 1

    total = max(sum(counts.values()), 1)
    rows: list[dict] = []
    for key, label in QUADRANT_LABELS.items():
        count = int(counts.get(key, 0))
        rows.append({
            "quadrant_key": key,
            "quadrant": label,
            "passes": count,
            "share_pct": round(count / total * 100.0, 1),
        })
    return rows


def aggregate_pass_destination_grids(
    passes: pd.DataFrame,
    *,
    dest_cols: int = 8,
    dest_rows: int = 6,
    xp_col: str = "xp_m4",
) -> dict[str, np.ndarray | int | list[dict]]:
    """Destination frequency and mean xP grids for a pool of completed passes."""
    count_grid = np.zeros((dest_rows, dest_cols), dtype=float)
    xp_sum_grid = np.zeros((dest_rows, dest_cols), dtype=float)

    if passes is None or passes.empty:
        return {
            "count_grid": count_grid,
            "mean_xp_grid": np.zeros_like(count_grid),
            "total_passes": 0,
            "quadrant_stats": [],
        }

    work = passes[passes["is_won"] & passes["has_end"]].dropna(subset=["x_end", "y_end"]).copy()
    if work.empty:
        return {
            "count_grid": count_grid,
            "mean_xp_grid": np.zeros_like(count_grid),
            "total_passes": 0,
            "quadrant_stats": [],
        }

    x_idx, y_idx = _cell_indices(
        work["x_end"].to_numpy(dtype=float),
        work["y_end"].to_numpy(dtype=float),
        cols=dest_cols,
        rows=dest_rows,
    )
    has_xp = xp_col in work.columns
    xp_values = work[xp_col].to_numpy(dtype=float) if has_xp else np.zeros(len(work))

    for ix, iy, xp_value in zip(x_idx, y_idx, xp_values):
        count_grid[iy, ix] += 1.0
        if has_xp:
            xp_sum_grid[iy, ix] += float(xp_value)

    mean_xp_grid = np.zeros_like(count_grid)
    mask = count_grid > 0
    mean_xp_grid[mask] = xp_sum_grid[mask] / count_grid[mask]

    return {
        "count_grid": count_grid,
        "mean_xp_grid": mean_xp_grid,
        "total_passes": int(len(work)),
        "quadrant_stats": summarize_destination_quadrants(work),
    }


ZONE_X_LABELS: tuple[str, ...] = ("Defesa", "Meio", "Ataque")
ZONE_Y_LABELS: tuple[str, ...] = ("esquerda", "centro", "direita")


def _zone_label(x: float, y: float) -> str:
    """Human-readable pitch zone such as 'Meio-centro' or 'Ataque-direita'."""
    x_idx = min(int(float(x) / (FIELD_X / 3.0)), 2)
    y_idx = min(int(float(y) / (FIELD_Y / 3.0)), 2)
    return f"{ZONE_X_LABELS[x_idx]}-{ZONE_Y_LABELS[y_idx]}"


def zone_label_grid(cols: int, rows: int) -> list[list[str]]:
    """Zone name for every cell centre, indexed as grid[row][col]."""
    cell_w = FIELD_X / cols
    cell_h = FIELD_Y / rows
    return [
        [_zone_label((c + 0.5) * cell_w, (r + 0.5) * cell_h) for c in range(cols)]
        for r in range(rows)
    ]


CELL_MAP_MIN_ORIGIN_PASSES = 25


def cell_index_labels(cols: int, rows: int) -> list[str]:
    """Flat (row-major) human label per grid cell, e.g. 'Meio-centro · C6/L4'."""
    cell_w = FIELD_X / cols
    cell_h = FIELD_Y / rows
    return [
        f"{_zone_label((c + 0.5) * cell_w, (r + 0.5) * cell_h)} · C{c + 1}/L{r + 1}"
        for r in range(rows)
        for c in range(cols)
    ]


def build_cell_heatmap_analysis(
    passes: pd.DataFrame,
    *,
    cols: int = XP_GRID_COLS,
    rows: int = XP_GRID_ROWS,
    xp_col: str = "xp_m4",
    min_cell_passes: int = 20,
    min_origin_passes: int = CELL_MAP_MIN_ORIGIN_PASSES,
) -> dict:
    """Destination heatmaps (volume and mean xP) for passes leaving each 12x8 cell.

    Every origin cell with enough sample carries a full destination grid, flattened
    row-major so the browser can swap heatmaps on hover without recomputing anything.
    """
    n_cells = cols * rows
    empty: dict = {
        "cols": cols,
        "rows": rows,
        "field_x": FIELD_X,
        "field_y": FIELD_Y,
        "xp_max": XP_PASS_MAX,
        "min_cell_passes": min_cell_passes,
        "min_origin_passes": min_origin_passes,
        "total_passes": 0,
        "cell_labels": cell_index_labels(cols, rows),
        "quadrant_labels": dict(QUADRANT_LABELS),
        "origin_counts": [0] * n_cells,
        "overall": None,
        "origins": {},
        "xp_scale_max": XP_PASS_MAX,
        "volume_scale_max": 1.0,
    }
    if passes is None or passes.empty:
        return empty

    work = passes[passes["is_won"] & passes["has_end"]].dropna(
        subset=["x_start", "y_start", "x_end", "y_end"]
    )
    if work.empty:
        return empty

    ox_idx, oy_idx = _cell_indices(
        work["x_start"].to_numpy(dtype=float),
        work["y_start"].to_numpy(dtype=float),
        cols=cols,
        rows=rows,
    )
    dx_idx, dy_idx = _cell_indices(
        work["x_end"].to_numpy(dtype=float),
        work["y_end"].to_numpy(dtype=float),
        cols=cols,
        rows=rows,
    )
    xp_values = (
        work[xp_col].to_numpy(dtype=float)
        if xp_col in work.columns
        else np.zeros(len(work), dtype=float)
    )

    origin_flat = oy_idx * cols + ox_idx
    dest_flat = dy_idx * cols + dx_idx
    pair_flat = origin_flat * n_cells + dest_flat
    total_passes = int(len(work))

    pair_counts = np.bincount(pair_flat, minlength=n_cells * n_cells).reshape(n_cells, n_cells)
    pair_xp = np.bincount(
        pair_flat, weights=xp_values, minlength=n_cells * n_cells
    ).reshape(n_cells, n_cells)

    origin_counts = pair_counts.sum(axis=1)
    origin_xp_sum = pair_xp.sum(axis=1)

    def _serialize(counts_row: np.ndarray, xp_row: np.ndarray) -> tuple[list[int], list[float | None]]:
        filled = counts_row > 0
        means = np.zeros(n_cells, dtype=float)
        means[filled] = xp_row[filled] / counts_row[filled]
        return (
            [int(v) for v in counts_row],
            [round(float(means[i]), 3) if filled[i] else None for i in range(n_cells)],
        )

    overall_counts = pair_counts.sum(axis=0)
    overall_xp = pair_xp.sum(axis=0)
    counts_list, xp_list = _serialize(overall_counts, overall_xp)
    overall = {
        "index": None,
        "label": "Todo o campo",
        "passes": total_passes,
        "share_pct": 100.0,
        "mean_xp": round(float(xp_values.mean()), 4),
        "counts": counts_list,
        "xp": xp_list,
    }

    labels = cell_index_labels(cols, rows)
    origins: dict[str, dict] = {}
    volume_shares: list[float] = []
    for idx in range(n_cells):
        origin_total = int(origin_counts[idx])
        if origin_total < min_origin_passes:
            continue
        counts_list, xp_list = _serialize(pair_counts[idx], pair_xp[idx])
        origins[str(idx)] = {
            "index": idx,
            "row": idx // cols,
            "col": idx % cols,
            "label": labels[idx],
            "passes": origin_total,
            "share_pct": round(origin_total / max(total_passes, 1) * 100.0, 2),
            "mean_xp": round(float(origin_xp_sum[idx]) / origin_total, 4),
            "counts": counts_list,
            "xp": xp_list,
        }
        volume_shares.extend(
            float(v) / origin_total * 100.0 for v in pair_counts[idx] if v > 0
        )

    # Fixed colour scales keep hover states comparable; percentiles stop a single
    # tiny-sample cell from flattening every other map.
    eligible_pairs = pair_counts >= min_cell_passes
    if eligible_pairs.any():
        pair_means = np.zeros_like(pair_xp)
        pair_means[eligible_pairs] = pair_xp[eligible_pairs] / pair_counts[eligible_pairs]
        xp_scale_max = float(np.percentile(pair_means[eligible_pairs], 99))
    else:
        xp_scale_max = XP_PASS_MAX
    volume_scale_max = (
        float(np.percentile(np.asarray(volume_shares, dtype=float), 98))
        if volume_shares
        else 1.0
    )

    return {
        "cols": cols,
        "rows": rows,
        "field_x": FIELD_X,
        "field_y": FIELD_Y,
        "xp_max": XP_PASS_MAX,
        "min_cell_passes": min_cell_passes,
        "min_origin_passes": min_origin_passes,
        "total_passes": total_passes,
        "cell_labels": labels,
        "quadrant_labels": dict(QUADRANT_LABELS),
        "origin_counts": [int(v) for v in origin_counts],
        "overall": overall,
        "origins": origins,
        "xp_scale_max": round(max(xp_scale_max, 1e-3), 4),
        "volume_scale_max": round(max(volume_scale_max, 0.1), 3),
    }


def build_origin_destination_routes(
    passes: pd.DataFrame,
    *,
    cols: int = XP_GRID_COLS,
    rows: int = XP_GRID_ROWS,
    xp_col: str = "xp_m4",
    top_n: int = 5,
    min_route_passes: int = 3,
) -> dict[str, list[dict]]:
    """Top origin→destination cell pairs by volume and by total xP."""
    empty: dict[str, list[dict]] = {"common": [], "high_xp": []}
    if passes is None or passes.empty:
        return empty

    work = passes[passes["is_won"] & passes["has_end"]].dropna(
        subset=["x_start", "y_start", "x_end", "y_end"]
    )
    if work.empty:
        return empty

    n_cells = cols * rows
    ox_idx, oy_idx = _cell_indices(
        work["x_start"].to_numpy(dtype=float),
        work["y_start"].to_numpy(dtype=float),
        cols=cols,
        rows=rows,
    )
    dx_idx, dy_idx = _cell_indices(
        work["x_end"].to_numpy(dtype=float),
        work["y_end"].to_numpy(dtype=float),
        cols=cols,
        rows=rows,
    )
    xp_values = (
        work[xp_col].to_numpy(dtype=float)
        if xp_col in work.columns
        else np.zeros(len(work), dtype=float)
    )
    origin_flat = oy_idx * cols + ox_idx
    dest_flat = dy_idx * cols + dx_idx
    pair_flat = origin_flat * n_cells + dest_flat
    total_passes = max(len(work), 1)

    pair_counts = np.bincount(pair_flat, minlength=n_cells * n_cells)
    pair_xp = np.bincount(pair_flat, weights=xp_values, minlength=n_cells * n_cells)
    labels = cell_index_labels(cols, rows)

    routes: list[dict] = []
    for pair_idx in np.flatnonzero(pair_counts >= min_route_passes):
        count = int(pair_counts[pair_idx])
        origin_idx = int(pair_idx // n_cells)
        dest_idx = int(pair_idx % n_cells)
        total_xp = float(pair_xp[pair_idx])
        routes.append({
            "origin_index": origin_idx,
            "dest_index": dest_idx,
            "origin_label": labels[origin_idx],
            "dest_label": labels[dest_idx],
            "route_label": f"{labels[origin_idx]} → {labels[dest_idx]}",
            "count": count,
            "share_pct": round(count / total_passes * 100.0, 2),
            "total_xp": round(total_xp, 2),
            "mean_xp": round(total_xp / count, 3),
        })

    if not routes:
        return empty
    common = sorted(routes, key=lambda row: row["count"], reverse=True)[:top_n]
    high_xp = sorted(routes, key=lambda row: row["total_xp"], reverse=True)[:top_n]
    return {"common": common, "high_xp": high_xp}


def build_player_cell_heatmap_bundle(
    passes: pd.DataFrame,
    *,
    min_player_passes: int = 100,
    **kwargs,
) -> dict:
    """Aggregate map plus per-player destination grids for the interactive selector."""
    aggregate = build_cell_heatmap_analysis(passes, **kwargs)
    if passes is None or passes.empty:
        aggregate["players"] = []
        aggregate["default_player_id"] = None
        return aggregate

    work = passes[passes["is_won"] & passes["has_end"]].copy()
    if work.empty:
        aggregate["players"] = []
        aggregate["default_player_id"] = None
        return aggregate

    name_lookup: dict[str, str] = {}
    team_lookup: dict[str, str] = {}
    if "player_name" in work.columns:
        name_lookup = {
            str(pid): str(val)
            for pid, val in work.groupby("player_id", sort=False)["player_name"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
            .astype(str)
            .items()
        }
    if "team" in work.columns:
        team_lookup = {
            str(pid): str(val)
            for pid, val in work.groupby("player_id", sort=False)["team"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
            .astype(str)
            .items()
        }

    players: list[dict] = []
    for pid, grp in work.groupby("player_id", sort=False):
        pid_s = str(pid)
        if len(grp) < min_player_passes:
            continue
        player_analysis = build_cell_heatmap_analysis(grp, **kwargs)
        overall = player_analysis.get("overall")
        if not overall or int(overall.get("passes") or 0) < min_player_passes:
            continue
        players.append({
            "id": pid_s,
            "name": str(name_lookup.get(pid_s, pid_s)),
            "team": str(team_lookup.get(pid_s, "—")),
            "passes": int(overall.get("passes") or 0),
            "mean_xp": float(overall.get("mean_xp") or 0.0),
            "overall": overall,
            "origins": player_analysis.get("origins") or {},
            "routes": build_origin_destination_routes(grp, cols=aggregate.get("cols", XP_GRID_COLS), rows=aggregate.get("rows", XP_GRID_ROWS)),
        })

    players.sort(key=lambda row: (-int(row.get("passes") or 0), str(row.get("name") or "")))
    aggregate["players"] = players
    aggregate["default_player_id"] = players[0]["id"] if players else None
    return aggregate
