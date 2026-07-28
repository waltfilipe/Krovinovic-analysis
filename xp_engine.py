"""Season-wide xP Model 4 scoring, expected-xP regression and threat classification."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

import passes_engine as pe
import xp_study_engine as xse

XP_DATA_CACHE_VERSION = 58
XP_POSITION_RANK_METRICS: tuple[str, ...] = (
    "xp_m4_total",
    "xp_m4_per_pass",
    "xp_m4_threat_passes_p90",
    "xp_m4_threat_rate",
    "xp_m4_total_short",
    "xp_m4_threat_short_p90",
    "xp_m4_total_long",
    "xp_m4_threat_long_p90",
)
XP_MODEL_VERSION = "m4_od_12x8_b4_a2_pr0_global_ita_laliga_ligue1_blend65_35_dist30_access_ridge_v8"
THREAT_QUANTILE = 0.10
THREAT_XP_QUANTILE = 0.75
THREAT_PROGRESS_MIN = 0.0
# Composite impact-pass rule (replaces residual∩xP dual threshold on is_threat_m4).
IMPACT_PASS_RULE_VERSION = "composite_v2_p925_prog65"
IMPACT_SCORE_W_XP = 0.45
IMPACT_SCORE_W_RESIDUAL = 0.35
IMPACT_SCORE_W_PROGRESS = 0.20
IMPACT_SCORE_PERCENTILE = 0.925
IMPACT_PROGRESS_PERCENTILE = 0.65
IMPACT_PASS_RULE_LABEL = (
    "composite score (45% destination xP + 35% residual + 20% progress) ≥ P92.5 per distance band "
    "and progress_ratio ≥ P65 per band"
)
XP_COL = "xp_m4"
XP_SPATIAL_COL = "xp_hier_od"
XP_ACCESSIBILITY_MULT_COL = "xp_accessibility_mult"
XP_EXPECTED_COL = "xp_expected"
XP_RESIDUAL_COL = "xp_residual"
THREAT_COL = "is_threat_m4"

# Accessibility model B: local connectivity is easier deep, harder in attack.
XP_ACCESS_LOCALITY_SCALE = 1.35
XP_ACCESS_PRESSURE_CENTER_X = 52.0
XP_ACCESS_PRESSURE_SCALE = 12.0
XP_ACCESS_PRESSURE_WEIGHT = 0.65
XP_ACCESS_BETA_DEEP_SHORT = 0.42
XP_ACCESS_BETA_SHORT = 0.22
XP_ACCESS_BETA_LONG = 0.08
XP_ACCESS_MULT_FLOOR = 0.68
XP_ACCESS_DEEP_X = 66.0
XP_ACCESS_SHORT_CUTOFF_M = 15.0

GRID = xse.STUDY_GRID
BANDS = xse.DISTANCE_BAND_ORDER
BAND_LABELS = xse.DISTANCE_BAND_LABELS

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RIDGE_MODEL_PATH = MODELS_DIR / "xp_expected_ridge.joblib"
THREAT_THRESHOLDS_PATH = MODELS_DIR / "xp_threat_quantile.json"
XP_PASSES_PARQUET = DATA_DIR / "xp_passes_worldcup.parquet"
XP_EUROPEAN_PASSES_PARQUET = DATA_DIR / "xp_passes_european.parquet"
XP_META_PATH = DATA_DIR / "xp_season_meta.json"
XP_EUROPEAN_META_PATH = DATA_DIR / "xp_european_meta.json"


def _n_origin_cells() -> int:
    return GRID.od_origin_rows * GRID.od_origin_cols


def _n_dest_cells() -> int:
    return GRID.od_dest_rows * GRID.od_dest_cols


def attach_od_cells(passes: pd.DataFrame, grid: xse.GridConfig = GRID) -> pd.DataFrame:
    out = passes.copy()
    mask = out["is_won"] & out["has_end"]
    out["ox"] = -1
    out["oy"] = -1
    out["dx"] = -1
    out["dy"] = -1
    if not mask.any():
        out["distance_band"] = xse._distance_band_series(out["pass_distance"])
        return out
    sub = out.loc[mask]
    ox, oy = xse._cell_indices(
        sub["x_start"].to_numpy(dtype=float),
        sub["y_start"].to_numpy(dtype=float),
        cols=grid.od_origin_cols,
        rows=grid.od_origin_rows,
    )
    dx, dy = xse._cell_indices(
        sub["x_end"].to_numpy(dtype=float),
        sub["y_end"].to_numpy(dtype=float),
        cols=grid.od_dest_cols,
        rows=grid.od_dest_rows,
    )
    out.loc[mask, "ox"] = ox
    out.loc[mask, "oy"] = oy
    out.loc[mask, "dx"] = dx
    out.loc[mask, "dy"] = dy
    out["distance_band"] = xse._distance_band_series(out["pass_distance"])
    return out


def _progress_ratio_array(df: pd.DataFrame) -> np.ndarray:
    if "progress_ratio" in df.columns:
        return df["progress_ratio"].to_numpy(dtype=float)
    dist = np.maximum(df["pass_distance"].to_numpy(dtype=float), 0.5)
    dx = df["x_end"].to_numpy(dtype=float) - df["x_start"].to_numpy(dtype=float)
    return np.clip(dx / dist, -1.0, 1.0)


def _build_design_matrix(df: pd.DataFrame, grid: xse.GridConfig = GRID) -> sparse.csr_matrix:
    """Spatial features only: distance + origin/destination cells (no progress)."""
    dist = df["pass_distance"].to_numpy(dtype=float)
    dist_feats = np.column_stack([dist, dist ** 2, np.sqrt(dist)])
    n = len(df)
    o_idx = df["oy"].to_numpy(int) * grid.od_origin_cols + df["ox"].to_numpy(int)
    d_idx = df["dy"].to_numpy(int) * grid.od_dest_cols + df["dx"].to_numpy(int)
    n_o = _n_origin_cells()
    n_d = _n_dest_cells()
    return sparse.hstack([
        sparse.csr_matrix(dist_feats),
        sparse.csr_matrix((np.ones(n), (np.arange(n), o_idx)), shape=(n, n_o)),
        sparse.csr_matrix((np.ones(n), (np.arange(n), d_idx)), shape=(n, n_d)),
    ])


def _progress_multiplier_array(df: pd.DataFrame) -> np.ndarray:
    if "xp_progress_mult" in df.columns:
        return df["xp_progress_mult"].to_numpy(dtype=float)
    return xse.progress_toward_goal_multiplier(_progress_ratio_array(df))


def _cell_distance_array(
    oy: np.ndarray,
    ox: np.ndarray,
    dy: np.ndarray,
    dx: np.ndarray,
) -> np.ndarray:
    return np.sqrt((oy - dy) ** 2 + (ox - dx) ** 2)


def _field_pressure_array(x_start: np.ndarray) -> np.ndarray:
    """0 = deep own half, 1 = attacking third."""
    return 1.0 / (1.0 + np.exp(-(x_start - XP_ACCESS_PRESSURE_CENTER_X) / XP_ACCESS_PRESSURE_SCALE))


def accessibility_multiplier_array(df: pd.DataFrame) -> np.ndarray:
    """Model B: discount easy local passes, stronger in deep zones."""
    oy = df["oy"].to_numpy(int)
    ox = df["ox"].to_numpy(int)
    dy = df["dy"].to_numpy(int)
    dx = df["dx"].to_numpy(int)
    x_start = df["x_start"].to_numpy(dtype=float)
    dist_m = df["pass_distance"].to_numpy(dtype=float)

    locality = np.exp(-_cell_distance_array(oy, ox, dy, dx) / XP_ACCESS_LOCALITY_SCALE)
    pressure = _field_pressure_array(x_start)
    ease = locality * (1.0 - XP_ACCESS_PRESSURE_WEIGHT * pressure)

    beta = np.where(
        (x_start < XP_ACCESS_DEEP_X) & (dist_m <= XP_ACCESS_SHORT_CUTOFF_M),
        XP_ACCESS_BETA_DEEP_SHORT,
        np.where(dist_m <= xse.XP_DISTANCE_BAND_MAX_SHORT_M, XP_ACCESS_BETA_SHORT, XP_ACCESS_BETA_LONG),
    )
    return np.clip(1.0 - beta * ease, XP_ACCESS_MULT_FLOOR, 1.0)


def _accessibility_multiplier_array(df: pd.DataFrame) -> np.ndarray:
    if XP_ACCESSIBILITY_MULT_COL in df.columns:
        return df[XP_ACCESSIBILITY_MULT_COL].to_numpy(dtype=float)
    return accessibility_multiplier_array(df)


def _expected_xp_from_model(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Expected xP = Ridge(spatial) × progress × accessibility (same as Model 4)."""
    spatial = np.maximum(model.predict(_build_design_matrix(df)), 0.0)
    return spatial * _progress_multiplier_array(df) * _accessibility_multiplier_array(df)


def score_match_passes_m4(
    match_frame: pd.DataFrame,
    league: dict[str, np.ndarray | float | int],
    *,
    grid: xse.GridConfig = GRID,
    team_season_od: dict[str, np.ndarray] | None = None,
    team_n_matches: dict[str, int] | None = None,
) -> pd.DataFrame:
    passes = xse._enrich_match_passes(match_frame)
    passes = pe.filter_live_ball_passes(passes)
    if passes is None or passes.empty:
        return pd.DataFrame()
    _, count_grids = xse.build_team_xp_surfaces(passes, grid)
    scored = xse._assign_study_xp_models(
        passes,
        grid=grid,
        count_grids_by_team=count_grids,
        league=league,
        team_season_od=team_season_od,
        team_n_matches=team_n_matches,
    )
    scored = attach_od_cells(scored, grid)
    if "progress_ratio" not in scored.columns:
        scored["progress_ratio"] = xse._progress_ratio_series(scored)
    progress_mult = xse.progress_toward_goal_multiplier(scored["progress_ratio"].to_numpy(dtype=float))
    scored["xp_progress_mult"] = progress_mult
    scored[XP_ACCESSIBILITY_MULT_COL] = 1.0
    comp_mask = scored["is_won"] & scored["has_end"] & (scored["ox"] >= 0) & (scored["dx"] >= 0)
    if comp_mask.any():
        scored.loc[comp_mask, XP_ACCESSIBILITY_MULT_COL] = accessibility_multiplier_array(scored.loc[comp_mask])
    base_xp = scored[xse.XP_MODEL_COLUMNS[xse.XP_MODEL_HIER_OD]].to_numpy(dtype=float) * progress_mult
    scored[XP_COL] = np.minimum(
        base_xp * scored[XP_ACCESSIBILITY_MULT_COL].to_numpy(dtype=float),
        xse.XP_PASS_MAX,
    )
    return scored


def _build_team_season_od_maps(
    frame: pd.DataFrame,
    *,
    grid: xse.GridConfig = GRID,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    enriched_chunks: list[pd.DataFrame] = []
    for eid in frame["event_id"].astype(int).unique():
        mf = frame[frame["event_id"].astype(int) == int(eid)]
        ep = xse._enrich_match_passes(mf)
        ep = pe.filter_live_ball_passes(ep)
        if ep is not None and not ep.empty:
            enriched_chunks.append(ep[ep["is_won"] & ep["has_end"]])
    if not enriched_chunks:
        return {}, {}
    all_comp = pd.concat(enriched_chunks, ignore_index=True)
    team_season_od: dict[str, np.ndarray] = {}
    team_n_matches: dict[str, int] = {}
    for team, grp in all_comp.groupby("team", sort=False):
        team_season_od[str(team)] = xse._count_od_tensor(grp, grid)
        team_n_matches[str(team)] = int(grp["event_id"].nunique())
    return team_season_od, team_n_matches


def _score_frame_completed(
    frame: pd.DataFrame,
    league_ref: dict[str, np.ndarray | float | int],
    team_season_od: dict[str, np.ndarray] | None = None,
    team_n_matches: dict[str, int] | None = None,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for eid in frame["event_id"].astype(int).unique():
        mf = frame[frame["event_id"].astype(int) == int(eid)].copy()
        scored = score_match_passes_m4(
            mf,
            league_ref,
            team_season_od=team_season_od or {},
            team_n_matches=team_n_matches or {},
        )
        if scored.empty:
            continue
        comp = scored[scored["is_won"] & scored["has_end"]].copy()
        if comp.empty:
            continue
        chunks.append(comp)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _fit_artifacts_on_passes(train_passes: pd.DataFrame) -> dict:
    train = train_passes[
        (train_passes["ox"] >= 0)
        & (train_passes["dx"] >= 0)
    ].copy()
    if train.empty:
        raise RuntimeError("No completed passes available for xP artifact training.")

    X = _build_design_matrix(train)
    if XP_SPATIAL_COL not in train.columns:
        raise RuntimeError(f"Missing {XP_SPATIAL_COL} for Ridge training.")
    y = train[XP_SPATIAL_COL].to_numpy(dtype=float)
    model = Pipeline([
        ("ridge", Ridge(alpha=10.0, solver="lsqr")),
    ])
    model.fit(X, y)
    joblib.dump(model, RIDGE_MODEL_PATH)

    train[XP_EXPECTED_COL] = _expected_xp_from_model(model, train)
    train[XP_RESIDUAL_COL] = train[XP_COL].to_numpy(dtype=float) - train[XP_EXPECTED_COL]

    residual_thresholds: dict[str, float] = {}
    xp_thresholds: dict[str, float] = {}
    for band in BANDS:
        sub = train[train["distance_band"] == band]
        if sub.empty:
            residual_thresholds[band] = 0.0
            xp_thresholds[band] = 0.0
        else:
            residual_thresholds[band] = float(sub[XP_RESIDUAL_COL].quantile(1.0 - THREAT_QUANTILE))
            xp_thresholds[band] = float(sub[XP_COL].quantile(THREAT_XP_QUANTILE))

    meta = {
        "version": XP_MODEL_VERSION,
        "threat_rule": IMPACT_PASS_RULE_VERSION,
        "impact_pass_rule": IMPACT_PASS_RULE_LABEL,
        "impact_pass_rule_version": IMPACT_PASS_RULE_VERSION,
        "impact_score_weights": {
            "xp": IMPACT_SCORE_W_XP,
            "residual": IMPACT_SCORE_W_RESIDUAL,
            "progress": IMPACT_SCORE_W_PROGRESS,
        },
        "impact_score_percentile": IMPACT_SCORE_PERCENTILE,
        "impact_progress_percentile": IMPACT_PROGRESS_PERCENTILE,
        "threat_quantile": THREAT_QUANTILE,
        "threat_xp_quantile": THREAT_XP_QUANTILE,
        "threat_progress_min": THREAT_PROGRESS_MIN,
        "residual_thresholds": residual_thresholds,
        "residual_threshold_labels": {BAND_LABELS[k]: v for k, v in residual_thresholds.items()},
        "xp_thresholds": xp_thresholds,
        "xp_threshold_labels": {BAND_LABELS[k]: v for k, v in xp_thresholds.items()},
        "progress_floor_mult": xse.XP_PROGRESS_FLOOR_MULT,
        "progress_logistic_k": xse.XP_PROGRESS_LOGISTIC_K,
        "blend_alpha": xse.XP_BLEND_ALPHA,
        "grid": "12x8",
        "training_pool": "global",
        "team_surface": "global_reference_only",
        "ridge_target": XP_SPATIAL_COL,
        "expected_formula": "ridge_spatial * xp_progress_mult * xp_accessibility_mult",
        "accessibility_model": "locality_pressure_v1",
        "training_passes": int(len(train)),
        "training_matches": int(train["event_id"].nunique()) if "event_id" in train.columns else 0,
    }
    with open(THREAT_THRESHOLDS_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def _load_global_scored_completed_passes() -> pd.DataFrame:
    """Score completed passes from the global multi-league pool for Ridge training."""
    league_ref = xse._league_reference_surfaces(
        GRID.dest_cols, GRID.dest_rows,
        GRID.od_origin_cols, GRID.od_origin_rows,
        GRID.od_dest_cols, GRID.od_dest_rows,
    )
    frame = xse._load_combined_league_pass_frame()
    if frame.empty:
        raise RuntimeError("No passes available in the global xP reference pool.")
    return _score_frame_completed(frame, league_ref)


def fit_and_save_artifacts(*, force: bool = False) -> dict:
    """Train expected-xP ridge and quantile threat thresholds on the global pass pool."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if (
        not force
        and RIDGE_MODEL_PATH.exists()
        and THREAT_THRESHOLDS_PATH.exists()
    ):
        with open(THREAT_THRESHOLDS_PATH, encoding="utf-8") as fh:
            meta = json.load(fh)
        if str(meta.get("version", "")) == XP_MODEL_VERSION:
            return meta

    league_passes = _load_global_scored_completed_passes()
    if league_passes.empty:
        raise RuntimeError("No globally scored passes available for xP artifact training.")
    return _fit_artifacts_on_passes(league_passes)


def load_threat_meta() -> dict:
    if not THREAT_THRESHOLDS_PATH.exists():
        fit_and_save_artifacts()
    with open(THREAT_THRESHOLDS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def load_threat_thresholds() -> dict[str, float]:
    meta = load_threat_meta()
    return {str(k): float(v) for k, v in meta["residual_thresholds"].items()}


def load_threat_xp_thresholds() -> dict[str, float]:
    meta = load_threat_meta()
    xp_thresholds = meta.get("xp_thresholds")
    if not xp_thresholds:
        fit_and_save_artifacts(force=True)
        meta = load_threat_meta()
        xp_thresholds = meta.get("xp_thresholds") or {}
    return {str(k): float(v) for k, v in xp_thresholds.items()}


def load_expected_model() -> Pipeline:
    if not RIDGE_MODEL_PATH.exists():
        fit_and_save_artifacts()
    return joblib.load(RIDGE_MODEL_PATH)


def _robust_zscore_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        q75, q25 = np.percentile(arr, [75, 25])
        scale = max(float(q75 - q25) / 1.349, 1e-9)
    return (arr - med) / scale


def _robust_zscore_by_band(series: pd.Series, bands: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    for band, idx in bands.groupby(bands, sort=False).groups.items():
        out.loc[idx] = _robust_zscore_array(series.loc[idx].to_numpy(dtype=float))
    return out


def _composite_impact_pass_flags(sub: pd.DataFrame) -> np.ndarray:
    """Hybrid impact pass: weighted robust-z score + forward-progress gate per band."""
    if sub.empty:
        return np.zeros(0, dtype=bool)

    work = sub.copy()
    if "progress_ratio" not in work.columns:
        work["progress_ratio"] = _progress_ratio_array(work)
    bands = work["distance_band"].astype(str)

    z_xp = _robust_zscore_by_band(work[XP_COL].astype(float), bands)
    z_res = _robust_zscore_by_band(work[XP_RESIDUAL_COL].astype(float), bands)
    z_prog = _robust_zscore_by_band(work["progress_ratio"].astype(float), bands)
    impact_score = (
        IMPACT_SCORE_W_XP * z_xp
        + IMPACT_SCORE_W_RESIDUAL * z_res
        + IMPACT_SCORE_W_PROGRESS * z_prog
    )

    score_cut = impact_score.groupby(bands, sort=False).transform(
        lambda s: s.quantile(IMPACT_SCORE_PERCENTILE)
    )
    prog_cut = work["progress_ratio"].groupby(bands, sort=False).transform(
        lambda s: s.quantile(IMPACT_PROGRESS_PERCENTILE)
    )
    return ((impact_score >= score_cut) & (work["progress_ratio"] >= prog_cut)).to_numpy(dtype=bool)


def apply_expected_and_threat(passes: pd.DataFrame) -> pd.DataFrame:
    out = passes.copy()
    out[XP_EXPECTED_COL] = 0.0
    out[XP_RESIDUAL_COL] = 0.0
    out[THREAT_COL] = False
    mask = out["is_won"] & out["has_end"] & (out["ox"] >= 0) & (out["dx"] >= 0)
    if not mask.any():
        return out

    model = load_expected_model()
    sub_idx = out.index[mask]
    sub = out.loc[mask]
    expected = _expected_xp_from_model(model, sub)
    residual = sub[XP_COL].to_numpy(dtype=float) - expected
    out.loc[sub_idx, XP_EXPECTED_COL] = expected
    out.loc[sub_idx, XP_RESIDUAL_COL] = residual

    threat_sub = sub.copy()
    threat_sub[XP_RESIDUAL_COL] = residual
    out.loc[sub_idx, THREAT_COL] = _composite_impact_pass_flags(threat_sub)
    return out


def _refresh_threat_flags_if_needed(
    passes: pd.DataFrame,
    meta_path: Path,
    parquet_path: Path,
) -> pd.DataFrame:
    """Re-apply impact-pass flags when the composite rule version changes."""
    if not meta_path.exists():
        return apply_expected_and_threat(passes)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if str(meta.get("impact_pass_rule_version", "")) == IMPACT_PASS_RULE_VERSION:
        return passes

    refreshed = apply_expected_and_threat(passes)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    refreshed.to_parquet(parquet_path, index=False)
    meta = {
        **meta,
        "impact_pass_rule_version": IMPACT_PASS_RULE_VERSION,
        "impact_pass_rule": IMPACT_PASS_RULE_LABEL,
        "threats": int(refreshed[THREAT_COL].sum()),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return refreshed


def build_world_cup_season_passes(*, force_artifacts: bool = False) -> pd.DataFrame:
    season = _build_season_passes_from_frame(
        pe._load_season_pass_frame(),
        force_artifacts=force_artifacts,
    )
    if season.empty:
        return season
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    season.to_parquet(XP_PASSES_PARQUET, index=False)
    meta = {
        "version": XP_MODEL_VERSION,
        "impact_pass_rule_version": IMPACT_PASS_RULE_VERSION,
        "impact_pass_rule": IMPACT_PASS_RULE_LABEL,
        "passes": int(len(season)),
        "completed": int((season["is_won"] & season["has_end"]).sum()),
        "threats": int(season[THREAT_COL].sum()),
        "players": int(season["player_id"].nunique()),
        "matches": int(season["event_id"].nunique()) if "event_id" in season.columns else 0,
    }
    with open(XP_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return season


def _build_season_passes_from_frame(
    frame: pd.DataFrame,
    *,
    force_artifacts: bool = False,
    blend_league_reference: bool = False,
    refit_artifacts: bool | None = None,
) -> pd.DataFrame:
    grid_dims = (
        GRID.dest_cols, GRID.dest_rows,
        GRID.od_origin_cols, GRID.od_origin_rows,
        GRID.od_dest_cols, GRID.od_dest_rows,
    )
    global_ref = xse._league_reference_surfaces(*grid_dims)
    if frame.empty:
        return pd.DataFrame()

    team_season_od, team_n_matches = _build_team_season_od_maps(frame)
    league_ref_cache: dict[str, dict] = {}

    chunks: list[pd.DataFrame] = []
    for eid in frame["event_id"].astype(int).unique():
        mf = frame[frame["event_id"].astype(int) == int(eid)].copy()
        league_ref = global_ref
        if blend_league_reference and "league_source" in mf.columns:
            league_values = mf["league_source"].dropna().astype(str)
            if not league_values.empty:
                league_source = str(league_values.iloc[0])
                if league_source not in league_ref_cache:
                    league_ref_cache[league_source] = xse.reference_surfaces_for_league_source(
                        league_source,
                        *grid_dims,
                    )
                league_ref = league_ref_cache[league_source]
        scored = score_match_passes_m4(
            mf,
            league_ref,
            team_season_od=team_season_od or {},
            team_n_matches=team_n_matches or {},
        )
        if scored.empty:
            continue
        chunks.append(scored)

    if not chunks:
        return pd.DataFrame()
    season_raw = pd.concat(chunks, ignore_index=True)

    should_refit = force_artifacts if refit_artifacts is None else refit_artifacts
    if not should_refit and THREAT_THRESHOLDS_PATH.exists():
        with open(THREAT_THRESHOLDS_PATH, encoding="utf-8") as fh:
            meta = json.load(fh)
        should_refit = str(meta.get("version", "")) != XP_MODEL_VERSION
    elif not should_refit and not THREAT_THRESHOLDS_PATH.exists():
        should_refit = True

    if should_refit:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        global_passes = _load_global_scored_completed_passes()
        _fit_artifacts_on_passes(global_passes)

    return apply_expected_and_threat(season_raw)


def build_european_league_season_passes(
    *,
    force_artifacts: bool = False,
    refit_artifacts: bool | None = None,
) -> pd.DataFrame:
    frame = pe._filter_pass_frame_to_midfielders(pe._load_european_league_pass_frame())
    season = _build_season_passes_from_frame(
        frame,
        force_artifacts=force_artifacts,
        blend_league_reference=True,
        refit_artifacts=refit_artifacts,
    )
    if season.empty:
        return season
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    season.to_parquet(XP_EUROPEAN_PASSES_PARQUET, index=False)
    meta = {
        "version": XP_MODEL_VERSION,
        "impact_pass_rule_version": IMPACT_PASS_RULE_VERSION,
        "impact_pass_rule": IMPACT_PASS_RULE_LABEL,
        "reference_global_weight": xse.XP_REFERENCE_GLOBAL_WEIGHT,
        "reference_league_weight": xse.XP_REFERENCE_LEAGUE_WEIGHT,
        "passes": int(len(season)),
        "completed": int((season["is_won"] & season["has_end"]).sum()),
        "threats": int(season[THREAT_COL].sum()),
        "players": int(season["player_id"].nunique()),
        "matches": int(season["event_id"].nunique()) if "event_id" in season.columns else 0,
    }
    with open(XP_EUROPEAN_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return season


def load_european_season_passes(*, rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not XP_EUROPEAN_PASSES_PARQUET.exists():
        return build_european_league_season_passes(force_artifacts=rebuild)
    df = pd.read_parquet(XP_EUROPEAN_PASSES_PARQUET)
    if (
        THREAT_COL not in df.columns
        or XP_COL not in df.columns
        or "xp_progress_mult" not in df.columns
        or XP_ACCESSIBILITY_MULT_COL not in df.columns
    ):
        return build_european_league_season_passes(refit_artifacts=False)
    if XP_EUROPEAN_META_PATH.exists():
        with open(XP_EUROPEAN_META_PATH, encoding="utf-8") as fh:
            meta = json.load(fh)
        if str(meta.get("version", "")) != XP_MODEL_VERSION:
            # Rescore European parquet only — never refit global Ridge on user load.
            return build_european_league_season_passes(refit_artifacts=False)
        if str(meta.get("impact_pass_rule_version", "")) != IMPACT_PASS_RULE_VERSION:
            return _refresh_threat_flags_if_needed(
                df,
                XP_EUROPEAN_META_PATH,
                XP_EUROPEAN_PASSES_PARQUET,
            )
    return df


def load_season_passes(*, rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not XP_PASSES_PARQUET.exists():
        return build_world_cup_season_passes(force_artifacts=rebuild)
    df = pd.read_parquet(XP_PASSES_PARQUET)
    if (
        THREAT_COL not in df.columns
        or XP_COL not in df.columns
        or "xp_progress_mult" not in df.columns
        or XP_ACCESSIBILITY_MULT_COL not in df.columns
    ):
        return build_world_cup_season_passes(force_artifacts=True)
    if XP_META_PATH.exists():
        with open(XP_META_PATH, encoding="utf-8") as fh:
            meta = json.load(fh)
        if str(meta.get("version", "")) != XP_MODEL_VERSION:
            return build_world_cup_season_passes(force_artifacts=True)
        if str(meta.get("impact_pass_rule_version", "")) != IMPACT_PASS_RULE_VERSION:
            return _refresh_threat_flags_if_needed(df, XP_META_PATH, XP_PASSES_PARQUET)
    return df


@functools.lru_cache(maxsize=4)
def load_xp_passes_grouped(cache_version: int = XP_DATA_CACHE_VERSION) -> dict[str, pd.DataFrame]:
    _ = cache_version
    season = load_season_passes()
    if season.empty:
        return {}
    return {str(pid): grp for pid, grp in season.groupby("player_id", sort=False)}


@functools.lru_cache(maxsize=4)
def load_european_league_season_passes(cache_version: int = XP_DATA_CACHE_VERSION) -> pd.DataFrame:
    _ = cache_version
    return load_european_season_passes()


@functools.lru_cache(maxsize=4)
def load_european_league_xp_passes_grouped(
    cache_version: int = XP_DATA_CACHE_VERSION,
) -> dict[str, pd.DataFrame]:
    _ = cache_version
    season = load_european_league_season_passes()
    if season.empty:
        return {}
    return {str(pid): grp for pid, grp in season.groupby("player_id", sort=False)}


def compute_player_xp_metrics(grp: pd.DataFrame) -> dict[str, float | int]:
    scored = grp[grp["is_won"] & grp["has_end"]]
    if scored.empty or XP_COL not in scored.columns:
        return {}
    n_passes = len(scored)
    out: dict[str, float | int] = {
        "xp_m4_total": float(scored[XP_COL].sum()),
        "xp_m4_per_pass": float(scored[XP_COL].mean()),
        "xp_m4_p90": float(scored[XP_COL].quantile(0.90)),
        "xp_m4_threat_passes": int(scored[THREAT_COL].sum()) if THREAT_COL in scored.columns else 0,
        "xp_m4_threat_rate": float(scored[THREAT_COL].mean()) if THREAT_COL in scored.columns else 0.0,
        "xp_m4_threat_xp_total": (
            float(scored.loc[scored[THREAT_COL], XP_COL].sum())
            if THREAT_COL in scored.columns and scored[THREAT_COL].any()
            else 0.0
        ),
        "xp_m4_per_threat_pass": (
            float(scored.loc[scored[THREAT_COL], XP_COL].mean())
            if THREAT_COL in scored.columns and scored[THREAT_COL].any()
            else 0.0
        ),
        "pass_mean_distance": (
            float(scored["pass_distance"].mean())
            if n_passes and "pass_distance" in scored.columns
            else 0.0
        ),
    }
    for band in BANDS:
        sub = scored[scored["distance_band"] == band]
        n_band = len(sub)
        out[f"passes_{band}"] = int(n_band)
        out[f"xp_m4_threat_{band}"] = int(sub[THREAT_COL].sum()) if THREAT_COL in sub.columns and n_band else 0
        out[f"xp_m4_mean_{band}"] = float(sub[XP_COL].mean()) if n_band else 0.0
        out[f"xp_m4_per_pass_{band}"] = float(sub[XP_COL].mean()) if n_band else 0.0
        out[f"xp_m4_total_{band}"] = float(sub[XP_COL].sum()) if n_band else 0.0
        out[f"xp_m4_threat_rate_{band}"] = (
            float(sub[THREAT_COL].mean()) if THREAT_COL in sub.columns and n_band else 0.0
        )
    return out


def build_xp_analytics(
    cache_version: int = XP_DATA_CACHE_VERSION,
) -> tuple[list[dict], list[dict]]:
    import xp_stats_engine as xstats

    _ = cache_version
    season = load_season_passes()
    frame = pe._load_season_pass_frame()
    if season.empty or frame.empty:
        return [], []

    registry = pe.build_player_registry(frame)
    minutes_info = pe._load_minutes_info(frame)
    players: list[dict] = []

    for player in registry:
        if not pe.is_outfield_position(player.get("position")):
            continue
        pid = player["code"]
        grp = season[season["player_id"].astype(str) == str(pid)]
        if grp.empty:
            continue
        mins = minutes_info.get(pid, {})
        metrics = xstats.compute_extended_xp_stats(grp)
        if not metrics:
            continue
        minutes = mins.get("minutes")
        player_raw = frame[
            (frame["player_id"].astype(str) == str(pid))
            & (frame["category"].astype(str).str.lower() == "passes")
        ]
        xstats.attach_regular_pass_stats(metrics, player_raw, minutes)
        xstats.apply_per90_metrics(metrics, minutes)
        players.append({
            "player_id": pid,
            "player_name": player["name"],
            "position": player.get("position", "—"),
            "position_group": pe.rating_position_group(player.get("position")),
            "team": mins.get("team", str(grp["team"].mode().iloc[0] if not grp["team"].mode().empty else "—")),
            "minutes": mins.get("minutes"),
            "minutes_pct": mins.get("minutes_pct"),
            "passes_completed": int((grp["is_won"] & grp["has_end"]).sum()),
            **metrics,
        })

    players.sort(key=lambda p: float(p.get("xp_m4_total", 0.0)), reverse=True)
    for i, p in enumerate(players, start=1):
        p["xp_m4_rank"] = i
    xstats.attach_distance_indices(players)
    xstats.attach_pass_length_profile(players)
    xstats.attach_composite_indices(players)
    xstats.attach_xp_pass_ratings(players)
    xstats.attach_all_stats_ranks(players)
    attach_xp_metric_ranks(players)
    return registry, players


def build_european_league_xp_analytics(
    cache_version: int = XP_DATA_CACHE_VERSION,
    *,
    min_passes: int = 100,
) -> tuple[list[dict], list[dict]]:
    """xP metrics for midfielders in Premier League, Serie A, La Liga, Bundesliga and Ligue 1."""
    import xp_stats_engine as xstats

    _ = cache_version
    season = load_european_league_season_passes()
    if season.empty:
        return [], []

    minutes_info = pe._minutes_from_passes_frame(season)
    league_by_player: dict[str, str] = {}
    if "league_source" in season.columns:
        league_by_player = (
            season.groupby("player_id", sort=False)["league_source"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
            .astype(str)
            .to_dict()
        )
    registry = pe.build_player_registry(season)
    raw_pass_frame = pe._filter_pass_frame_to_midfielders(pe._load_european_league_pass_frame())
    players: list[dict] = []
    registry_by_id = {str(p["code"]): p for p in registry}

    for pid, grp in season.groupby("player_id", sort=False):
        pid = str(pid)
        player = registry_by_id.get(pid)
        if player is None:
            continue
        completed = int((grp["is_won"] & grp["has_end"]).sum())
        if completed < min_passes:
            continue
        mins = minutes_info.get(pid, {})
        metrics = xstats.compute_extended_xp_stats(grp)
        if not metrics:
            continue
        minutes = mins.get("minutes")
        player_raw = raw_pass_frame[raw_pass_frame["player_id"].astype(str) == pid]
        xstats.attach_regular_pass_stats(metrics, player_raw, minutes)
        xstats.apply_per90_metrics(metrics, minutes)
        league_source = str(league_by_player.get(pid, ""))
        players.append({
            "player_id": pid,
            "player_name": player["name"],
            "position": player.get("position", "—"),
            "position_group": pe.rating_position_group(player.get("position")),
            "team": mins.get("team", str(grp["team"].mode().iloc[0] if "team" in grp.columns and not grp["team"].mode().empty else "—")),
            "minutes": mins.get("minutes"),
            "minutes_pct": mins.get("minutes_pct"),
            "league": pe._european_league_label(league_source),
            "league_source": league_source,
            "passes_completed": completed,
            **metrics,
        })

    players.sort(key=lambda p: float(p.get("xp_m4_total", 0.0)), reverse=True)
    for i, p in enumerate(players, start=1):
        p["xp_m4_rank"] = i
    xstats.attach_distance_indices(players)
    xstats.attach_pass_length_profile(players)
    xstats.attach_composite_indices(players)
    xstats.attach_xp_pass_ratings(players)
    xstats.attach_all_stats_ranks(players)
    attach_xp_metric_ranks(players)
    return registry, players


def attach_xp_metric_ranks(players: list[dict]) -> None:
    """Attach within-position ranks for core xP dashboard metrics (eligible peers only)."""
    import xp_stats_engine as xstats

    xstats.attach_metric_ranks_within_position(
        players,
        XP_POSITION_RANK_METRICS,
        eligible_only=True,
    )


def refresh_xp_midfield_origin_rankings(players: list[dict]) -> None:
    """Recompute xP ranks after campo ofensivo / campo defensivo groups are assigned."""
    import xp_stats_engine as xstats

    xstats.attach_distance_indices(players)
    xstats.attach_pass_length_profile(players)
    xstats.attach_composite_indices(players)
    xstats.attach_xp_pass_ratings(players)
    xstats.attach_all_stats_ranks(players)
    attach_xp_metric_ranks(players)


def rank_xp_players_by_position(players: list[dict]) -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = {}
    for p in players:
        grp = str(p.get("position_group") or "CM")
        pools.setdefault(grp, []).append(p)
    for grp, rows in pools.items():
        rows.sort(key=lambda r: float(r.get("xp_m4_total", 0.0)), reverse=True)
        for i, row in enumerate(rows, start=1):
            row["xp_m4_rank_in_group"] = i
    return pools


def season_meta() -> dict:
    if XP_META_PATH.exists():
        with open(XP_META_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    season = load_season_passes()
    if season.empty:
        return {}
    return {
        "version": XP_MODEL_VERSION,
        "passes": int(len(season)),
        "threats": int(season[THREAT_COL].sum()),
        "players": int(season["player_id"].nunique()),
    }
