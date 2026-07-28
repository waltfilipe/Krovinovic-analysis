"""Extended xP player stats for the Stats tab."""

from __future__ import annotations

import numpy as np
import pandas as pd

import passes_engine as pe
import xp_study_engine as xse

FIELD_X = pe.FIELD_X
FIELD_Y = pe.FIELD_Y
DEF_FIELD_SHARE = 0.40
FINAL_FIELD_SHARE = 0.40
DEF_X_MAX = FIELD_X * DEF_FIELD_SHARE
FINAL_X_MIN = FIELD_X * (1.0 - FINAL_FIELD_SHARE)
FIRST_THIRD_X = FIELD_X / 3.0
CENTRAL_Y_MIN = 20.0
CENTRAL_Y_MAX = 60.0
LINE_BREAK_FORWARD_ANGLE_DEG = 50.0
LINE_BREAK_ORIGIN_MIN_X = 30.0
LINE_BREAK_ORIGIN_ZONE1_MAX_X = 60.0
LINE_BREAK_ORIGIN_ZONE2_MAX_X = 80.0
LINE_BREAK_ORIGIN_ZONE3_MAX_X = FIELD_X
LINE_BREAK_ORIGIN_LATERAL_EXCLUDE_SHARE = 0.10
LINE_BREAK_DEST_LATERAL_EXCLUDE_SHARE = 0.15
LINE_BREAK_DIST_MIN_ZONE1_M = 20.0
LINE_BREAK_DIST_MIN_ZONE2_M = 15.0
LINE_BREAK_DIST_MIN_ZONE3_M = 10.0
LINE_BREAK_DIST_MAX_M = 30.0
CROSS_DIST_MIN_M = 15.0
CROSS_LATERAL_DELTA_MIN_M = 8.0
CROSS_MAX_START_X = 102.0
FROM_DEEP_DIST_MIN_M = 15.0
PENALTY_X_MIN = pe.PENALTY_BOX_X_MIN
PENALTY_Y_MIN = pe.PENALTY_BOX_Y_MIN
PENALTY_Y_MAX = pe.PENALTY_BOX_Y_MAX

XP_COL = "xp_m4"
THREAT_COL = "is_threat_m4"
IMPACT_PASS_ABBR = "I.P."
RESIDUAL_COL = "xp_residual"
DISTANCE_BAND_LABELS = xse.DISTANCE_BAND_LABELS
XP_DISTANCE_BAND_MAX_SHORT_M = xse.XP_DISTANCE_BAND_MAX_SHORT_M
BANDS = xse.DISTANCE_BAND_ORDER
DISTANCE_INDEX_MIN_PASS_PERCENTILE = 30
XP_PROFILE_MIN_MINUTES_PCT = 0.30
XP_PROFILE_BAR_PASS_PERCENTILE = DISTANCE_INDEX_MIN_PASS_PERCENTILE
XP_PROFILE_TOP_PASS_POOL_SIZE = 250

DISTANCE_INDEX_GRADES: tuple[tuple[str, float], ...] = (
    ("Good", 0.20),
    ("Above Average", 0.40),
    ("Average", 0.60),
    ("Under Average", 0.80),
    ("Poor", 1.00),
)
DISTANCE_INDEX_GRADE_ORDER: dict[str, int] = {
    "Poor": 1,
    "Under Average": 2,
    "Average": 3,
    "Above Average": 4,
    "Good": 5,
}
# Skill metrics share the bulk of the index; volume enters with a small weight.
DISTANCE_INDEX_SKILL_WEIGHT = 0.30
DISTANCE_INDEX_VOLUME_WEIGHT = 0.10
DISTANCE_INDEX_BALANCE_MIN_WEIGHT = 0.40
DISTANCE_INDEX_BALANCE_MEAN_WEIGHT = 0.60
DISTANCE_INDEX_VOLUME_GRADE_PENALTY_PCTS: tuple[tuple[float, int], ...] = (
    (0.85, 2),
    (0.70, 1),
)


def _zone_x(x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), "mid", dtype=object)
    out[x <= FIRST_THIRD_X] = "def"
    out[x > FINAL_X_MIN] = "att"
    return out


def _is_left_corridor(y: np.ndarray) -> np.ndarray:
    return y < CENTRAL_Y_MIN


def _is_right_corridor(y: np.ndarray) -> np.ndarray:
    return y > CENTRAL_Y_MAX


def _is_central_corridor(y: np.ndarray) -> np.ndarray:
    return (y >= CENTRAL_Y_MIN) & (y <= CENTRAL_Y_MAX)


def _is_lateral_corridor(y: np.ndarray) -> np.ndarray:
    return _is_left_corridor(y) | _is_right_corridor(y)


def _is_diagonal_long_pass(y_start: np.ndarray, y_end: np.ndarray) -> np.ndarray:
    """Swap laterally: left/central -> right, or right/central -> left."""
    to_right = (
        (_is_left_corridor(y_start) | _is_central_corridor(y_start))
        & _is_right_corridor(y_end)
    )
    to_left = (
        (_is_right_corridor(y_start) | _is_central_corridor(y_start))
        & _is_left_corridor(y_end)
    )
    return to_right | to_left


def _not_in_outer_lateral_band(y: np.ndarray, exclude_share: float) -> np.ndarray:
    """True when y is outside the outer exclude_share fraction on each touchline."""
    margin = FIELD_Y * float(exclude_share)
    return (y >= margin) & (y <= FIELD_Y - margin)


def _line_break_origin_ok(y: np.ndarray) -> np.ndarray:
    """Origin allowed everywhere except the outer 10% lateral bands."""
    return _not_in_outer_lateral_band(y, LINE_BREAK_ORIGIN_LATERAL_EXCLUDE_SHARE)


def _line_break_destination_ok(y: np.ndarray) -> np.ndarray:
    """Destination allowed everywhere except the outer 15% lateral bands."""
    return _not_in_outer_lateral_band(y, LINE_BREAK_DEST_LATERAL_EXCLUDE_SHARE)


def _line_break_distance_ok(x_start: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Distance bands by origin x: 30–60 m → 20–30 m; 60–80 → 15–30; 80–120 → 10–30."""
    zone1 = (x_start >= LINE_BREAK_ORIGIN_MIN_X) & (x_start <= LINE_BREAK_ORIGIN_ZONE1_MAX_X)
    zone2 = (x_start > LINE_BREAK_ORIGIN_ZONE1_MAX_X) & (x_start <= LINE_BREAK_ORIGIN_ZONE2_MAX_X)
    zone3 = (x_start > LINE_BREAK_ORIGIN_ZONE2_MAX_X) & (x_start <= LINE_BREAK_ORIGIN_ZONE3_MAX_X)
    return (
        (zone1 & (dist >= LINE_BREAK_DIST_MIN_ZONE1_M) & (dist <= LINE_BREAK_DIST_MAX_M))
        | (zone2 & (dist >= LINE_BREAK_DIST_MIN_ZONE2_M) & (dist <= LINE_BREAK_DIST_MAX_M))
        | (zone3 & (dist >= LINE_BREAK_DIST_MIN_ZONE3_M) & (dist <= LINE_BREAK_DIST_MAX_M))
    )


def _is_forward_angle(dx: np.ndarray, dy: np.ndarray, *, max_angle_deg: float) -> np.ndarray:
    """True when the pass aims forward (+x) within ±max_angle_deg of the goal direction."""
    forward = dx > 0.0
    angle_deg = np.degrees(np.arctan2(dy, np.where(forward, dx, 1.0)))
    return forward & (np.abs(angle_deg) <= max_angle_deg)


def _is_left_right_inversion(y_start: np.ndarray, y_end: np.ndarray) -> np.ndarray:
    """Long pass that switches directly between left and right lateral corridors."""
    left_to_right = _is_left_corridor(y_start) & _is_right_corridor(y_end)
    right_to_left = _is_right_corridor(y_start) & _is_left_corridor(y_end)
    return left_to_right | right_to_left


def _in_penalty_box(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x >= PENALTY_X_MIN) & (y >= PENALTY_Y_MIN) & (y <= PENALTY_Y_MAX)


def _is_long_pass(scored: pd.DataFrame, dist: np.ndarray) -> np.ndarray:
    if "distance_band" in scored.columns:
        return scored["distance_band"].astype(str).to_numpy() == "long"
    return dist > XP_DISTANCE_BAND_MAX_SHORT_M


SPECIAL_PASS_MAP_FILTERS: tuple[tuple[str, str], ...] = (
    ("progressive", "Progressive Passes"),
    ("diagonal_long", "Long Diagonal"),
    ("line_break", "Line Break"),
    ("inversion", "Inversions"),
    ("cross", "Cross"),
    ("from_deep", "xP from Deep"),
    ("final_third", "% xP in Final Third"),
    ("in_box", "% xP in Box"),
)
SPECIAL_PASS_MAP_FILTER_KEYS: tuple[str, ...] = tuple(key for key, _label in SPECIAL_PASS_MAP_FILTERS)
SPECIAL_PASS_MAP_FILTER_LABELS: dict[str, str] = dict(SPECIAL_PASS_MAP_FILTERS)
SPECIAL_PASS_COUNT_KEYS: tuple[str, ...] = SPECIAL_PASS_MAP_FILTER_KEYS

# Maps tab — selectable pass types grouped by stat type.
MAPS_REGULAR_PASS_OPTIONS: tuple[tuple[str, str], ...] = (
    ("progressive", "Progressive Passes"),
    ("into_final_third", "Passes into Final Third"),
    ("into_box", "Passes into Box"),
)
MAPS_SPECIAL_PASS_OPTIONS: tuple[tuple[str, str], ...] = (
    ("xp_threat_all", f"xP {IMPACT_PASS_ABBR}"),
    (
        "xp_threat_short",
        f"xP {IMPACT_PASS_ABBR} · Short ({DISTANCE_BAND_LABELS['short']})",
    ),
    (
        "xp_threat_long",
        f"xP {IMPACT_PASS_ABBR} · Long ({DISTANCE_BAND_LABELS['long']})",
    ),
    ("diagonal_long", "Long Diagonal"),
    ("line_break", "Line Break"),
    ("top_residual", "Top Residual"),
)
MAPS_TOP_RESIDUAL_PASS_KEY = "top_residual"
MAPS_TOP_RESIDUAL_N = 20
MAPS_STAT_TYPE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("regular", "Regular Stats"),
    ("special", "xP Stats"),
)
MAPS_PASS_TYPE_OPTIONS: tuple[tuple[str, str], ...] = (
    *MAPS_REGULAR_PASS_OPTIONS,
    *MAPS_SPECIAL_PASS_OPTIONS,
)
MAPS_PASS_TYPE_LABELS: dict[str, str] = dict(MAPS_PASS_TYPE_OPTIONS)
MAPS_SPECIAL_PASS_TYPE_KEYS: frozenset[str] = frozenset(
    key for key, _label in MAPS_SPECIAL_PASS_OPTIONS
)


def maps_stat_type_options() -> tuple[tuple[str, str], ...]:
    return MAPS_STAT_TYPE_OPTIONS


def maps_pass_options_for_type(stat_type: str) -> tuple[tuple[str, str], ...]:
    if str(stat_type) == "special":
        return MAPS_SPECIAL_PASS_OPTIONS
    return MAPS_REGULAR_PASS_OPTIONS


def maps_pass_type_label(filter_key: str) -> str:
    return MAPS_PASS_TYPE_LABELS.get(str(filter_key), str(filter_key))


def is_maps_special_pass(filter_key: str) -> bool:
    return str(filter_key) in MAPS_SPECIAL_PASS_TYPE_KEYS


def _xp_threat_map_band(filter_key: str) -> str | None:
    key = str(filter_key or "").strip()
    if key == "xp_threat_all":
        return "all"
    if key == "xp_threat_short":
        return "short"
    if key == "xp_threat_long":
        return "long"
    return None


def is_maps_xp_threat_pass(filter_key: str) -> bool:
    return _xp_threat_map_band(filter_key) is not None


def is_maps_top_residual_pass(filter_key: str) -> bool:
    return str(filter_key or "").strip() == MAPS_TOP_RESIDUAL_PASS_KEY


def _ensure_residual_column(work: pd.DataFrame) -> pd.DataFrame:
    if work.empty:
        return work
    if RESIDUAL_COL not in work.columns and {"xp_m4", "xp_expected"}.issubset(work.columns):
        out = work.copy()
        out[RESIDUAL_COL] = out["xp_m4"].astype(float) - out["xp_expected"].astype(float)
        return out
    return work


def filter_top_residual_passes(
    passes: pd.DataFrame,
    *,
    n: int = MAPS_TOP_RESIDUAL_N,
) -> pd.DataFrame:
    """Top-N completed passes by xP residual (actual − expected)."""
    work = _ensure_residual_column(_completed_pass_frame(passes))
    if work.empty or RESIDUAL_COL not in work.columns:
        return work.iloc[0:0].copy()
    return work.sort_values(RESIDUAL_COL, ascending=False).head(int(n)).reset_index(drop=True)


def filter_passes_by_map_round(passes: pd.DataFrame, round_key: str) -> pd.DataFrame:
    """Filter completed passes to one match (rodada) or keep all when round_key is 'all'."""
    if passes is None or passes.empty:
        return passes
    key = str(round_key or "all").strip()
    if key in {"", "all"}:
        return passes
    if key.startswith("event:") and "event_id" in passes.columns:
        event_id = int(key.split(":", 1)[1])
        return passes[passes["event_id"].astype(int) == event_id].copy()
    return passes


def map_round_options(passes: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Build selectbox options for per-match / all-round pass maps."""
    labels = {"all": "Todas as rodadas"}
    keys = ["all"]
    if passes is None or passes.empty or "event_id" not in passes.columns:
        return keys, labels
    work = passes.dropna(subset=["event_id"]).copy()
    if work.empty:
        return keys, labels
    agg: dict[str, str] = {"match_date": "first"}
    if "home_team" in work.columns:
        agg["home_team"] = "first"
    if "away_team" in work.columns:
        agg["away_team"] = "first"
    matches = work.groupby("event_id", as_index=False).agg(agg).sort_values("match_date")
    for idx, row in enumerate(matches.itertuples(index=False), start=1):
        event_id = int(getattr(row, "event_id"))
        date_txt = str(getattr(row, "match_date", "") or "—")[:10]
        home = str(getattr(row, "home_team", "") or "—")
        away = str(getattr(row, "away_team", "") or "—")
        key = f"event:{event_id}"
        labels[key] = f"Rodada {idx} · {date_txt} · {home} vs {away}"
        keys.append(key)
    return keys, labels


def filter_passes_for_map(passes: pd.DataFrame, filter_key: str) -> pd.DataFrame:
    """Return completed passes matching a Maps pass-type selection."""
    work = _completed_pass_frame(passes)
    if work.empty:
        return work
    key = str(filter_key or "").strip()
    if key == MAPS_TOP_RESIDUAL_PASS_KEY:
        return filter_top_residual_passes(passes)
    threat_band = _xp_threat_map_band(key)
    if threat_band is not None:
        return filter_passes_by_threat_type(work, threat_band)
    if key == "into_final_third":
        x_end = work["x_end"].to_numpy(dtype=float)
        return work.loc[x_end >= pe.FINAL_THIRD_LINE_X].copy()
    if key == "into_box":
        x_end = work["x_end"].to_numpy(dtype=float)
        y_end = work["y_end"].to_numpy(dtype=float)
        return work.loc[_in_penalty_box(x_end, y_end)].copy()
    return filter_passes_by_special_type(work, key)


def special_pass_count_key(filter_key: str) -> str:
    return f"special_{filter_key}"


def special_pass_per_game_key(filter_key: str) -> str:
    return f"special_{filter_key}_p90"


THREAT_ZONE_FILTER_KEYS: tuple[str, ...] = ("final_third", "in_box", "from_deep")


def threat_zone_count_key(filter_key: str) -> str:
    return f"threat_{filter_key}_passes"


def threat_zone_per_game_key(filter_key: str) -> str:
    return f"threat_{filter_key}_p90"


def _completed_pass_frame(passes: pd.DataFrame) -> pd.DataFrame:
    if passes is None or passes.empty:
        return pd.DataFrame()
    mask = passes["is_won"] & passes["has_end"] if "is_won" in passes.columns else passes["has_end"]
    return passes[mask].copy()


def compute_special_pass_masks(scored: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks for each special-pass category on completed passes."""
    n = len(scored)
    empty = np.zeros(n, dtype=bool)
    if scored is None or scored.empty:
        return {key: empty.copy() for key in SPECIAL_PASS_MAP_FILTER_KEYS}

    x_start = scored["x_start"].to_numpy(dtype=float)
    y_start = scored["y_start"].to_numpy(dtype=float)
    x_end = scored["x_end"].to_numpy(dtype=float)
    y_end = scored["y_end"].to_numpy(dtype=float)
    dist = scored["pass_distance"].to_numpy(dtype=float)
    dx = x_end - x_start
    dy = y_end - y_start

    start_zone = _zone_x(x_start)
    end_zone = _zone_x(x_end)
    long_pass = _is_long_pass(scored, dist)
    lateral_start = _is_lateral_corridor(y_start)
    lateral_end = _is_lateral_corridor(y_end)
    in_box = _in_penalty_box(x_end, y_end)

    return {
        "progressive": pe._progressive_wyscout_vec(x_start, y_start, x_end, y_end),
        "diagonal_long": (
            long_pass
            & (x_start <= DEF_X_MAX)
            & (x_end >= FINAL_X_MIN)
            & lateral_end
            & _is_diagonal_long_pass(y_start, y_end)
        ),
        "line_break": (
            _line_break_origin_ok(y_start)
            & _line_break_destination_ok(y_end)
            & (x_end > x_start)
            & _line_break_distance_ok(x_start, dist)
            & _is_forward_angle(dx, dy, max_angle_deg=LINE_BREAK_FORWARD_ANGLE_DEG)
        ),
        "inversion": long_pass & _is_left_right_inversion(y_start, y_end),
        "cross": (
            lateral_start
            & (x_start >= FINAL_X_MIN)
            & (x_start < CROSS_MAX_START_X)
            & in_box
            & (dist >= CROSS_DIST_MIN_M)
            & (np.abs(dy) >= CROSS_LATERAL_DELTA_MIN_M)
        ),
        "from_deep": (
            (start_zone == "def")
            & (end_zone == "att")
            & (dist >= FROM_DEEP_DIST_MIN_M)
        ),
        "final_third": start_zone == "att",
        "in_box": in_box,
    }


def filter_passes_by_special_type(passes: pd.DataFrame, filter_key: str) -> pd.DataFrame:
    """Return completed passes matching a special-pass map filter."""
    work = _completed_pass_frame(passes)
    if work.empty:
        return work
    key = str(filter_key or "").strip()
    if key not in SPECIAL_PASS_MAP_FILTER_KEYS:
        return work.iloc[0:0].copy()
    masks = compute_special_pass_masks(work)
    return work.loc[masks[key]].copy()


def special_pass_map_label(filter_key: str) -> str:
    return SPECIAL_PASS_MAP_FILTER_LABELS.get(str(filter_key), str(filter_key))


def _sum_xp(mask: np.ndarray, xp: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(xp[mask].sum())


XP_ROUND_SERIES_KEY = "xp_round_series"


def _opponent_array(scored: pd.DataFrame) -> np.ndarray:
    needed = {"home_team", "away_team", "team"}
    if not needed.issubset(scored.columns):
        return np.full(len(scored), "", dtype=object)
    team = scored["team"].astype(str).to_numpy()
    home = scored["home_team"].astype(str).to_numpy()
    away = scored["away_team"].astype(str).to_numpy()
    return np.where(team == home, away, home)


def round_production_series(scored: pd.DataFrame) -> tuple[dict[str, float | int | str], ...]:
    """Per-match xP and I.P. production ordered by match date, for the profile chart."""
    if scored is None or scored.empty or "event_id" not in scored.columns:
        return ()
    work = pd.DataFrame({
        "event_id": scored["event_id"].astype(str).to_numpy(),
        "xp": scored[XP_COL].to_numpy(dtype=float),
        "impact": (
            scored[THREAT_COL].to_numpy(dtype=bool)
            if THREAT_COL in scored.columns
            else np.zeros(len(scored), dtype=bool)
        ),
        "date": (
            scored["match_date"].astype(str).to_numpy()
            if "match_date" in scored.columns
            else np.full(len(scored), "", dtype=object)
        ),
        "opponent": _opponent_array(scored),
    })
    grouped = (
        work.groupby("event_id", sort=False)
        .agg(
            xp=("xp", "sum"),
            impact=("impact", "sum"),
            passes=("xp", "size"),
            date=("date", "first"),
            opponent=("opponent", "first"),
        )
        .sort_values(["date", "event_id"], kind="stable")
    )
    return tuple(
        {
            "round": index,
            "date": str(row.date),
            "opponent": str(row.opponent),
            "xp": round(float(row.xp), 3),
            "impact": int(row.impact),
            "passes": int(row.passes),
        }
        for index, row in enumerate(grouped.itertuples(), start=1)
    )


def compute_extended_xp_stats(grp: pd.DataFrame) -> dict[str, float | int]:
    """Compute full xP stat bundle for one player's season passes."""
    import xp_engine as xe

    base = xe.compute_player_xp_metrics(grp)
    if not base:
        return {}

    scored = grp[grp["is_won"] & grp["has_end"]].copy()
    if scored.empty or XP_COL not in scored.columns:
        return base

    xp = scored[XP_COL].to_numpy(dtype=float)
    n = len(scored)
    threat = (
        scored[THREAT_COL].to_numpy(dtype=bool)
        if THREAT_COL in scored.columns
        else np.zeros(n, dtype=bool)
    )
    if "progress_ratio" not in scored.columns:
        scored["progress_ratio"] = xse._progress_ratio_series(scored)

    xp_total = float(xp.sum())
    masks = compute_special_pass_masks(scored)

    out: dict[str, float | int] = dict(base)
    for sp_key in SPECIAL_PASS_COUNT_KEYS:
        out[special_pass_count_key(sp_key)] = int(masks[sp_key].sum())
    final_third_count = int(masks["final_third"].sum())
    out.update({
        "xp_diagonal_long_total": _sum_xp(masks["diagonal_long"], xp),
        "xp_line_break_total": _sum_xp(masks["line_break"], xp),
        "xp_inversion_total": _sum_xp(masks["inversion"], xp),
        "xp_cross_total": _sum_xp(masks["cross"], xp),
        "xp_final_third_share": _sum_xp(masks["final_third"], xp) / xp_total if xp_total > 0 else 0.0,
        "xp_m4_per_pass_final_third": (
            float(xp[masks["final_third"]].mean()) if final_third_count else 0.0
        ),
        "passes_final_third": final_third_count,
        "xp_box_share": _sum_xp(masks["in_box"], xp) / xp_total if xp_total > 0 else 0.0,
        "xp_from_deep": _sum_xp(masks["from_deep"], xp),
        "xp_from_deep_share": _sum_xp(masks["from_deep"], xp) / xp_total if xp_total > 0 else 0.0,
        "xp_max_pass": float(xp.max()) if n else 0.0,
        "xp_pass_std": float(xp.std()) if n > 1 else 0.0,
        "xp_pass_cv": float(xp.std() / xp.mean()) if n > 1 and xp.mean() > 0 else 0.0,
    })
    for zone_key in THREAT_ZONE_FILTER_KEYS:
        out[threat_zone_count_key(zone_key)] = int((masks[zone_key] & threat).sum())

    if RESIDUAL_COL in scored.columns:
        residual = scored[RESIDUAL_COL].to_numpy(dtype=float)
        n = len(residual)
        if n:
            out["xp_residual_positive"] = float(np.maximum(residual, 0.0).sum()) / n
            out["xp_residual_negative"] = float(np.minimum(residual, 0.0).sum()) / n
            out["xp_residual_mean"] = float(residual.mean())
            out["xp_residual_median"] = float(np.median(residual))
        else:
            out["xp_residual_positive"] = 0.0
            out["xp_residual_negative"] = 0.0
            out["xp_residual_mean"] = 0.0
            out["xp_residual_median"] = 0.0
        out["xp_surprise_rate"] = float((residual > 0).mean())
        p75 = float(np.quantile(xp, 0.75)) if n else 0.0
        high_xp = xp >= p75
        out["xp_threat_conversion"] = float(threat.sum() / high_xp.sum()) if high_xp.any() else 0.0
        if threat.any():
            out["xp_threat_mean_xp"] = float(xp[threat].mean())
            out["xp_threat_mean_residual"] = float(residual[threat].mean())
        else:
            out["xp_threat_mean_xp"] = 0.0
            out["xp_threat_mean_residual"] = 0.0
    else:
        out["xp_residual_positive"] = 0.0
        out["xp_residual_negative"] = 0.0
        out["xp_residual_mean"] = 0.0
        out["xp_residual_median"] = 0.0
        out["xp_surprise_rate"] = 0.0
        out["xp_threat_conversion"] = 0.0
        out["xp_threat_mean_xp"] = 0.0
        out["xp_threat_mean_residual"] = 0.0

    if "event_id" in scored.columns:
        game_xp = scored.groupby("event_id")[XP_COL].sum()
        out["xp_game_mean"] = float(game_xp.mean()) if len(game_xp) else 0.0
        out["xp_game_std"] = float(game_xp.std()) if len(game_xp) > 1 else 0.0
        med = float(game_xp.median()) if len(game_xp) else 0.0
        out["xp_games_above_median_pct"] = float((game_xp > med).mean()) if len(game_xp) else 0.0
        out[XP_ROUND_SERIES_KEY] = round_production_series(scored)
    else:
        out["xp_game_mean"] = 0.0
        out["xp_game_std"] = 0.0
        out["xp_games_above_median_pct"] = 0.0
        out[XP_ROUND_SERIES_KEY] = ()

    return out


def attach_regular_pass_stats_from_enriched(
    metrics: dict[str, float | int],
    enriched_passes: pd.DataFrame,
    minutes: float | None,
) -> None:
    """Attach regular volume stats when passes are already enriched."""
    if enriched_passes is None or enriched_passes.empty:
        attach_regular_pass_stats(metrics, pd.DataFrame(), minutes)
        return

    pass_metrics = pe.compute_player_metrics(enriched_passes, {"minutes": minutes})
    mins = float(minutes or 0)
    factor = 90.0 / mins if mins > 0 else 0.0

    for key in ("passes_total", "long_balls", "passes_to_box", "key_passes"):
        metrics[key] = round(float(pass_metrics.get(key, 0) or 0) * factor, 3)

    metrics["progressive_passes"] = float(pass_metrics.get("progressive_passes_p90", 0) or 0)
    metrics["final_third_passes"] = float(pass_metrics.get("final_third_passes_p90", 0) or 0)
    metrics["pass_completion_pct"] = pass_metrics.get("pass_completion_pct", 0.0)
    metrics["long_ball_completion_pct"] = pass_metrics.get("long_ball_completion_pct", 0.0)


def attach_regular_pass_stats(
    metrics: dict[str, float | int],
    raw_pass_frame: pd.DataFrame,
    minutes: float | None,
) -> None:
    """Attach regular volume stats (per 90) aligned with passes_engine definitions."""
    if raw_pass_frame is None or raw_pass_frame.empty:
        for key in (
            "passes_total",
            "long_balls",
            "progressive_passes",
            "final_third_passes",
            "passes_to_box",
            "key_passes",
        ):
            metrics.setdefault(key, 0.0)
        metrics.setdefault("pass_completion_pct", 0.0)
        metrics.setdefault("long_ball_completion_pct", 0.0)
        return

    enriched = pe._enrich_passes(raw_pass_frame)
    pass_metrics = pe.compute_player_metrics(enriched, {"minutes": minutes})
    mins = float(minutes or 0)
    factor = 90.0 / mins if mins > 0 else 0.0

    for key in ("passes_total", "long_balls", "passes_to_box", "key_passes"):
        metrics[key] = round(float(pass_metrics.get(key, 0) or 0) * factor, 3)

    metrics["progressive_passes"] = float(pass_metrics.get("progressive_passes_p90", 0) or 0)
    metrics["final_third_passes"] = float(pass_metrics.get("final_third_passes_p90", 0) or 0)
    metrics["pass_completion_pct"] = pass_metrics.get("pass_completion_pct", 0.0)
    metrics["long_ball_completion_pct"] = pass_metrics.get("long_ball_completion_pct", 0.0)


def apply_per90_metrics(metrics: dict[str, float | int], minutes: float | None) -> None:
    """Add per-90 variants in place."""
    if not minutes or float(minutes) <= 0:
        metrics["xp_per_90"] = 0.0
        metrics["threat_passes_p90"] = 0.0
        metrics["impact_passes_p90"] = 0.0
        for sp_key in SPECIAL_PASS_COUNT_KEYS:
            metrics[special_pass_per_game_key(sp_key)] = 0.0
        for zone_key in THREAT_ZONE_FILTER_KEYS:
            metrics[threat_zone_per_game_key(zone_key)] = 0.0
        return
    mins_f = float(minutes)
    factor = 90.0 / mins_f
    metrics["xp_per_90"] = float(metrics.get("xp_m4_total", 0.0)) * factor
    threat_count = int(metrics.get("xp_m4_threat_passes", 0))
    metrics["threat_passes_p90"] = float(threat_count) * factor
    metrics["impact_passes_p90"] = metrics["threat_passes_p90"]
    metrics["xp_m4_threat_passes_p90"] = float(metrics.get("xp_m4_threat_xp_total", 0.0)) * factor
    for band in BANDS:
        band_threats = int(metrics.get(f"xp_m4_threat_{band}", 0))
        metrics[f"xp_m4_threat_{band}_p90"] = float(band_threats) * factor
    for sp_key in SPECIAL_PASS_COUNT_KEYS:
        count = int(metrics.get(special_pass_count_key(sp_key), 0))
        metrics[special_pass_per_game_key(sp_key)] = float(count) * factor
    for zone_key in THREAT_ZONE_FILTER_KEYS:
        count = int(metrics.get(threat_zone_count_key(zone_key), 0))
        metrics[threat_zone_per_game_key(zone_key)] = float(count) * factor


# (section_title, metric_keys)
XP_STATS_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Totals", (
        "xp_per_90", "xp_m4_per_pass", "xp_m4_threat_rate",
    )),
    ("xP Stats", (
        "special_diagonal_long_p90", "xp_diagonal_long_total",
        "special_line_break_p90", "xp_line_break_total",
        "special_inversion_p90", "xp_inversion_total",
        "special_cross_p90", "xp_cross_total",
        "xp_final_third_share", "threat_final_third_p90",
        "xp_box_share", "threat_in_box_p90",
        "xp_from_deep", "threat_from_deep_p90",
    )),
    ("Quality", (
        "xp_residual_median", "xp_surprise_rate",
    )),
    ("Consistency", (
        "xp_game_mean", "xp_game_std_adj_score", "xp_games_above_median_pct",
    )),
    (f"Short ({DISTANCE_BAND_LABELS['short']})", (
        "xp_m4_per_pass_short", "xp_m4_threat_rate_short",
    )),
    (f"Long ({DISTANCE_BAND_LABELS['long']})", (
        "xp_m4_per_pass_long", "xp_m4_threat_rate_long",
    )),
)

# Player Analysis passing blocks
XP_PLAYER_ANALYSIS_BLOCKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Volume", (
        "xp_per_90",
        "threat_passes_p90",
    )),
    ("Effectiveness", (
        "xp_m4_per_pass",
        "xp_m4_per_threat_pass",
        "xp_m4_threat_rate",
    )),
    ("Quality", (
        "xp_residual_median",
        "xp_surprise_rate",
    )),
    ("Consistency", (
        "xp_game_std_adj_score",
        "xp_games_above_median_pct",
    )),
)

XP_COMPOSITE_INDEX_KEYS: tuple[str, ...] = (
    "xp_builder_index",
    "xp_creator_index",
    "xp_progressor_index",
    "xp_finisher_pass_index",
    "xp_quality_index",
    "xp_consistency_index",
)

XP_ARCHETYPE_RADAR_KEYS: tuple[str, ...] = (
    "xp_archetype_builder_display",
    "xp_archetype_creator_display",
    "xp_archetype_progressor_display",
    "xp_archetype_finisher_display",
)

XP_ARCHETYPE_RADAR_LABELS: dict[str, str] = {
    "xp_archetype_builder_display": "Builder",
    "xp_archetype_creator_display": "Creator",
    "xp_archetype_progressor_display": "Progressor",
    "xp_archetype_finisher_display": "Finisher-pass",
}

# The two pillars rendered as gradient bars in the xP Profile (grade drivers).
XP_PROFILE_BAR_KEYS: tuple[str, ...] = (
    "xp_activity_display",
    "xp_edge_display",
)

# Axes used only to classify the xP profile archetype (not all are rendered).
XP_ARCHETYPE_AXIS_KEYS: tuple[str, ...] = (
    "xp_activity_display",
    "xp_edge_display",
    "xp_quality_display",
    "xp_consistency_display",
)

XP_PROFILE_BAR_LABELS: dict[str, str] = {
    "xp_activity_display": "Productivity",
    "xp_edge_display": "Effectiveness",
    "xp_quality_display": "Quality",
    "xp_consistency_display": "Consistency",
}

XP_PROFILE_BAR_ICONS: dict[str, str] = {
    "xp_activity_display": "fa-chart-simple",
    "xp_edge_display": "fa-bolt",
    "xp_quality_display": "fa-arrow-trend-up",
    "xp_consistency_display": "fa-wave-square",
}

XP_PROFILE_BAR_METRICS: dict[str, tuple[str, ...]] = {
    "xp_activity_display": ("xp_per_90",),
    "xp_edge_display": ("xp_m4_per_pass",),
    "xp_quality_display": ("xp_residual_median",),
    "xp_consistency_display": ("xp_game_std_adj_score",),
}

# Metrics that carry their own rank-based mini-bar in the comparison view.
XP_PROFILE_SUBMETRICS: tuple[str, ...] = (
    "xp_per_90",
    "threat_passes_p90",
    "xp_m4_per_pass",
    "xp_m4_per_threat_pass",
    "xp_residual_median",
)

# Secondary indices shown as coloured status boxes below the regular stats.
# (index_key, label, metrics, invert_metrics)
XP_INDEX_SPECS: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("xp_idx_consistency", "Consistency", ("xp_game_std_adj_score",), ()),
)

XP_INDEX_TIER_LABELS: dict[str, str] = {
    "below": "Below average",
    "mid": "Average",
    "above": "Above average",
}

XP_INDEX_TOOLTIPS: dict[str, str] = {
    "xp_idx_consistency": "Game-to-game xP stability — how consistent the player's delivery is.",
}

# Icons for the index rows (tier indices + badges) shown in the xP Profile card.
XP_INDEX_ICONS: dict[str, str] = {
    "xp_idx_consistency": "fa-wave-square",
}

# Achievement badges — earned when ranked in the top N among eligible peers on the
# composite of the badge metrics. (badge_key, label, metrics, icon)
XP_BADGE_TOP_SIZE = 25
XP_BADGE_SPECS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("xp_badge_threat", IMPACT_PASS_ABBR, ("threat_passes_p90", "xp_m4_per_threat_pass"), "fa-crosshairs"),
)

XP_BADGE_TOOLTIPS: dict[str, str] = {
    "xp_badge_threat": (
        f"Top 25 in campo ofensivo or campo defensivo in {IMPACT_PASS_ABBR} per game "
        f"and xP per {IMPACT_PASS_ABBR}."
    ),
}

# Player Analysis compare panel.
# Highlighted comparison metrics (more emphasis, with mini-bars).
XP_COMPARE_HIGHLIGHT_KEYS: tuple[str, ...] = (
    "xp_per_90",
    "threat_passes_p90",
    "xp_m4_per_pass",
    "xp_m4_per_threat_pass",
)
XP_COMPARE_HIGHLIGHT_LABELS: dict[str, str] = {
    "xp_per_90": "xP (Per game)",
    "threat_passes_p90": f"{IMPACT_PASS_ABBR} (Per game)",
    "xp_m4_per_pass": "xP/Pass",
    "xp_m4_per_threat_pass": f"xP/{IMPACT_PASS_ABBR}",
}
XP_COMPARE_HIGHLIGHT_TOOLTIPS: dict[str, str] = {
    "xp_per_90": (
        "xP volume normalized per 90 minutes — how much offensive value the player "
        "produces per game."
    ),
    "threat_passes_p90": (
        f"{IMPACT_PASS_ABBR} produced per 90 minutes — passes with high destination value, "
        "positive surprise vs expected, and meaningful forward progress."
    ),
    "xp_m4_per_pass": (
        "Average xP per pass — measures the efficiency of each delivery, "
        "independent of volume."
    ),
    "xp_m4_per_threat_pass": (
        f"Average xP on {IMPACT_PASS_ABBR} — the quality of high-value passes produced."
    ),
}
# Secondary key metrics (less emphasis) = traditional volume stats.
XP_COMPARE_METRIC_KEYS: tuple[str, ...] = (
    "passes_total",
    "progressive_passes",
    "final_third_passes",
    "key_passes",
)
XP_COMPARE_METRIC_LABELS: dict[str, str] = {
    "passes_total": "Passes",
    "progressive_passes": "Progressive",
    "final_third_passes": "Final Third",
    "key_passes": "Key Passes",
}
XP_COMPARE_METRIC_TOOLTIPS: dict[str, str] = {
    "passes_total": "Passes attempted per game (p90).",
    "progressive_passes": (
        "Progressive passes completed per game (p90) — Wyscout criterion: "
        "advance ≥ 10 m toward goal, or ≥ 5 m inside the final third."
    ),
    "final_third_passes": (
        "Passes completed into the final third (x ≥ 80 m) per game (p90) — "
        "volume of deliveries in the creation zone."
    ),
    "key_passes": (
        "Passes leading to a shot per game (p90) — deliveries that result in a shot."
    ),
}

XP_PROFILE_BAR_TOOLTIPS: dict[str, str] = {
    "xp_activity_display": "How much xP the player generates per game — passing volume times value.",
    "xp_edge_display": "Average xP per pass — how valuable each delivery is, regardless of volume.",
    "xp_quality_display": "Median xP above the model's expectation — value that comes from surprise.",
    "xp_consistency_display": "How stable game-to-game xP delivery is.",
}

XP_PROFILE_ARCHETYPE_KEYS: tuple[str, ...] = (
    "elite",
    "criativo",
    "seguranca",
    "impacto",
    "limitado",
    "regular",
)

XP_PROFILE_ARCHETYPE_LABELS: dict[str, str] = {
    "elite": "Elite",
    "criativo": "Creative",
    "seguranca": "Safety",
    "impacto": "Impact",
    "limitado": "Limited",
    "regular": "Regular",
}

XP_PROFILE_ARCHETYPE_DESCRIPTIONS: dict[str, str] = {
    "elite": (
        "Complete profile: volume, effectiveness, quality and consistency above the "
        "position median."
    ),
    "criativo": (
        "Selective specialist: effectiveness and quality above the median, with volume and "
        "consistency below."
    ),
    "seguranca": (
        "Safety profile: volume and consistency above the median — reliable, understated "
        "and stable in passing."
    ),
    "impacto": (
        "High-impact producer: volume, effectiveness and quality above the median, with "
        "more volatile consistency."
    ),
    "limitado": (
        "Low relative impact at the position: three or more xP Profile axes below the "
        "group median."
    ),
    "regular": (
        "Balanced profile at the position, without clearly fitting the other archetypes."
    ),
}

XP_PROFILE_ARCHETYPE_STYLES: dict[str, str] = {
    "elite": "elite",
    "criativo": "attack",
    "seguranca": "build",
    "impacto": "impacto",
    "limitado": "reference",
    "regular": "link",
}

XP_PROFILE_ARCHETYPE_ICONS: dict[str, str] = {
    "elite": "fa-crown",
    "criativo": "fa-wand-magic-sparkles",
    "seguranca": "fa-shield-halved",
    "impacto": "fa-bolt",
    "limitado": "fa-arrow-trend-down",
    "regular": "fa-equals",
}

XP_PROFILE_ARCHETYPE_FILTER_ALL = ""

ACTIVITY_METRICS: tuple[str, ...] = ("xp_per_90",)
EDGE_METRICS: tuple[str, ...] = ("xp_m4_per_pass",)

# Grade = weighted arithmetic mean of Productivity (xP/game) and Effectiveness (xP/pass).
XP_PASS_RATING_FEATURE_WEIGHTS: dict[str, float] = {
    "xp_per_90": 0.50,
    "xp_m4_per_pass": 0.50,
}
XP_PASS_RATING_FEATURES: tuple[str, ...] = tuple(XP_PASS_RATING_FEATURE_WEIGHTS)

# Weight each rendered pillar carries in the composite grade.
XP_PROFILE_BAR_WEIGHTS: dict[str, float] = {
    display_key: XP_PASS_RATING_FEATURE_WEIGHTS[metrics[0]]
    for display_key, metrics in XP_PROFILE_BAR_METRICS.items()
    if metrics and metrics[0] in XP_PASS_RATING_FEATURE_WEIGHTS
}
XP_PASS_RATING_TANH_SCALE = 1.25
XP_PASS_RATING_TANH_AMPLITUDE = 1.15
XP_PASS_RATING_PERCENTILE_BANDS: tuple[tuple[float, float, float], ...] = (
    # (max_rank_pct, score_at_band_start, score_at_band_end) — rank 1 = lowest pct.
    (0.10, 9.0, 8.0),   # top 10%
    (0.30, 8.0, 7.0),   # 10–30%
    (1.00, 7.0, 4.5),   # rest
)
XP_PASS_RATING_CONFIDENCE_WEIGHT = 0.4
XP_PASS_RATING_CONFIDENCE_PASS_WEIGHT = 0.5
XP_PASS_RATING_CONFIDENCE_MINUTES_WEIGHT = 0.5

BUILDER_BASE_METRICS: tuple[str, ...] = (
    "xp_line_break_total",
    "special_line_break_p90",
    "xp_m4_per_pass_short",
    "xp_m4_threat_rate_short",
)
BUILDER_FB_METRICS: tuple[str, ...] = (
    "xp_inversion_total",
    "special_inversion_p90",
)
CREATOR_METRICS: tuple[str, ...] = (
    "xp_final_third_share",
    "threat_final_third_p90",
    "xp_m4_per_pass_final_third",
)
PROGRESSOR_METRICS: tuple[str, ...] = (
    "xp_diagonal_long_total",
    "special_diagonal_long_p90",
    "xp_from_deep",
    "xp_m4_per_pass_long",
    "xp_m4_threat_rate_long",
)
FINISHER_METRICS: tuple[str, ...] = (
    "xp_box_share",
    "threat_in_box_p90",
    "xp_cross_total",
    "special_cross_p90",
)
QUALITY_METRICS: tuple[str, ...] = ("xp_residual_median",)
CONSISTENCY_METRICS: tuple[str, ...] = ("xp_game_std_adj_score",)
CONSISTENCY_INVERT_METRICS: tuple[str, ...] = ()

XP_PROFILE_BAR_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("xp_activity_index", "xp_activity_display", ACTIVITY_METRICS),
    ("xp_edge_index", "xp_edge_display", EDGE_METRICS),
    ("xp_quality_index", "xp_quality_display", QUALITY_METRICS),
    ("xp_consistency_index", "xp_consistency_display", CONSISTENCY_METRICS),
)


def iter_xp_player_analysis_blocks() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Yield (title, keys) for every Player Analysis passing block."""
    return XP_PLAYER_ANALYSIS_BLOCKS


def p20_pass_thresholds_by_group(
    players: list[dict],
    passes_col: str,
    *,
    percentile: int = DISTANCE_INDEX_MIN_PASS_PERCENTILE,
) -> dict[str, float]:
    """Minimum passes at the position-group percentile (default P30)."""
    pools: dict[str, list[float]] = {}
    for player in players:
        group = _metric_rank_pool_key(player)
        pools.setdefault(group, []).append(float(player.get(passes_col) or 0.0))
    return {
        group: float(np.percentile(counts, percentile)) if counts else 0.0
        for group, counts in pools.items()
    }


def iter_xp_stats_sections() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Yield (title, keys) for every Stats tab section."""
    for entry in XP_STATS_SECTIONS:
        if len(entry) == 2:
            title, keys = entry
            yield title, keys
        else:
            title, keys, _summary = entry
            yield title, keys


XP_STATS_LABELS: dict[str, str] = {
    "xp_per_90": "xP (Per game)",
    "threat_passes_p90": f"{IMPACT_PASS_ABBR} (Per game)",
    "xp_m4_total": "xP Total",
    "xp_m4_threat_passes": f"{IMPACT_PASS_ABBR} Total",
    "xp_m4_threat_passes_p90": f"xP {IMPACT_PASS_ABBR} (Per game)",
    "xp_m4_per_pass": "xP/Pass",
    "xp_m4_per_threat_pass": f"xP/{IMPACT_PASS_ABBR}",
    "xp_m4_threat_rate": f"% {IMPACT_PASS_ABBR}",
    "xp_m4_per_pass_short": "xP/Pass",
    "xp_m4_per_pass_long": "xP/Pass",
    "xp_m4_threat_rate_short": f"% {IMPACT_PASS_ABBR}",
    "xp_m4_threat_rate_long": f"% {IMPACT_PASS_ABBR}",
    "xp_m4_threat_short_p90": f"xP {IMPACT_PASS_ABBR} (Per game)",
    "xp_m4_threat_long_p90": f"xP {IMPACT_PASS_ABBR} (Per game)",
    "passes_short": "Passes in band",
    "passes_long": "Passes in band",
    "xp_m4_total_short": "xP Total (Short)",
    "xp_m4_threat_short_p90": f"{IMPACT_PASS_ABBR} p/game (Short)",
    "xp_m4_total_long": "xP Total (Long)",
    "xp_m4_threat_long_p90": f"{IMPACT_PASS_ABBR} p/game (Long)",
    "xp_diagonal_long_total": "Long Diagonal (xP)",
    "special_diagonal_long_p90": "Long Diagonal (Per game)",
    "xp_line_break_total": "Line Break (xP)",
    "special_line_break_p90": "Line Break (Per game)",
    "xp_inversion_total": "Inversions (xP)",
    "special_inversion_p90": "Inversions (Per game)",
    "xp_cross_total": "Crosses (xP)",
    "special_cross_p90": "Crosses (Per game)",
    "xp_final_third_share": "%xP - Final Third",
    "threat_final_third_p90": f"{IMPACT_PASS_ABBR} - Final Third",
    "xp_box_share": "%xP - Box",
    "threat_in_box_p90": f"{IMPACT_PASS_ABBR} - Box",
    "xp_from_deep": "xP Deep",
    "threat_from_deep_p90": f"{IMPACT_PASS_ABBR} - Deep",
    "special_final_third_p90": "Final third (Per game)",
    "special_in_box_p90": "In box (Per game)",
    "special_from_deep_p90": "From deep (Per game)",
    "xp_residual_mean": "Mean residual/Pass",
    "xp_residual_median": "Median residual/Pass",
    "xp_residual_positive": "xP above expected/Pass",
    "xp_residual_negative": "xP below expected/Pass",
    "xp_surprise_rate": "Surprise Rate",
    "xp_threat_conversion": f"{IMPACT_PASS_ABBR} conversion",
    "xp_threat_mean_xp": f"Mean {IMPACT_PASS_ABBR} xP",
    "xp_threat_mean_residual": f"Mean {IMPACT_PASS_ABBR} residual",
    "xp_m4_p90": "xP P90 (pass)",
    "xp_max_pass": "Max single-pass xP",
    "xp_game_mean": "Mean xP (Per game)",
    "xp_game_std": "xP Std Dev",
    "xp_game_std_adj": "Adjusted xP Std Dev",
    "xp_game_std_adj_score": "Adjusted stability",
    "xp_pass_cv": "xP CV (passes)",
    "xp_games_above_median_pct": "% Games above median",
    "xp_pass_std": "xP Std Dev (passes)",
    "xp_builder_index": "Builder",
    "xp_creator_index": "Creator",
    "xp_progressor_index": "Progressor",
    "xp_finisher_pass_index": "Finisher-pass",
    "xp_quality_index": "Quality",
    "xp_consistency_index": "Consistency",
    "xp_m4_per_pass_final_third": "xP/Pass Final Third",
    "xp_archetype_creator_display": "Creator",
    "xp_archetype_progressor_display": "Progressor",
    "xp_archetype_finisher_display": "Finisher-pass",
    "xp_quality_display": "Quality",
    "xp_consistency_display": "Consistency",
}

XP_PA_LABELS: dict[str, str] = {
    "xp_per_90": "xP / game",
    "threat_passes_p90": f"{IMPACT_PASS_ABBR} / game",
    "xp_m4_per_pass": "xP / pass",
    "xp_m4_per_threat_pass": f"xP / {IMPACT_PASS_ABBR}",
    "xp_m4_threat_rate": f"% {IMPACT_PASS_ABBR}",
    "xp_residual_median": "Median residual",
    "xp_surprise_rate": "% above expected",
    "xp_game_std_adj_score": "Stability",
    "xp_games_above_median_pct": "% strong games",
}

XP_PA_TOOLTIPS: dict[str, str] = {
    "xp_per_90": "xP volume from passing, normalized per 90 minutes.",
    "threat_passes_p90": (
        f"{IMPACT_PASS_ABBR} per game — composite score (45% xP + 35% residual + 20% progress) "
        "in the top 7.5% for distance band, with forward progress ≥ P65."
    ),
    "xp_m4_per_pass": "Average xP per pass — measures the efficiency of each delivery.",
    "xp_m4_per_threat_pass": f"Average xP on {IMPACT_PASS_ABBR} (surprise + high value for distance).",
    "xp_m4_threat_rate": f"Share of passes classified as {IMPACT_PASS_ABBR}.",
    "xp_residual_median": (
        "Median residual (actual xP − expected) per pass, ×100. Positive values mean passes "
        "beat the model."
    ),
    "xp_surprise_rate": "Share of passes with a positive residual — passes that beat expectations.",
    "xp_game_std_adj_score": "Delivery stability across games, adjusted for the player's average xP level.",
    "xp_games_above_median_pct": "Share of games where the player's xP was above their own median.",
}

def iter_stats_metric_options() -> tuple[tuple[str, str], ...]:
    """Ordered (metric_key, label) pairs for every Stats tab metric."""
    seen: dict[str, str] = {}
    for _title, keys in iter_xp_stats_sections():
        for key in keys:
            if key not in seen:
                seen[key] = stats_metric_label(key)
    return tuple(seen.items())


# Dispersão (scatter) — analyst-facing metrics split into two stat types.
# Regular Stats: card stats minus the completion (% acerto) ones.
SCATTER_REGULAR_METRIC_OPTIONS: tuple[tuple[str, str], ...] = (
    ("passes_total", "Passes / game"),
    ("long_balls", "Long passes / game"),
    ("progressive_passes", "Progressive passes / game"),
    ("final_third_passes", "Passes into final third / game"),
    ("passes_to_box", "Passes into box / game"),
    ("key_passes", "Key passes / game"),
    ("pass_mean_distance", "Mean pass distance"),
)
# Special Stat: special passes plus the xP metrics.
SCATTER_SPECIAL_METRIC_OPTIONS: tuple[tuple[str, str], ...] = (
    ("xp_per_90", "xP / game"),
    ("threat_passes_p90", f"{IMPACT_PASS_ABBR} / game"),
    ("xp_m4_per_pass", "xP / pass"),
    ("xp_m4_per_threat_pass", f"xP / {IMPACT_PASS_ABBR}"),
    ("xp_game_std_adj_score", "Stability"),
)
SCATTER_STAT_TYPE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("regular", "Regular Stats"),
    ("special", "xP Stats"),
)
SCATTER_METRIC_OPTIONS: tuple[tuple[str, str], ...] = (
    *SCATTER_REGULAR_METRIC_OPTIONS,
    *SCATTER_SPECIAL_METRIC_OPTIONS,
)
SCATTER_METRIC_LABELS: dict[str, str] = dict(SCATTER_METRIC_OPTIONS)
# Scatter axes that come from the special-pass family (flagged in the UI).
SCATTER_SPECIAL_METRIC_KEYS: frozenset[str] = frozenset()


def iter_scatter_metric_options() -> tuple[tuple[str, str], ...]:
    return SCATTER_METRIC_OPTIONS


def scatter_stat_type_options() -> tuple[tuple[str, str], ...]:
    return SCATTER_STAT_TYPE_OPTIONS


def scatter_metric_options_for_type(stat_type: str) -> tuple[tuple[str, str], ...]:
    if str(stat_type) == "special":
        return SCATTER_SPECIAL_METRIC_OPTIONS
    return SCATTER_REGULAR_METRIC_OPTIONS


def scatter_metric_label(key: str) -> str:
    return SCATTER_METRIC_LABELS.get(key, stats_metric_label(key))


def is_scatter_special_metric(key: str) -> bool:
    return str(key) in SCATTER_SPECIAL_METRIC_KEYS


SCATTER_BAND_OPTIONS: tuple[tuple[str, str], ...] = (
    ("total", "Total"),
    ("short", f"Short ({DISTANCE_BAND_LABELS['short']})"),
    ("long", f"Long ({DISTANCE_BAND_LABELS['long']})"),
)
SCATTER_BANDED_BASE_KEYS: frozenset[str] = frozenset({
    "xp_m4_total",
    "xp_m4_per_pass",
    "xp_m4_threat_passes_p90",
    "xp_m4_threat_rate",
})
SCATTER_EXTRA_BASE_KEYS: tuple[tuple[str, str], ...] = (
    ("xp_m4_total", "xP Total"),
    ("xp_m4_threat_passes_p90", f"xP {IMPACT_PASS_ABBR} (Per game)"),
)


def _is_scatter_band_variant_key(key: str) -> bool:
    if key.endswith(("_short", "_long")):
        return True
    return key in {
        "xp_m4_threat_short_p90",
        "xp_m4_threat_long_p90",
    }


def iter_scatter_base_metric_options() -> tuple[tuple[str, str], ...]:
    """Base Stats metrics for scatter axes (band chosen separately)."""
    seen: dict[str, str] = {}
    for key, label in iter_stats_metric_options():
        if _is_scatter_band_variant_key(key):
            continue
        seen[key] = label
    for key, label in SCATTER_EXTRA_BASE_KEYS:
        seen.setdefault(key, label)
    return tuple(seen.items())


def resolve_scatter_metric_key(base_key: str, band: str) -> str:
    """Map base metric + distance band to the player-profile column key."""
    if band == "total" or base_key not in SCATTER_BANDED_BASE_KEYS:
        return base_key
    if base_key == "xp_m4_threat_passes_p90":
        return f"xp_m4_threat_{band}_p90"
    return f"{base_key}_{band}"


def scatter_axis_label(base_key: str, band: str) -> str:
    base_label = stats_metric_label(base_key)
    if band == "total" or base_key not in SCATTER_BANDED_BASE_KEYS:
        return base_label
    band_label = DISTANCE_BAND_LABELS.get(band, band)
    return f"{base_label} · {band_label}"


THREAT_PASS_MAP_FILTERS: tuple[tuple[str, str], ...] = (
    ("all", "Total"),
    ("short", f"Short ({DISTANCE_BAND_LABELS['short']})"),
    ("long", f"Long ({DISTANCE_BAND_LABELS['long']})"),
)
THREAT_PASS_MAP_FILTER_KEYS: tuple[str, ...] = tuple(key for key, _label in THREAT_PASS_MAP_FILTERS)
THREAT_PASS_MAP_FILTER_LABELS: dict[str, str] = dict(THREAT_PASS_MAP_FILTERS)


def threat_pass_map_label(filter_key: str) -> str:
    return THREAT_PASS_MAP_FILTER_LABELS.get(str(filter_key), str(filter_key))


def filter_passes_by_threat_type(passes: pd.DataFrame, filter_key: str) -> pd.DataFrame:
    """Return completed xP threat passes, optionally filtered by distance band."""
    work = _completed_pass_frame(passes)
    if work.empty:
        return work
    if THREAT_COL not in work.columns:
        return work.iloc[0:0].copy()
    work = work[work[THREAT_COL]].copy()
    band = str(filter_key or "all").strip()
    if band in {"", "all"}:
        return work
    if band not in BANDS:
        return work.iloc[0:0].copy()
    if "distance_band" not in work.columns:
        work = work.copy()
        work["distance_band"] = xse._distance_band_series(work["pass_distance"])
    return work[work["distance_band"].astype(str) == band].copy()


XP_STATS_RANK_METRICS: tuple[str, ...] = tuple(
    dict.fromkeys(
        key
        for _title, keys in iter_xp_stats_sections()
        for key in keys
    )
)

XP_PLAYER_ANALYSIS_RANK_METRICS: tuple[str, ...] = tuple(
    dict.fromkeys(
        key
        for _title, keys in XP_PLAYER_ANALYSIS_BLOCKS
        for key in keys
    ) | dict.fromkeys(XP_COMPOSITE_INDEX_KEYS)
)

XP_REGULAR_STAT_RANK_KEYS: tuple[str, ...] = (
    "passes_total",
    "pass_completion_pct",
    "long_balls",
    "long_ball_completion_pct",
    "progressive_passes",
    "final_third_passes",
    "passes_to_box",
    "key_passes",
    "long_pass_share_pct",
    "special_line_break_p90",
    "impact_passes_p90",
)


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std())
    if std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _rank_descending(values: pd.Series) -> pd.Series:
    return values.astype(float).rank(method="min", ascending=False)


def _grade_from_rank_pct(pct: float) -> str:
    for label, cutoff in DISTANCE_INDEX_GRADES:
        if pct <= cutoff:
            return label
    return DISTANCE_INDEX_GRADES[-1][0]


def _grade_from_tier_value(value: float) -> str:
    tier = max(1.0, min(5.0, float(value)))
    if tier >= 4.5:
        return "Good"
    if tier >= 3.5:
        return "Above Average"
    if tier >= 2.5:
        return "Average"
    if tier >= 1.5:
        return "Under Average"
    return "Poor"


def _balanced_grade_from_rank_pcts(rank_pcts: list[float]) -> str:
    """Blend mean and worst metric so one outlier cannot inflate the grade."""
    if not rank_pcts:
        return "Poor"
    tier_vals = [DISTANCE_INDEX_GRADE_ORDER[_grade_from_rank_pct(pct)] for pct in rank_pcts]
    blended = (
        DISTANCE_INDEX_BALANCE_MIN_WEIGHT * min(tier_vals)
        + DISTANCE_INDEX_BALANCE_MEAN_WEIGHT * float(np.mean(tier_vals))
    )
    return _grade_from_tier_value(blended)


def _apply_volume_grade_penalty(grade: str, volume_rank_pct: float) -> str:
    tier = DISTANCE_INDEX_GRADE_ORDER.get(grade, 3)
    for cutoff, steps in DISTANCE_INDEX_VOLUME_GRADE_PENALTY_PCTS:
        if volume_rank_pct > cutoff:
            tier -= steps
    return _grade_from_tier_value(float(max(1, tier)))


def _series_or_zero(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].astype(float)
    return pd.Series(0.0, index=df.index)


def _mean_z_columns(
    df: pd.DataFrame,
    cols: tuple[str, ...],
    *,
    invert: tuple[str, ...] = (),
) -> pd.Series:
    if not cols:
        return pd.Series(0.0, index=df.index)
    parts: list[pd.Series] = []
    for col in cols:
        z = _zscore(_series_or_zero(df, col))
        if col in invert:
            z = -z
        parts.append(z)
    return sum(parts) / len(parts)


def _attach_index_display_scores(
    rows: list[dict],
    raw_key: str,
    display_key: str,
    composite: pd.Series,
) -> None:
    import passes_engine as pe

    pool_size = len(rows)
    ranks = _rank_descending(composite)
    for i, row in enumerate(rows):
        row[raw_key] = float(composite.iloc[i])
        rank = int(ranks.iloc[i])
        row[f"{raw_key}_rank_in_group"] = rank
        row[f"{raw_key}_rank_pool_in_group"] = pool_size
        row[display_key] = float(pe.rank_to_display_score(rank, pool_size))


def _attach_median_rank_display_scores(
    rows: list[dict],
    cols: tuple[str, ...],
    raw_key: str,
    display_key: str,
) -> None:
    """Rank each metric within position, take the median rank, then map to 3–9 display."""
    if not rows or not cols:
        return
    df = pd.DataFrame(rows)
    pool_size = len(rows)
    rank_frame = pd.concat(
        [_rank_descending(_series_or_zero(df, col).astype(float)) for col in cols],
        axis=1,
    )
    median_rank = rank_frame.median(axis=1).astype(float)
    composite = float(pool_size + 1) - median_rank
    _attach_index_display_scores(rows, raw_key, display_key, composite)


def _attach_game_std_adjusted(rows: list[dict]) -> None:
    """Residual of game-level xP std after regressing on game mean (within position)."""
    if not rows:
        return
    means = np.array([float(r.get("xp_game_mean") or 0.0) for r in rows], dtype=float)
    stds = np.array([float(r.get("xp_game_std") or 0.0) for r in rows], dtype=float)
    if len(rows) < 3 or float(np.std(means)) <= 1e-12:
        for row in rows:
            row["xp_game_std_adj"] = 0.0
            row["xp_game_std_adj_score"] = 0.0
        return
    slope, intercept = np.polyfit(means, stds, 1)
    adjusted = stds - (slope * means + intercept)
    for row, val in zip(rows, adjusted):
        row["xp_game_std_adj"] = float(val)
        row["xp_game_std_adj_score"] = float(-val)


def _xp_profile_axis_medians(rows: list[dict]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for key in XP_ARCHETYPE_AXIS_KEYS:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and np.isfinite(float(row[key]))
        ]
        medians[key] = float(np.median(values)) if values else 6.0
    return medians


def classify_xp_profile_archetype(
    row: dict,
    medians: dict[str, float],
) -> str:
    """Classify a player into one of six xP profile archetypes (within position)."""
    scores: dict[str, float] = {}
    for key in XP_ARCHETYPE_AXIS_KEYS:
        raw = row.get(key)
        if raw is None or not np.isfinite(float(raw)):
            return "regular"
        scores[key] = float(raw)

    below = {key: scores[key] < medians[key] for key in XP_ARCHETYPE_AXIS_KEYS}
    above = {key: scores[key] > medians[key] for key in XP_ARCHETYPE_AXIS_KEYS}

    volume = "xp_activity_display"
    effectiveness = "xp_edge_display"
    quality = "xp_quality_display"
    consistency = "xp_consistency_display"

    if above[volume] and above[effectiveness] and above[quality] and above[consistency]:
        return "elite"
    if above[effectiveness] and above[quality] and below[consistency] and below[volume]:
        return "criativo"
    if above[volume] and above[consistency]:
        return "seguranca"
    if above[volume] and above[effectiveness] and above[quality]:
        return "impacto"
    if sum(below.values()) >= 3:
        return "limitado"
    return "regular"


def _attach_xp_profile_archetypes(rows: list[dict]) -> None:
    if not rows:
        return
    medians = _xp_profile_axis_medians(rows)
    for row in rows:
        archetype = classify_xp_profile_archetype(row, medians)
        row["xp_profile_archetype"] = archetype
        row["xp_profile_archetype_label"] = XP_PROFILE_ARCHETYPE_LABELS[archetype]
        row["xp_profile_archetype_description"] = XP_PROFILE_ARCHETYPE_DESCRIPTIONS[archetype]


def _clear_xp_profile_bar_scores(row: dict) -> None:
    for raw_key, display_key, _metrics in XP_PROFILE_BAR_SPECS:
        row[display_key] = None
        row[raw_key] = None
        row.pop(f"{raw_key}_rank_in_group", None)
        row.pop(f"{raw_key}_rank_pool_in_group", None)
    for metric in XP_PROFILE_SUBMETRICS:
        row[f"{metric}_sub_display"] = None
        row.pop(f"{metric}_sub_index", None)
        row.pop(f"{metric}_sub_index_rank_in_group", None)
        row.pop(f"{metric}_sub_index_rank_pool_in_group", None)
    for idx_key, _lbl, _metrics, _inv in XP_INDEX_SPECS:
        row.pop(idx_key, None)
        row.pop(f"{idx_key}_tier", None)
    for badge_key, _lbl, _metrics, _icon in XP_BADGE_SPECS:
        row.pop(f"{badge_key}_earned", None)
    row.pop("xp_profile_archetype", None)
    row.pop("xp_profile_archetype_label", None)
    row.pop("xp_profile_archetype_description", None)


def xp_profile_bar_pass_thresholds(
    players: list[dict],
    *,
    percentile: int = XP_PROFILE_BAR_PASS_PERCENTILE,
) -> dict[str, float]:
    """Minimum completed passes at the position-group percentile (default P30)."""
    return p20_pass_thresholds_by_group(
        players,
        "passes_completed",
        percentile=percentile,
    )


def is_xp_profile_bar_eligible(
    player: dict,
    pass_thresholds: dict[str, float],
    *,
    min_minutes_pct: float = XP_PROFILE_MIN_MINUTES_PCT,
) -> bool:
    minutes_pct = player.get("minutes_pct")
    if minutes_pct is None or float(minutes_pct) <= float(min_minutes_pct):
        return False
    group = _metric_rank_pool_key(player)
    min_passes = float(pass_thresholds.get(group, 0.0))
    return float(player.get("passes_completed") or 0.0) >= min_passes


def _xp_profile_passes_base_eligible(
    player: dict,
    pass_thresholds: dict[str, float],
) -> bool:
    return is_xp_profile_bar_eligible(player, pass_thresholds)


def _attach_xp_profile_bar_eligibility_for_pool(rows: list[dict]) -> list[dict]:
    """Flag profile-bar eligibility within one position group.

    Base filter: minutes > 30% and passes >= P30.
    When at least 250 players pass the base filter, keep only the top 250 by passes.
    Otherwise keep everyone who passes the base filter.
    """
    if not rows:
        return []

    pass_thresholds = xp_profile_bar_pass_thresholds(rows)
    group = _metric_rank_pool_key(rows[0])
    p30_threshold = round(float(pass_thresholds.get(group, 0.0)), 1)
    base_eligible = [
        row for row in rows
        if _xp_profile_passes_base_eligible(row, pass_thresholds)
    ]
    use_top_pool = len(base_eligible) >= XP_PROFILE_TOP_PASS_POOL_SIZE

    if use_top_pool:
        top_pool = sorted(
            base_eligible,
            key=lambda row: float(row.get("passes_completed") or 0.0),
            reverse=True,
        )[:XP_PROFILE_TOP_PASS_POOL_SIZE]
        eligible_ids = {id(row) for row in top_pool}
        top_cutoff = round(float(top_pool[-1].get("passes_completed") or 0.0), 1)
    else:
        eligible_ids = {id(row) for row in base_eligible}
        top_cutoff = p30_threshold

    for row in rows:
        row["xp_profile_p30_min_passes"] = p30_threshold
        row["xp_profile_min_minutes_pct"] = XP_PROFILE_MIN_MINUTES_PCT
        row["xp_profile_eligibility_mode"] = "top_pool" if use_top_pool else "threshold"
        if use_top_pool:
            row["xp_profile_top_pool_size"] = XP_PROFILE_TOP_PASS_POOL_SIZE
            row["xp_profile_min_passes"] = top_cutoff
        else:
            row.pop("xp_profile_top_pool_size", None)
            row["xp_profile_min_passes"] = p30_threshold

        minutes_ok = (
            row.get("minutes_pct") is not None
            and float(row.get("minutes_pct") or 0) > XP_PROFILE_MIN_MINUTES_PCT
        )
        passes_p30_ok = float(row.get("passes_completed") or 0.0) >= p30_threshold
        in_top = id(row) in eligible_ids
        row["xp_profile_bars_eligible"] = in_top

        if in_top:
            row.pop("xp_profile_ineligible_reason", None)
        elif not minutes_ok:
            row["xp_profile_ineligible_reason"] = "minutes"
        elif not passes_p30_ok:
            row["xp_profile_ineligible_reason"] = "passes_p30"
        elif use_top_pool:
            row["xp_profile_ineligible_reason"] = "top_pool_cutoff"
        else:
            row["xp_profile_ineligible_reason"] = "passes_p30"

    return [row for row in rows if row.get("xp_profile_bars_eligible")]


def attach_xp_profile_bar_eligibility(players: list[dict]) -> None:
    """Flag players who meet profile-bar eligibility within each position group."""
    pools: dict[str, list[dict]] = {}
    for player in players:
        group = _metric_rank_pool_key(player)
        pools.setdefault(group, []).append(player)

    for rows in pools.values():
        _attach_xp_profile_bar_eligibility_for_pool(rows)


def attach_composite_indices(players: list[dict]) -> None:
    """Within-position z-score composites for xP archetype radar and profile bars."""
    if not players:
        return
    pools: dict[str, list[dict]] = {}
    for player in players:
        group = _metric_rank_pool_key(player)
        pools.setdefault(group, []).append(player)

    for rows in pools.values():
        _attach_game_std_adjusted(rows)
        df = pd.DataFrame(rows)
        position_group = str(rows[0].get("position_group") or "")
        eligible_rows = _attach_xp_profile_bar_eligibility_for_pool(rows)
        builder_cols = list(BUILDER_BASE_METRICS)
        if position_group == "fullbacks":
            builder_cols.extend(BUILDER_FB_METRICS)

        composites = {
            "xp_builder_index": _mean_z_columns(df, tuple(builder_cols)),
            "xp_creator_index": _mean_z_columns(df, CREATOR_METRICS),
            "xp_progressor_index": _mean_z_columns(df, PROGRESSOR_METRICS),
            "xp_finisher_pass_index": _mean_z_columns(df, FINISHER_METRICS),
        }
        display_map = {
            "xp_builder_index": "xp_archetype_builder_display",
            "xp_creator_index": "xp_archetype_creator_display",
            "xp_progressor_index": "xp_archetype_progressor_display",
            "xp_finisher_pass_index": "xp_archetype_finisher_display",
        }
        for raw_key, composite in composites.items():
            _attach_index_display_scores(rows, raw_key, display_map[raw_key], composite)

        # Profile bars: rank only among eligible peers (base filters, or top 250 by passes).
        if eligible_rows:
            for raw_key, display_key, metric_cols in XP_PROFILE_BAR_SPECS:
                _attach_median_rank_display_scores(
                    eligible_rows,
                    metric_cols,
                    raw_key,
                    display_key,
                )
            for metric in XP_PROFILE_SUBMETRICS:
                _attach_median_rank_display_scores(
                    eligible_rows,
                    (metric,),
                    f"{metric}_sub_index",
                    f"{metric}_sub_display",
                )
            _attach_secondary_indices(eligible_rows)
            _attach_xp_profile_archetypes(eligible_rows)
        for row in rows:
            if not row.get("xp_profile_bars_eligible"):
                _clear_xp_profile_bar_scores(row)


def _attach_secondary_indices(eligible_rows: list[dict]) -> None:
    """Attach z-composite indices and tercile tiers (below/mid/above) among eligible peers."""
    if not eligible_rows:
        return
    edf = pd.DataFrame(eligible_rows)
    pool = len(eligible_rows)
    for idx_key, _label, metrics, invert in XP_INDEX_SPECS:
        composite = _mean_z_columns(edf, metrics, invert=invert)
        order = composite.rank(method="min", ascending=False)
        for i, row in enumerate(eligible_rows):
            row[idx_key] = float(composite.iloc[i])
            pct = float(order.iloc[i]) / pool if pool else 1.0
            if pct <= 1.0 / 3.0:
                tier = "above"
            elif pct <= 2.0 / 3.0:
                tier = "mid"
            else:
                tier = "below"
            row[f"{idx_key}_tier"] = tier

    # Achievement badges: earned only when the player is inside the top N of the
    # position on EVERY metric of the pair (e.g. Impacto = top 25 em xP/Jogo E xP/Passe).
    for badge_key, _label, metrics, _icon in XP_BADGE_SPECS:
        metric_top: list[pd.Series] = []
        for metric in metrics:
            col = pd.to_numeric(edf.get(metric), errors="coerce")
            order = col.rank(method="min", ascending=False)
            metric_top.append((order <= XP_BADGE_TOP_SIZE) & col.notna())
        earned_mask = metric_top[0]
        for extra in metric_top[1:]:
            earned_mask = earned_mask & extra
        for i, row in enumerate(eligible_rows):
            row[f"{badge_key}_earned"] = bool(earned_mask.iloc[i])


def _xp_pass_rating_shrink_sample(feature_key: str, player: dict) -> float:
    if feature_key in {"xp_per_90", "threat_passes_p90"}:
        return float(player.get("minutes") or 0.0)
    return float(player.get("passes_completed") or 0.0)


def _xp_pass_rating_shrink_k(feature_key: str) -> float:
    if feature_key in {"xp_per_90", "threat_passes_p90"}:
        return float(pe.SHRINKAGE_MINUTES_K)
    return float(pe.SHRINKAGE_PASS_K)


def _xp_pass_rating_shrink_value(
    feature_key: str,
    player: dict,
    pool_values: list[float],
) -> float:
    clean = [float(v) for v in pool_values if v is not None and np.isfinite(float(v))]
    prior = float(np.mean(clean)) if clean else 0.0
    raw = player.get(feature_key)
    sample = _xp_pass_rating_shrink_sample(feature_key, player)
    if raw is None or sample <= 0:
        return prior
    weight = sample / (sample + _xp_pass_rating_shrink_k(feature_key))
    return weight * float(raw) + (1.0 - weight) * prior


def _xp_pass_rating_tanh_display(z_score: float) -> float:
    return float(
        pe.RATING_DISPLAY_MID
        + XP_PASS_RATING_TANH_AMPLITUDE * np.tanh(float(z_score) / XP_PASS_RATING_TANH_SCALE)
    )


def _xp_pass_rating_confidence(player: dict) -> float:
    minutes = float(player.get("minutes") or 0)
    passes = float(player.get("passes_completed") or 0)
    pass_ref = max(float(player.get("position_p25_passes") or pe.RATING_CONFIDENCE_PASSES), 1.0)
    conf_minutes = min(1.0, minutes / pe.RATING_CONFIDENCE_MINUTES)
    conf_passes = min(1.0, passes / pass_ref)
    return (
        XP_PASS_RATING_CONFIDENCE_PASS_WEIGHT * conf_passes
        + XP_PASS_RATING_CONFIDENCE_MINUTES_WEIGHT * conf_minutes
    )


def _apply_xp_pass_rating_confidence(score_percentile: float, confidence: float) -> tuple[float, float]:
    efetivo = 1.0 - XP_PASS_RATING_CONFIDENCE_WEIGHT * (1.0 - confidence)
    grade = efetivo * score_percentile + (1.0 - efetivo) * pe.RATING_DISPLAY_MID
    uncertainty = (1.0 - efetivo) * pe.RATING_TANH_AMPLITUDE
    return float(grade), float(uncertainty)


def xp_pass_rating_percentile_display(rank: int, pool_size: int) -> float:
    """Map within-position rank to a 4.5–9.0 display score.

    Top 10% -> 8.0–9.0, 10–30% -> 7.0–8.0, rest -> 4.5–7.0.
    """
    if pool_size <= 0 or rank <= 0:
        return pe.RATING_DISPLAY_MID
    pct = float(rank) / float(pool_size)
    prev_pct = 0.0
    for max_pct, score_start, score_end in XP_PASS_RATING_PERCENTILE_BANDS:
        if pct <= max_pct:
            span = max_pct - prev_pct
            if span <= 0:
                return score_end
            t = (pct - prev_pct) / span
            return score_start - t * (score_start - score_end)
        prev_pct = max_pct
    return XP_PASS_RATING_PERCENTILE_BANDS[-1][2]


def attach_xp_pass_ratings(players: list[dict]) -> None:
    """Attach xP pass rating (2-metric weighted mean + shrinkage) with percentile display.

    The composite is a weighted arithmetic mean of within-position z-scores:
    Productivity (xP/game, 50%) and Effectiveness (xP/pass, 50%). Players are ranked
    by that composite; the displayed grade maps the rank to a 4.5–9.0 scale (top 10%
    -> 8–9, 10–30% -> 7–8, rest -> 4.5–7) with a single light confidence pull
    toward 6.0.
    """
    if not players:
        return

    pools: dict[str, list[dict]] = {}
    for player in players:
        group = _metric_rank_pool_key(player)
        pools.setdefault(group, []).append(player)

    for rows in pools.values():
        pool_size = len(rows)
        if pool_size == 0:
            continue

        passes = [float(p.get("passes_completed") or 0.0) for p in rows]
        p25_passes = float(np.percentile(passes, 25)) if passes else float(pe.RATING_CONFIDENCE_PASSES)
        p25_passes = max(p25_passes, 1.0)

        shrunk_by_feature: dict[str, list[float]] = {}
        for feature_key in XP_PASS_RATING_FEATURES:
            pool_values = [float(p.get(feature_key) or 0.0) for p in rows]
            shrunk_by_feature[feature_key] = [
                _xp_pass_rating_shrink_value(feature_key, player, pool_values)
                for player in rows
            ]

        feature_frame = pd.DataFrame(shrunk_by_feature)
        z_frame = feature_frame.apply(_zscore)
        weights = pd.Series(XP_PASS_RATING_FEATURE_WEIGHTS)
        composite_scores = (
            z_frame.mul(weights, axis=1).sum(axis=1) / weights.sum()
        ).astype(float).tolist()

        for player in rows:
            player["position_p25_passes"] = round(p25_passes, 1)

        raw_displays = [_xp_pass_rating_tanh_display(score) for score in composite_scores]
        for player, raw_display, composite_z in zip(rows, raw_displays, composite_scores):
            player["xp_pass_rating_raw_display"] = round(raw_display, 2)
            player["xp_pass_rating_confidence"] = round(_xp_pass_rating_confidence(player), 4)
            player["xp_pass_rating_composite_z"] = round(float(composite_z), 4)

        ranked = sorted(
            zip(rows, raw_displays),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        for rank, (row, _display) in enumerate(ranked, start=1):
            row["xp_pass_rating_rank_in_group"] = rank
            row["xp_pass_rating_rank_pool_in_group"] = pool_size
            pct_display = xp_pass_rating_percentile_display(rank, pool_size)
            confidence = float(row.get("xp_pass_rating_confidence") or 0.0)
            adjusted, uncertainty = _apply_xp_pass_rating_confidence(pct_display, confidence)
            row["xp_pass_rating_percentile_display"] = round(pct_display, 2)
            row["xp_pass_rating_uncertainty"] = round(uncertainty, 2)
            row["xp_pass_rating"] = round(adjusted / 10.0, 4)
            metric_ranks = row.get("metric_ranks")
            if not isinstance(metric_ranks, dict):
                metric_ranks = {}
            metric_ranks["xp_pass_rating"] = {
                "rank": rank,
                "total": pool_size,
                "value": row.get("xp_pass_rating"),
            }
            row["metric_ranks"] = metric_ranks


PASS_LENGTH_MIN_PEERS = 5


def attach_pass_length_profile(players: list[dict]) -> None:
    """Long/short pass share per player plus the peer average inside the same rank pool."""
    if not players:
        return

    pools: dict[str, list[dict]] = {}
    for player in players:
        pools.setdefault(_metric_rank_pool_key(player), []).append(player)

    for rows in pools.values():
        shares: list[float] = []
        for row in rows:
            short = float(row.get("passes_short") or 0.0)
            long_ = float(row.get("passes_long") or 0.0)
            band_total = short + long_
            if band_total > 0:
                long_share = long_ / band_total * 100.0
                row["long_pass_share_pct"] = round(long_share, 1)
                row["short_pass_share_pct"] = round(100.0 - long_share, 1)
                shares.append(long_share)
            else:
                row["long_pass_share_pct"] = None
                row["short_pass_share_pct"] = None

        if len(shares) < PASS_LENGTH_MIN_PEERS:
            for row in rows:
                row["long_pass_share_peer_avg_pct"] = None
                row["long_pass_share_peer_count"] = len(shares)
                row["long_pass_share_pctile"] = None
            continue

        sorted_shares = np.sort(np.asarray(shares, dtype=float))
        peer_avg = round(float(sorted_shares.mean()), 1)
        p10, p90 = np.percentile(sorted_shares, [10, 90])
        peer_span = max(5.0, float(p90 - p10) / 2.0)
        peer_span = min(peer_span, 11.0)
        for row in rows:
            row["long_pass_share_peer_avg_pct"] = peer_avg
            row["long_pass_share_peer_count"] = len(shares)
            row["long_pass_share_peer_span_pp"] = round(peer_span, 2)
            share = row.get("long_pass_share_pct")
            if share is None:
                row["long_pass_share_pctile"] = None
                continue
            rank = float(np.searchsorted(sorted_shares, float(share), side="right"))
            row["long_pass_share_pctile"] = round(rank / len(sorted_shares) * 100.0, 1)


def attach_distance_indices(players: list[dict]) -> None:
    """Within-position index per band with balanced grades and light volume weight."""
    if not players:
        return
    pools: dict[str, list[dict]] = {}
    for player in players:
        group = _metric_rank_pool_key(player)
        pools.setdefault(group, []).append(player)

    skill_weight = DISTANCE_INDEX_SKILL_WEIGHT
    volume_weight = DISTANCE_INDEX_VOLUME_WEIGHT

    for rows in pools.values():
        df = pd.DataFrame(rows)
        for band in BANDS:
            per_pass_col = f"xp_m4_per_pass_{band}"
            rate_col = f"xp_m4_threat_rate_{band}"
            p90_col = f"xp_m4_threat_{band}_p90"
            passes_col = f"passes_{band}"
            pass_counts = df.get(passes_col, pd.Series(0, index=df.index)).astype(float)
            min_passes = float(
                np.percentile(pass_counts.to_numpy(dtype=float), DISTANCE_INDEX_MIN_PASS_PERCENTILE)
            )
            eligible = pass_counts >= min_passes

            for i, row in enumerate(rows):
                row[f"xp_dist_index_{band}_min_passes"] = min_passes
                row[f"xp_dist_index_{band}_eligible"] = bool(eligible.iloc[i])
                row.pop(f"xp_dist_index_{band}_grade", None)

            if int(eligible.sum()) < 2:
                for row in rows:
                    row[f"xp_dist_index_{band}"] = None
                    row.pop(f"xp_dist_index_{band}_rank_in_group", None)
                    row.pop(f"xp_dist_index_{band}_rank_pool_in_group", None)
                continue

            sub = df.loc[eligible]
            pool_size = int(len(sub))
            z_per = _zscore(sub[per_pass_col].astype(float))
            z_rate = _zscore(sub[rate_col].astype(float))
            z_p90 = _zscore(sub[p90_col].astype(float))
            z_vol = _zscore(np.log1p(sub[passes_col].astype(float)))
            composite = (
                skill_weight * z_per
                + skill_weight * z_rate
                + skill_weight * z_p90
                + volume_weight * z_vol
            )

            rank_per = _rank_descending(sub[per_pass_col])
            rank_rate = _rank_descending(sub[rate_col])
            rank_p90 = _rank_descending(sub[p90_col])
            rank_vol = _rank_descending(sub[passes_col])

            eligible_rows = [rows[i] for i in sub.index]
            ranked = sorted(
                zip(eligible_rows, composite.tolist(), sub.index.tolist()),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            for rank, (row, z_val, sub_idx) in enumerate(ranked, start=1):
                row[f"xp_dist_index_{band}"] = float(z_val)
                row[f"xp_dist_index_{band}_rank_in_group"] = rank
                row[f"xp_dist_index_{band}_rank_pool_in_group"] = pool_size

                skill_pcts = [
                    float(rank_per.loc[sub_idx]) / pool_size,
                    float(rank_rate.loc[sub_idx]) / pool_size,
                    float(rank_p90.loc[sub_idx]) / pool_size,
                ]
                grade = _balanced_grade_from_rank_pcts(skill_pcts)
                vol_pct = float(rank_vol.loc[sub_idx]) / pool_size
                row[f"xp_dist_index_{band}_grade"] = _apply_volume_grade_penalty(grade, vol_pct)

            for i, row in enumerate(rows):
                if not eligible.iloc[i]:
                    row[f"xp_dist_index_{band}"] = None
                    row.pop(f"xp_dist_index_{band}_rank_in_group", None)
                    row.pop(f"xp_dist_index_{band}_rank_pool_in_group", None)
                    row.pop(f"xp_dist_index_{band}_grade", None)

        for row in rows:
            band_vals = [
                float(row[f"xp_dist_index_{band}"])
                for band in BANDS
                if row.get(f"xp_dist_index_{band}_eligible")
                and row.get(f"xp_dist_index_{band}") is not None
            ]
            row["xp_dist_index_mean"] = float(np.mean(band_vals)) if band_vals else None


def distance_index_grade(rank: int | None, total: int | None) -> str | None:
    """Legacy helper: map composite rank to a grade label."""
    if not rank or not total or rank <= 0 or total <= 0:
        return None
    return _grade_from_rank_pct(float(rank) / float(total))


def distance_index_grade_for_profile(profile: dict, band: str) -> str | None:
    if not profile.get(f"xp_dist_index_{band}_eligible", True):
        return None
    stored = profile.get(f"xp_dist_index_{band}_grade")
    if stored:
        return str(stored)
    return distance_index_grade(
        profile.get(f"xp_dist_index_{band}_rank_in_group"),
        profile.get(f"xp_dist_index_{band}_rank_pool_in_group"),
    )


def _eligible_profile_pool_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("xp_profile_bars_eligible")]


def _metric_rank_pool_key(player: dict) -> str:
    """Rank peers within campo ofensivo/defensivo when assigned; else position group."""
    origin = str(player.get("midfield_origin_profile") or "").strip().lower()
    if origin in {"campo_ofensivo", "campo_defensivo"}:
        return origin
    return str(player.get("position_group") or "CM")


def _clear_metric_ranks(row: dict, metrics: tuple[str, ...]) -> None:
    for metric in metrics:
        if metric.startswith("xp_dist_index_"):
            continue
        row.pop(f"{metric}_rank_in_group", None)
        row.pop(f"{metric}_rank_pool_in_group", None)


def attach_metric_ranks_within_position(
    players: list[dict],
    metrics: tuple[str, ...],
    *,
    eligible_only: bool = False,
) -> None:
    """Rank metrics within position group, optionally restricted to profile-eligible peers."""
    pools: dict[str, list[dict]] = {}
    for player in players:
        group = _metric_rank_pool_key(player)
        pools.setdefault(group, []).append(player)

    rank_metrics = tuple(
        metric for metric in metrics if not metric.startswith("xp_dist_index_")
    )
    for rows in pools.values():
        comparison_rows = _eligible_profile_pool_rows(rows) if eligible_only else rows
        pool_size = len(comparison_rows)
        if eligible_only:
            for row in rows:
                if not row.get("xp_profile_bars_eligible"):
                    _clear_metric_ranks(row, rank_metrics)
        if not comparison_rows:
            continue
        for metric in rank_metrics:
            comparison_rows.sort(
                key=lambda row: float(row.get(metric) or 0.0),
                reverse=True,
            )
            for rank, row in enumerate(comparison_rows, start=1):
                row[f"{metric}_rank_in_group"] = rank
                row[f"{metric}_rank_pool_in_group"] = pool_size


def attach_all_stats_ranks(players: list[dict]) -> None:
    """Rank stats-tab and Player Analysis metrics within eligible profile peers."""
    rank_metrics = tuple(
        dict.fromkeys(
            (*XP_STATS_RANK_METRICS, *XP_PLAYER_ANALYSIS_RANK_METRICS, *XP_REGULAR_STAT_RANK_KEYS)
        )
    )
    attach_metric_ranks_within_position(players, rank_metrics, eligible_only=True)


def metric_qualitative_grade(profile: dict, key: str) -> str | None:
    rank = profile.get(f"{key}_rank_in_group")
    total = profile.get(f"{key}_rank_pool_in_group")
    if not rank or not total:
        return None
    return _grade_from_rank_pct(float(rank) / float(total))


def format_threat_rate_display(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{100 * float(value):.1f}%"


def stats_metric_label(key: str) -> str:
    return XP_STATS_LABELS.get(key, key)


def pa_stats_metric_label(key: str) -> str:
    return XP_PA_LABELS.get(key, stats_metric_label(key))


def pa_stats_metric_tooltip(key: str) -> str:
    return XP_PA_TOOLTIPS.get(key, "")


def _format_residual_display(value: float) -> str:
    return f"{100.0 * value:+.1f}"


def format_pa_stats_value(key: str, value: float | int | None) -> str:
    if value is None:
        return "—"
    val = float(value)
    if key in {"xp_m4_threat_rate", "xp_surprise_rate", "xp_games_above_median_pct"}:
        return f"{100 * val:.1f}%"
    if key.startswith("xp_residual"):
        return _format_residual_display(val)
    if key in {"xp_game_std_adj", "xp_game_std_adj_score"}:
        return f"{val:+.2f}"
    if key in {"xp_m4_per_pass", "xp_m4_per_threat_pass"}:
        return f"{val:.2f}"
    if key in {"xp_per_90", "threat_passes_p90"}:
        return f"{val:.1f}"
    return format_stats_value(key, value)


def format_stats_value(key: str, value: float | int | None) -> str:
    if value is None:
        return "—"
    val = float(value)
    if key == "passes_completed":
        return f"{int(val):,}"
    if key.startswith("passes_"):
        return f"{int(val):,}"
    if key.startswith("xp_dist_index_"):
        if value is None:
            return "— (< P30)"
        return f"{val:.2f}"
    if key.startswith("xp_m4_threat_rate"):
        return f"{100 * val:.1f}%"
    if key == "xp_m4_threat_passes":
        return f"{int(val):,}"
    if key.endswith("_rate") or key.endswith("_share") or key.endswith("_pct") or key == "xp_surprise_rate" or key == "xp_threat_conversion":
        return f"{100 * val:.1f}%"
    if key.startswith("xp_m4_per_pass_") or key == "xp_m4_per_pass_final_third":
        return f"{val:.3f}"
    if key == "xp_m4_per_pass":
        return f"{val:.3f}"
    if key == "xp_m4_per_threat_pass":
        return f"{val:.3f}"
    if key == "threat_passes_p90":
        return f"{val:.2f}"
    if key in {
        "long_balls", "progressive_passes", "final_third_passes",
        "passes_to_box", "key_passes",
    }:
        return f"{val:.1f}"
    if key == "pass_mean_distance":
        return f"{val:.1f} m"
    if key.startswith("xp_residual"):
        return _format_residual_display(val)
    if key.endswith("_p90") or key == "xp_per_90" or key == "xp_game_mean" or key == "xp_game_std":
        return f"{val:.2f}"
    if key in {"xp_game_std_adj", "xp_game_std_adj_score"}:
        return f"{val:+.3f}"
    if key == "xp_pass_cv" or key == "xp_pass_std":
        return f"{val:.3f}"
    if key == "xp_max_pass" or key == "xp_m4_p90":
        return f"{val:.3f}"
    if key in XP_COMPOSITE_INDEX_KEYS:
        return f"{val:+.2f}"
    if key in XP_ARCHETYPE_RADAR_KEYS or key in XP_PROFILE_BAR_KEYS:
        return f"{val:.1f}"
    return f"{val:.1f}"
