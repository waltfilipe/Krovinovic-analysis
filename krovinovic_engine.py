"""Build and inject Filip Krovinović metrics from season_all.csv into the European comparison pool."""

from __future__ import annotations

import functools
import math

import numpy as np
import pandas as pd

import midfield_origin as mo
import passes_engine as pe
import player_profiles as pp
import progression_engine as pge
import xp_engine as xe
import xp_stats_engine as xstats

from krovinovic_config import (
    CROATIAN_MIN_MINUTES_PCT,
    CROATIAN_PASS_PERCENTILE,
    CROATIAN_RANK_POOL_LABEL,
    TARGET_LEAGUE_LABEL,
    TARGET_LEAGUE_SOURCE,
    TARGET_PLAYER_ID,
    TARGET_PLAYER_NAME,
)

compute_pass_ratings = pe.compute_pass_ratings
pg_compute_progression_ratings = pge.compute_progression_ratings

CROATIAN_RANK_METRICS: tuple[str, ...] = tuple(
    dict.fromkeys(
        key
        for key in (
            *(
                metric
                for metric in xstats.XP_REGULAR_STAT_RANK_KEYS
                if metric != "long_pass_share_pct"
            ),
            *xstats.XP_PLAYER_ANALYSIS_RANK_METRICS,
            *xe.XP_POSITION_RANK_METRICS,
            "xp_per_90",
            "xp_m4_per_pass",
            "impact_passes_p90",
            "special_line_break_p90",
            "progressive_passes",
            "final_third_passes",
            "key_passes",
            "long_balls",
        )
    )
)


def _load_target_pass_frame() -> pd.DataFrame:
    frame = pe._load_season_pass_frame()
    if frame.empty:
        return frame
    work = frame[frame["player_id"].astype(str) == TARGET_PLAYER_ID].copy()
    if work.empty:
        return work
    work["league_source"] = TARGET_LEAGUE_SOURCE
    return work


@functools.lru_cache(maxsize=4)
def load_target_xp_passes() -> pd.DataFrame:
    frame = _load_target_pass_frame()
    if frame.empty:
        return pd.DataFrame()
    return xe._build_season_passes_from_frame(frame, blend_league_reference=False)


def build_target_pass_player(
    *,
    tier_model: str,
    classification_model: str,
    xt_surface_mode: str,
) -> tuple[dict | None, pd.DataFrame | None]:
    frame = _load_target_pass_frame()
    if frame.empty:
        return None, None

    frame = frame.copy()
    frame["player_id"] = frame["player_id"].astype(str)
    if "position" in frame.columns:
        frame["position"] = frame["position"].astype(str).str.strip().str.upper()

    passes = pe._enrich_passes(
        frame,
        tier_model=tier_model,
        classification_model=classification_model,
        xt_surface_mode=xt_surface_mode,
    )
    grp_passes = pe.filter_live_ball_passes(passes[passes["player_id"] == TARGET_PLAYER_ID])
    if grp_passes is None or grp_passes.empty:
        return None, None

    registry = pe.build_player_registry(frame)
    player_meta = next((p for p in registry if str(p["code"]) == TARGET_PLAYER_ID), None)
    if player_meta is None:
        return None, None

    minutes_info = pe._minutes_from_passes_frame(frame)
    mins = minutes_info.get(TARGET_PLAYER_ID, {})
    metrics = pe.compute_player_metrics(grp_passes, mins)
    player = {
        "player_id": TARGET_PLAYER_ID,
        "player_name": player_meta.get("name", TARGET_PLAYER_NAME),
        "position": player_meta.get("position", "CM"),
        "position_group": pe.rating_position_group(player_meta.get("position")),
        "team": mins.get("team", "—"),
        "minutes": mins.get("minutes"),
        "minutes_pct": mins.get("minutes_pct"),
        "league": TARGET_LEAGUE_LABEL,
        "league_source": TARGET_LEAGUE_SOURCE,
        "passes_completed": metrics.get("passes_completed", 0),
        **{
            k: round(v, 4) if isinstance(v, float) and abs(v) < 1000 else v
            for k, v in metrics.items()
        },
    }
    return player, grp_passes.copy()


def build_target_xp_profile(xp_passes: pd.DataFrame) -> dict | None:
    if xp_passes is None or xp_passes.empty:
        return None

    frame = _load_target_pass_frame()
    minutes_info = pe._minutes_from_passes_frame(frame)
    mins = minutes_info.get(TARGET_PLAYER_ID, {})
    metrics = xstats.compute_extended_xp_stats(xp_passes)
    if not metrics:
        return None

    minutes = mins.get("minutes")
    xstats.attach_regular_pass_stats(metrics, frame, minutes)
    xstats.apply_per90_metrics(metrics, minutes)

    registry = pe.build_player_registry(frame)
    player_meta = next((p for p in registry if str(p["code"]) == TARGET_PLAYER_ID), None)
    position = player_meta.get("position", "CM") if player_meta else "CM"
    name = player_meta.get("name", TARGET_PLAYER_NAME) if player_meta else TARGET_PLAYER_NAME

    return {
        "player_id": TARGET_PLAYER_ID,
        "player_name": name,
        "position": position,
        "position_group": pe.rating_position_group(position),
        "team": mins.get("team", "—"),
        "minutes": mins.get("minutes"),
        "minutes_pct": mins.get("minutes_pct"),
        "league": TARGET_LEAGUE_LABEL,
        "league_source": TARGET_LEAGUE_SOURCE,
        "passes_completed": int((xp_passes["is_won"] & xp_passes["has_end"]).sum()),
        **metrics,
    }


def _load_croatian_midfield_pass_frame() -> pd.DataFrame:
    frame = pe._load_season_pass_frame()
    if frame.empty:
        return frame
    work = pe._filter_pass_frame_to_midfielders(frame).copy()
    if work.empty:
        return work
    work["league_source"] = TARGET_LEAGUE_SOURCE
    return work


def _build_xp_profile_from_group(
    pid: str,
    grp: pd.DataFrame,
    *,
    registry_by_id: dict[str, dict],
    minutes_info: dict[str, dict],
    raw_pass_frame: pd.DataFrame,
) -> dict | None:
    player_meta = registry_by_id.get(str(pid))
    if player_meta is None:
        return None

    metrics = xstats.compute_extended_xp_stats(grp)
    if not metrics:
        return None

    mins = minutes_info.get(str(pid), {})
    minutes = mins.get("minutes")
    player_raw = raw_pass_frame[raw_pass_frame["player_id"].astype(str) == str(pid)]
    xstats.attach_regular_pass_stats(metrics, player_raw, minutes)
    xstats.apply_per90_metrics(metrics, minutes)

    position = player_meta.get("position", "CM")
    return _sanitize_numeric_profile({
        "player_id": str(pid),
        "player_name": player_meta.get("name", "—"),
        "position": position,
        "position_group": pe.rating_position_group(position),
        "team": mins.get("team", "—"),
        "minutes": mins.get("minutes"),
        "minutes_pct": mins.get("minutes_pct"),
        "league": TARGET_LEAGUE_LABEL,
        "league_source": TARGET_LEAGUE_SOURCE,
        "passes_completed": int((grp["is_won"] & grp["has_end"]).sum()),
        **metrics,
    })


@functools.lru_cache(maxsize=2)
def build_croatian_midfield_xp_profiles() -> tuple[tuple[dict, ...], float]:
    """xP profiles for all Croatian-league midfielders in season_all.csv."""
    frame = _load_croatian_midfield_pass_frame()
    if frame.empty:
        return (), 0.0

    minutes_info = pe._minutes_from_passes_frame(frame)
    registry = pe.build_player_registry(frame)
    registry_by_id = {str(player["code"]): player for player in registry}
    xp_passes = xe._build_season_passes_from_frame(frame, blend_league_reference=False)
    if xp_passes.empty:
        return (), 0.0

    profiles: list[dict] = []
    for pid, grp in xp_passes.groupby("player_id", sort=False):
        profile = _build_xp_profile_from_group(
            str(pid),
            grp,
            registry_by_id=registry_by_id,
            minutes_info=minutes_info,
            raw_pass_frame=frame,
        )
        if profile is not None:
            profiles.append(profile)

    pass_counts = [float(profile.get("passes_completed") or 0.0) for profile in profiles]
    p25_threshold = (
        float(np.percentile(pass_counts, CROATIAN_PASS_PERCENTILE))
        if pass_counts
        else 0.0
    )
    return tuple(profiles), p25_threshold


def _croatian_eligible_pool(profiles: tuple[dict, ...], p25_threshold: float) -> list[dict]:
    eligible: list[dict] = []
    for profile in profiles:
        minutes_pct = profile.get("minutes_pct")
        if minutes_pct is None or float(minutes_pct) < CROATIAN_MIN_MINUTES_PCT:
            continue
        if float(profile.get("passes_completed") or 0.0) <= p25_threshold:
            continue
        eligible.append(profile)
    return eligible


def _rank_target_in_pool(target_profile: dict, pool: list[dict], metrics: tuple[str, ...]) -> None:
    pool_size = len(pool)
    target_profile["croatian_rank_pool_size"] = pool_size
    target_profile["croatian_rank_pool_label"] = CROATIAN_RANK_POOL_LABEL
    if pool_size <= 0:
        return

    target_id = str(target_profile.get("player_id"))
    for metric in metrics:
        ranked = sorted(
            pool,
            key=lambda row: float(row.get(metric) or 0.0),
            reverse=True,
        )
        for rank, row in enumerate(ranked, start=1):
            if str(row.get("player_id")) != target_id:
                continue
            target_profile[f"{metric}_rank_in_group"] = rank
            target_profile[f"{metric}_rank_pool_in_group"] = pool_size
            break


def attach_croatian_stat_ranks(target_profile: dict) -> None:
    """Rank Krovinović against Croatian-league midfielders (≥25% minutes, >P25 passes)."""
    profiles, p25_threshold = build_croatian_midfield_xp_profiles()
    pool = _croatian_eligible_pool(profiles, p25_threshold)
    if not pool:
        return

    target_id = str(target_profile.get("player_id"))
    if not any(str(row.get("player_id")) == target_id for row in pool):
        pool = [*pool, target_profile]

    for row in pool:
        row["midfield_origin_profile"] = "croatian_league"
    xstats.attach_pass_length_profile(pool)
    _rank_target_in_pool(target_profile, pool, CROATIAN_RANK_METRICS)


def _sanitize_numeric_profile(profile: dict) -> dict:
    """Replace non-finite metric values so single-player pools never break ranking."""
    clean = dict(profile)
    for key, value in clean.items():
        if isinstance(value, float) and not math.isfinite(value):
            clean[key] = 0.0
    return clean


def _copy_origin_fields(target: dict, source: dict) -> None:
    for key in (
        "position_group",
        "midfield_offensive_origin_pct",
        "midfield_origin_profile",
        "league",
        "league_source",
    ):
        if source.get(key) is not None:
            target[key] = source[key]


def _merge_xp_profiles(european_profiles: list[dict], target_profile: dict) -> dict[str, dict]:
    merged = [p for p in european_profiles if str(p.get("player_id")) != TARGET_PLAYER_ID]
    merged.append(_sanitize_numeric_profile(target_profile))
    merged.sort(key=lambda p: float(p.get("xp_m4_total", 0.0)), reverse=True)
    for i, profile in enumerate(merged, start=1):
        profile["xp_m4_rank"] = i
    return {str(p["player_id"]): p for p in merged}


def inject_target_into_bundle(
    bundle: tuple,
    *,
    tier_model: str,
    classification_model: str,
    xt_surface_mode: str,
) -> tuple:
    (
        analysis_players,
        passes_by_player,
        progression_by_id,
        pass_by_id,
        carries_by_id,
        progression_pool_by_position,
        pool_by_position,
        carries_pool_by_position,
        xp_by_id,
    ) = bundle

    target_player, target_passes = build_target_pass_player(
        tier_model=tier_model,
        classification_model=classification_model,
        xt_surface_mode=xt_surface_mode,
    )
    if target_player is None or target_passes is None:
        return bundle

    target_xp_passes = load_target_xp_passes()
    target_xp_profile = build_target_xp_profile(target_xp_passes)
    if target_xp_profile is None:
        return bundle

    european_players = [p for p in analysis_players if str(p.get("player_id")) != TARGET_PLAYER_ID]
    combined_players = european_players + [target_player]

    passes_by_player = dict(passes_by_player)
    passes_by_player[TARGET_PLAYER_ID] = target_passes

    empty_carries: dict[str, dict] = {}
    combined_players = mo.apply_midfield_position_groups(
        combined_players,
        passes_by_player,
        empty_carries,
    )
    target_player = next(
        p for p in combined_players if str(p.get("player_id")) == TARGET_PLAYER_ID
    )
    _copy_origin_fields(target_xp_profile, target_player)
    target_xp_profile = _sanitize_numeric_profile(target_xp_profile)

    _, pass_by_id, pool_by_position = compute_pass_ratings(combined_players)
    _, progression_by_id, progression_pool_by_position = pg_compute_progression_ratings(
        combined_players,
        [],
        pass_by_id=pass_by_id,
        carry_by_id=carries_by_id,
    )

    european_xp_profiles = [p for p in xp_by_id.values() if str(p.get("player_id")) != TARGET_PLAYER_ID]
    xp_by_id = _merge_xp_profiles(european_xp_profiles, target_xp_profile)

    origin_by_id = {
        str(p["player_id"]): {
            "position_group": p.get("position_group"),
            "midfield_offensive_origin_pct": p.get("midfield_offensive_origin_pct"),
            "midfield_origin_profile": p.get("midfield_origin_profile"),
            "league": p.get("league"),
            "league_source": p.get("league_source"),
        }
        for p in combined_players
    }
    for player in combined_players:
        pid = str(player["player_id"])
        player["age"] = pp.read_cached_age(pid)
    for xp_profile in xp_by_id.values():
        pid = str(xp_profile["player_id"])
        xp_profile["age"] = pp.read_cached_age(pid)
        origin = origin_by_id.get(pid)
        if origin:
            _copy_origin_fields(xp_profile, origin)
    xe.refresh_xp_midfield_origin_rankings(list(xp_by_id.values()))
    target_xp = xp_by_id.get(TARGET_PLAYER_ID)
    if target_xp:
        attach_croatian_stat_ranks(target_xp)

    for prof in progression_by_id.values():
        pid = str(prof.get("player_id"))
        prof["age"] = pp.read_cached_age(pid)
        origin = origin_by_id.get(pid)
        if origin:
            prof.setdefault("league", origin.get("league"))
            prof.setdefault("league_source", origin.get("league_source"))

    return (
        combined_players,
        passes_by_player,
        progression_by_id,
        pass_by_id,
        carries_by_id,
        progression_pool_by_position,
        pool_by_position,
        carries_pool_by_position,
        xp_by_id,
    )


def merge_xp_passes_grouped(european_grouped: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    merged = dict(european_grouped)
    target_xp = load_target_xp_passes()
    if not target_xp.empty:
        merged[TARGET_PLAYER_ID] = target_xp
    return merged
