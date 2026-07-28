"""Pre-populate the player profile cache (age, height, foot) for European midfielders.

Run offline so the app reads ages from cache without network calls in the hot path.

Examples:
    python scripts/prefetch_player_profiles.py --only-missing
    python scripts/prefetch_player_profiles.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import passes_engine as pe
import player_profiles as pp

LIGUE1_TEAM_TOKENS: frozenset[str] = frozenset({
    "paris saint-germain",
    "olympique de marseille",
    "olympique lyonnais",
    "as monaco",
    "lille",
    "nice",
    "ogc nice",
    "rc lens",
    "stade rennais",
    "stade brestois",
    "toulouse",
    "nantes",
    "fc nantes",
    "rc strasbourg",
    "angers",
    "auxerre",
    "le havre",
    "lorient",
    "metz",
    "paris fc",
    "red star",
    "saint-etienne",
    "saint-étienne",
})


def _is_ligue1_player(player: dict) -> bool:
    team = str(player.get("team") or "").strip().lower()
    return any(token in team for token in LIGUE1_TEAM_TOKENS)


def _needs_fetch(player: dict, *, only_missing: bool, force: bool) -> bool:
    if force:
        return True
    if not only_missing:
        return True
    return pp.read_cached_age(str(player.get("player_id", ""))) is None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch midfielder profile metadata.")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Fetch only players without a cached age (retries old empty resolved entries).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch every player, even when the cache already has an age.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.12,
        help="Delay between network requests in seconds (default: 0.12).",
    )
    parser.add_argument(
        "--ligue1-only",
        action="store_true",
        help="Restrict prefetch to Ligue 1 midfielders (by team name).",
    )
    args = parser.parse_args()

    players = pe.build_european_league_midfielders()
    if args.ligue1_only:
        players = [player for player in players if _is_ligue1_player(player)]
    targets = [
        player
        for player in players
        if _needs_fetch(player, only_missing=args.only_missing, force=args.force)
    ]
    total = len(targets)
    print(
        f"Prefetching profiles for {total}/{len(players)} midfielders "
        f"(only_missing={args.only_missing}, force={args.force})…",
        flush=True,
    )

    resolved_age = 0
    resolved_any = 0
    for i, player in enumerate(targets, start=1):
        profile = pp.get_player_profile(
            str(player.get("player_id", "")),
            str(player.get("player_name", "")),
            str(player.get("team", "")),
            force=args.force or args.only_missing,
        )
        if profile.get("age") is not None:
            resolved_age += 1
        if any(profile.get(k) for k in ("age", "photo_url", "height", "dominant_foot", "nationality")):
            resolved_any += 1
        if i % 20 == 0 or i == total:
            print(
                f"  {i}/{total} · ages: {resolved_age} · any profile field: {resolved_any}",
                flush=True,
            )
        if args.sleep > 0:
            time.sleep(args.sleep)

    all_with_age = sum(
        1 for player in players if pp.read_cached_age(str(player["player_id"])) is not None
    )
    print(
        f"Done. Ages resolved for {all_with_age}/{len(players)} players in cache.",
        flush=True,
    )


if __name__ == "__main__":
    main()
