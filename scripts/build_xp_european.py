#!/usr/bin/env python3
"""Build xP artifacts and the European midfielders parquet (run offline, not in Streamlit)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import xp_engine as xe


def main() -> None:
    print(f"Building global xP artifacts ({xe.XP_MODEL_VERSION})...")
    meta = xe.fit_and_save_artifacts(force=True)
    print("Ridge / threat meta version:", meta.get("version"))

    print("Building European season parquet (Ligue 1 + 65/35 blend)...")
    season = xe.build_european_league_season_passes(refit_artifacts=False)
    print(f"European passes: {len(season):,}")
    print(f"Completed: {int((season['is_won'] & season['has_end']).sum()):,}")
    print(f"Players: {season['player_id'].nunique():,}")
    print(f"Matches: {season['event_id'].nunique():,}")
    print(f"Threat passes: {int(season[xe.THREAT_COL].sum()):,}")
    print("Done:", xe.XP_EUROPEAN_PASSES_PARQUET)


if __name__ == "__main__":
    main()
