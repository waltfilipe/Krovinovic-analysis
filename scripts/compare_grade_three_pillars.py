"""Compare the xP grade before/after the 3-pillar weighted mean.

Old grade: PCA on 4 features (xP/game, IP/game, xP/pass, xP/IP).
New grade: weighted mean of 3 pillars (xP/game 40%, xP/pass 30%, residual 30%).

Run manually:

    python3 scripts/compare_grade_three_pillars.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import xp_engine as xe  # noqa: E402
import xp_stats_engine as xs  # noqa: E402

LEGACY_FEATURES = (
    "xp_per_90",
    "threat_passes_p90",
    "xp_m4_per_pass",
    "xp_m4_per_threat_pass",
)
MIN_POOL_FOR_REPORT = 10


def legacy_ratings(rows: list[dict]) -> dict[int, dict]:
    """Reproduce the previous 4-feature PCA grade for one position pool."""
    from sklearn.decomposition import PCA

    pool_size = len(rows)
    shrunk = {}
    for key in LEGACY_FEATURES:
        pool_values = [float(r.get(key) or 0.0) for r in rows]
        shrunk[key] = [
            xs._xp_pass_rating_shrink_value(key, row, pool_values) for row in rows
        ]
    z_frame = pd.DataFrame(shrunk).apply(xs._zscore)
    if pool_size >= 8:
        scores = PCA(n_components=1, random_state=42).fit_transform(
            z_frame.to_numpy(dtype=float)
        ).ravel()
    else:
        scores = z_frame.mean(axis=1).to_numpy(dtype=float)

    displays = [xs._xp_pass_rating_tanh_display(s) for s in scores]
    order = sorted(range(pool_size), key=lambda i: displays[i], reverse=True)
    out: dict[int, dict] = {}
    for rank, i in enumerate(order, start=1):
        pct = xs.xp_pass_rating_percentile_display(rank, pool_size)
        confidence = float(rows[i].get("xp_pass_rating_confidence") or 0.0)
        adjusted, _ = xs._apply_xp_pass_rating_confidence(pct, confidence)
        out[i] = {"rank": rank, "pool": pool_size, "rating": adjusted / 10.0}
    return out


def main() -> int:
    _, players = xe.build_european_league_xp_analytics()
    if not players:
        print("no players")
        return 1

    pools: dict[str, list[dict]] = {}
    for player in players:
        pools.setdefault(xs._metric_rank_pool_key(player), []).append(player)

    records = []
    for pool_rows in pools.values():
        legacy = legacy_ratings(pool_rows)
        for i, row in enumerate(pool_rows):
            new_rating = row.get("xp_pass_rating")
            if new_rating is None:
                continue
            old = legacy[i]
            records.append({
                "player": row.get("player_name"),
                "team": row.get("team"),
                "grade_old": round(float(old["rating"]) * 10.0, 2),
                "grade_new": round(float(new_rating) * 10.0, 2),
                "delta": round((float(new_rating) - float(old["rating"])) * 10.0, 2),
                "rank_old": old["rank"],
                "rank_new": int(row.get("xp_pass_rating_rank_in_group") or 0),
                "pool": old["pool"],
                "xp_per_90": round(float(row.get("xp_per_90") or 0.0), 2),
                "xp_pass": round(float(row.get("xp_m4_per_pass") or 0.0), 3),
                "residual": round(float(row.get("xp_residual_median") or 0.0) * 100.0, 2),
                "ip_p90": round(float(row.get("threat_passes_p90") or 0.0), 2),
            })

    df = pd.DataFrame(records)
    df["rank_gain"] = df["rank_old"] - df["rank_new"]

    cols = [
        "player", "team", "grade_old", "grade_new", "delta",
        "rank_old", "rank_new", "xp_per_90", "xp_pass", "residual", "ip_p90",
    ]

    # Tiny pools (e.g. two wingers) swing wildly on rank-based display bands.
    ranked = df[df["pool"] >= MIN_POOL_FOR_REPORT]

    print(f"Jogadores avaliados: {len(df)} (pools >= {MIN_POOL_FOR_REPORT}: {len(ranked)})")
    print(f"Delta medio: {ranked.delta.mean():+.3f} | std: {ranked.delta.std():.3f}")
    print(
        "Correlacao de rank (Spearman): "
        f"{ranked.rank_old.corr(ranked.rank_new, method='spearman'):.3f}"
    )
    print(f"Mudam >= 0.5 ponto: {(ranked.delta.abs() >= 0.5).sum()}")

    print("\n=== TOP 15 MAIOR AUMENTO DE RATING ===")
    print(ranked.nlargest(15, "delta")[cols].to_string(index=False))

    print("\n=== TOP 15 MAIOR PERDA DE RATING ===")
    print(ranked.nsmallest(15, "delta")[cols].to_string(index=False))

    out_dir = pathlib.Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)
    df.sort_values("delta", ascending=False).to_csv(
        out_dir / "grade_three_pillars_shift.csv", index=False
    )
    print(f"\nCSV completo: {out_dir / 'grade_three_pillars_shift.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
