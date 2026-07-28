#!/usr/bin/env python3
"""Offline study: composite Impact Pass score from xP, residual and progress_ratio.

Runs multiple weight schemes and threshold strategies, compares against the
current is_threat_m4 rule and reports statistical diagnostics.

Usage:
    python3 scripts/study_impact_pass_composite.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import xp_engine as xe
import xp_study_engine as xse

OUT_DIR = ROOT / "scripts" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

XP_COL = xe.XP_COL
RESIDUAL_COL = xe.XP_RESIDUAL_COL
THREAT_COL = xe.THREAT_COL
BANDS = list(xse.DISTANCE_BAND_ORDER)


@dataclass(frozen=True)
class WeightScheme:
    key: str
    label: str
    w_xp: float
    w_res: float
    w_prog: float


WEIGHT_SCHEMES: tuple[WeightScheme, ...] = (
    WeightScheme("equal", "Equal (⅓ each)", 1 / 3, 1 / 3, 1 / 3),
    WeightScheme("xp_heavy", "xP-heavy (50/35/15)", 0.50, 0.35, 0.15),
    WeightScheme("res_heavy", "Residual-heavy (30/50/20)", 0.30, 0.50, 0.20),
    WeightScheme("prog_heavy", "Progress-heavy (35/35/30)", 0.35, 0.35, 0.30),
    WeightScheme("value_surprise", "Value+surprise (45/45/10)", 0.45, 0.45, 0.10),
    WeightScheme("balanced", "Balanced (45/35/20)", 0.45, 0.35, 0.20),
    WeightScheme("xp_residual", "xP+residual only (50/50/0)", 0.50, 0.50, 0.00),
)

PERCENTILE_CUTS: tuple[float, ...] = (0.90, 0.925, 0.95, 0.97)


def robust_z(series: pd.Series) -> pd.Series:
    arr = series.astype(float)
    med = float(arr.median())
    mad = float((arr - med).abs().median())
    scale = 1.4826 * mad
    if scale <= 1e-12:
        q75, q25 = np.percentile(arr, [75, 25])
        scale = max(float(q75 - q25) / 1.349, 1e-9)
    return (arr - med) / scale


def attach_robust_zscores(df: pd.DataFrame, *, by_band: bool) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["distance_band"] if by_band else []
    if group_cols:
        for feat, col in [
            ("z_xp", XP_COL),
            ("z_res", RESIDUAL_COL),
            ("z_prog", "progress_ratio"),
        ]:
            out[feat] = out.groupby(group_cols, sort=False)[col].transform(robust_z)
    else:
        out["z_xp"] = robust_z(out[XP_COL])
        out["z_res"] = robust_z(out[RESIDUAL_COL])
        out["z_prog"] = robust_z(out["progress_ratio"])
    return out


def composite_score(df: pd.DataFrame, scheme: WeightScheme) -> pd.Series:
    return (
        scheme.w_xp * df["z_xp"]
        + scheme.w_res * df["z_res"]
        + scheme.w_prog * df["z_prog"]
    )


def pca_weights(df: pd.DataFrame) -> tuple[float, float, float]:
    mat = df[["z_xp", "z_res", "z_prog"]].to_numpy(dtype=float)
    mat = mat - mat.mean(axis=0)
    cov = np.cov(mat, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    vec = eigvecs[:, int(np.argmax(eigvals))]
    vec = np.abs(vec)
    vec = vec / vec.sum()
    return float(vec[0]), float(vec[1]), float(vec[2])


def inverse_variance_weights(df: pd.DataFrame) -> tuple[float, float, float]:
    vars_ = df[["z_xp", "z_res", "z_prog"]].var().to_numpy(dtype=float)
    inv = 1.0 / np.maximum(vars_, 1e-9)
    w = inv / inv.sum()
    return float(w[0]), float(w[1]), float(w[2])


def flag_top_percentile(
    scores: pd.Series,
    pct: float,
    *,
    group: pd.Series | None = None,
) -> pd.Series:
    if group is None:
        cutoff = scores.quantile(pct)
        return scores >= cutoff
    out = pd.Series(False, index=scores.index)
    for _, idx in scores.groupby(group, sort=False).groups.items():
        sub = scores.loc[idx]
        cutoff = sub.quantile(pct)
        out.loc[idx] = sub >= cutoff
    return out


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 0.0


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float)
    b = b.astype(float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if pooled <= 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def load_completed_passes() -> pd.DataFrame:
    season = xe.load_european_league_season_passes()
    work = season[season["is_won"] & season["has_end"]].copy()
    if "progress_ratio" not in work.columns:
        work["progress_ratio"] = xse._progress_ratio_series(work)
  # Forward-only progress for impact candidacy (lateral/back passes rarely "impact")
    work["progress_ratio_pos"] = work["progress_ratio"].clip(lower=0.0)
    return work


def player_impact_rates(df: pd.DataFrame, flag_col: str) -> pd.Series:
    return (
        df.groupby("player_id", sort=False)[flag_col]
        .mean()
        .astype(float)
    )


def evaluate_variant(
    df: pd.DataFrame,
    flag_col: str,
    baseline_col: str,
    label: str,
) -> dict:
    flags = df[flag_col].to_numpy(dtype=bool)
    base = df[baseline_col].to_numpy(dtype=bool)
    rates = player_impact_rates(df, flag_col)
    base_rates = player_impact_rates(df, baseline_col)
    return {
        "label": label,
        "flag_rate_pct": round(100.0 * flags.mean(), 3),
        "short_rate_pct": round(100.0 * df.loc[df["distance_band"] == "short", flag_col].mean(), 3),
        "long_rate_pct": round(100.0 * df.loc[df["distance_band"] == "long", flag_col].mean(), 3),
        "jaccard_vs_threat_m4": round(jaccard(flags, base), 4),
        "player_rate_spearman": round(
            float(rates.corr(base_rates, method="spearman")), 4
        ),
        "mean_xp_flagged": round(float(df.loc[flags, XP_COL].mean()), 4),
        "mean_xp_other": round(float(df.loc[~flags, XP_COL].mean()), 4),
        "mean_res_flagged": round(float(df.loc[flags, RESIDUAL_COL].mean()), 4),
        "mean_prog_flagged": round(float(df.loc[flags, "progress_ratio"].mean()), 4),
        "cohens_d_xp": round(cohens_d(df.loc[flags, XP_COL], df.loc[~flags, XP_COL]), 3),
        "cohens_d_res": round(cohens_d(df.loc[flags, RESIDUAL_COL], df.loc[~flags, RESIDUAL_COL]), 3),
        "cohens_d_prog": round(cohens_d(df.loc[flags, "progress_ratio"], df.loc[~flags, "progress_ratio"]), 3),
    }


def main() -> int:
    print("Loading completed European league passes…")
    raw = load_completed_passes()
    n = len(raw)
    print(f"  {n:,} completed passes · threat_m4 rate {raw[THREAT_COL].mean()*100:.2f}%")

    # ── Feature distributions ─────────────────────────────────────────────
    feat_summary = {
        "n_passes": n,
        "xp": {
            "mean": round(float(raw[XP_COL].mean()), 4),
            "p50": round(float(raw[XP_COL].median()), 4),
            "p90": round(float(raw[XP_COL].quantile(0.90)), 4),
            "p95": round(float(raw[XP_COL].quantile(0.95)), 4),
        },
        "residual": {
            "mean": round(float(raw[RESIDUAL_COL].mean()), 4),
            "p50": round(float(raw[RESIDUAL_COL].median()), 4),
            "p90": round(float(raw[RESIDUAL_COL].quantile(0.90)), 4),
        },
        "progress_ratio": {
            "mean": round(float(raw["progress_ratio"].mean()), 4),
            "p50": round(float(raw["progress_ratio"].median()), 4),
            "share_forward": round(float((raw["progress_ratio"] > 0.05).mean()), 4),
            "share_backward": round(float((raw["progress_ratio"] < -0.05).mean()), 4),
        },
        "correlations": {
            "xp_vs_residual": round(float(raw[XP_COL].corr(raw[RESIDUAL_COL])), 4),
            "xp_vs_progress": round(float(raw[XP_COL].corr(raw["progress_ratio"])), 4),
            "residual_vs_progress": round(float(raw[RESIDUAL_COL].corr(raw["progress_ratio"])), 4),
        },
    }

    # ── Data-driven weights ─────────────────────────────────────────────────
    z_global = attach_robust_zscores(raw, by_band=False)
    z_band = attach_robust_zscores(raw, by_band=True)

    pca_w = pca_weights(z_band)
    iv_w = inverse_variance_weights(z_band)
    data_schemes = (
        WeightScheme("pca_band", f"PCA band ({pca_w[0]:.2f}/{pca_w[1]:.2f}/{pca_w[2]:.2f})", *pca_w),
        WeightScheme("invvar_band", f"Inv-var band ({iv_w[0]:.2f}/{iv_w[1]:.2f}/{iv_w[2]:.2f})", *iv_w),
    )
    all_schemes = WEIGHT_SCHEMES + data_schemes

    results: list[dict] = []

    for norm_label, frame in [("global_z", z_global), ("band_z", z_band)]:
        for scheme in all_schemes:
            scores = composite_score(frame, scheme)
            col_score = f"score_{norm_label}_{scheme.key}"
            frame[col_score] = scores

            for pct in PERCENTILE_CUTS:
                # Global cut
                flag_global = flag_top_percentile(scores, pct)
                col_g = f"flag_{norm_label}_{scheme.key}_p{int(pct*100)}_g"
                frame[col_g] = flag_global
                results.append(
                    evaluate_variant(
                        frame,
                        col_g,
                        THREAT_COL,
                        f"{norm_label} · {scheme.label} · P{int(pct*100)} global",
                    )
                )

                # Per-band cut
                flag_band = flag_top_percentile(scores, pct, group=frame["distance_band"])
                col_b = f"flag_{norm_label}_{scheme.key}_p{int(pct*100)}_b"
                frame[col_b] = flag_band
                results.append(
                    evaluate_variant(
                        frame,
                        col_b,
                        THREAT_COL,
                        f"{norm_label} · {scheme.label} · P{int(pct*100)} per-band",
                    )
                )

            # Hybrid: high composite + forward progress floor (progress-aware gate)
            if scheme.w_prog > 0:
                prog_floor = frame["progress_ratio"].quantile(0.60)
                flag_hybrid = (scores >= scores.quantile(0.90)) & (frame["progress_ratio"] >= prog_floor)
                col_h = f"flag_{norm_label}_{scheme.key}_hybrid90_prog60"
                frame[col_h] = flag_hybrid
                results.append(
                    evaluate_variant(
                        frame,
                        col_h,
                        THREAT_COL,
                        f"{norm_label} · {scheme.label} · P90 score + progress≥P60",
                    )
                )

    results_df = pd.DataFrame(results)

    # Filter to reasonable flag rates (2–12%) for recommendation shortlist
    shortlist = results_df[
        (results_df["flag_rate_pct"] >= 2.0) & (results_df["flag_rate_pct"] <= 12.0)
    ].copy()
    shortlist["score"] = (
        0.35 * shortlist["cohens_d_xp"]
        + 0.35 * shortlist["cohens_d_res"]
        + 0.15 * shortlist["cohens_d_prog"]
        + 0.15 * (1.0 - (shortlist["flag_rate_pct"] - 5.0).abs() / 10.0)
    )
    shortlist = shortlist.sort_values("score", ascending=False)

    # Best per family
    best_rows = []
    for norm in ["global_z", "band_z"]:
        sub = shortlist[shortlist["label"].str.startswith(norm)]
        if not sub.empty:
            best_rows.append(sub.iloc[0].to_dict())
    best_df = pd.DataFrame(best_rows)

    # Compare top composite vs threat_m4 at player level (band_z balanced P90)
    rec_col = None
    for c in z_band.columns:
        if c.startswith("flag_band_z_balanced_p90_b"):
            rec_col = c
            break
    player_compare = []
    if rec_col:
        pc = (
            z_band.groupby(["player_id", "player_name"], sort=False)
            .agg(
                passes=(XP_COL, "size"),
                threat_rate=(THREAT_COL, "mean"),
                composite_rate=(rec_col, "mean"),
                xp_per_pass=(XP_COL, "mean"),
            )
            .reset_index()
        )
        pc = pc[pc["passes"] >= 300].sort_values("composite_rate", ascending=False)
        player_compare = pc.head(15).to_dict(orient="records")

    # Progress ablation: does progress add signal beyond xp+res?
    ablation = []
    for scheme in [s for s in all_schemes if s.key in {"balanced", "xp_residual", "equal"}]:
        s_band = composite_score(z_band, scheme)
        flag = flag_top_percentile(s_band, 0.90, group=z_band["distance_band"])
        ablation.append(
            evaluate_variant(
                z_band.assign(_f=flag),
                "_f",
                THREAT_COL,
                f"ablation · {scheme.label} · P90 band",
            )
        )

    payload = {
        "feature_summary": feat_summary,
        "pca_weights_band": {"xp": pca_w[0], "residual": pca_w[1], "progress": pca_w[2]},
        "invvar_weights_band": {"xp": iv_w[0], "residual": iv_w[1], "progress": iv_w[2]},
        "n_variants_tested": len(results_df),
        "top_10_variants": shortlist.head(10).to_dict(orient="records"),
        "best_per_norm": best_df.to_dict(orient="records"),
        "ablation": ablation,
        "player_leaders_recommended": player_compare,
    }

    json_path = OUT_DIR / "impact_pass_composite_study.json"
    csv_path = OUT_DIR / "impact_pass_composite_study.csv"
    md_path = OUT_DIR / "impact_pass_composite_study.md"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    results_df.to_csv(csv_path, index=False)

    lines = [
        "# Impact Pass Composite Study (offline)",
        "",
        f"Base: **{n:,}** completed midfielder passes (European leagues).",
        "",
        "## Feature summary",
        "",
        f"- xP mean / P90: {feat_summary['xp']['mean']} / {feat_summary['xp']['p90']}",
        f"- Residual mean / P90: {feat_summary['residual']['mean']} / {feat_summary['residual']['p90']}",
        f"- Progress ratio mean: {feat_summary['progress_ratio']['mean']} "
        f"(forward share {feat_summary['progress_ratio']['share_forward']:.1%})",
        f"- Corr(xP, residual): {feat_summary['correlations']['xp_vs_residual']}",
        f"- Corr(xP, progress): {feat_summary['correlations']['xp_vs_progress']}",
        "",
        "## Data-driven weights (robust z per distance band)",
        "",
        f"- PCA: xP {pca_w[0]:.3f} · residual {pca_w[1]:.3f} · progress {pca_w[2]:.3f}",
        f"- Inverse variance: xP {iv_w[0]:.3f} · residual {iv_w[1]:.3f} · progress {iv_w[2]:.3f}",
        "",
        "## Current baseline",
        "",
        f"- `is_threat_m4` flag rate: **{raw[THREAT_COL].mean()*100:.2f}%**",
        "  (residual > P90 band AND xP ≥ P75 band)",
        "",
        "## Top 10 variants (flag rate 2–12%, ranked by separation score)",
        "",
        "| Variant | Flag % | Jaccard vs M4 | ρ player rates | d(xP) | d(res) | d(prog) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in shortlist.head(10).itertuples():
        lines.append(
            f"| {row.label} | {row.flag_rate_pct} | {row.jaccard_vs_threat_m4} | "
            f"{row.player_rate_spearman} | {row.cohens_d_xp} | {row.cohens_d_res} | {row.cohens_d_prog} |"
        )

    lines.extend([
        "",
        "## Ablation (progress contribution)",
        "",
        "| Variant | Flag % | Jaccard vs M4 | d(prog) |",
        "|---|---:|---:|---:|",
    ])
    for row in ablation:
        lines.append(
            f"| {row['label']} | {row['flag_rate_pct']} | {row['jaccard_vs_threat_m4']} | {row['cohens_d_prog']} |"
        )

    if best_df.shape[0]:
        lines.extend(["", "## Recommended starting points", ""])
        for row in best_df.itertuples():
            lines.append(
                f"- **{row.label}**: flag {row.flag_rate_pct}%, "
                f"Jaccard {row.jaccard_vs_threat_m4}, player ρ {row.player_rate_spearman}"
            )

    lines.extend([
        "",
        "## Interpretation notes",
        "",
        "- **band_z** normalisation is preferred: short/long passes have different xP/residual scales.",
        "- **Balanced (45/35/20)** or **PCA-derived** weights give interpretable trade-offs.",
        "- Progress adds separation (higher Cohen's d) but lowers Jaccard vs `is_threat_m4` "
        "(expected — M4 ignores progress).",
        "- Target operational flag rate: **~5–8%** for a rare 'impact' event label.",
        "",
    ])

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print("\nTop 5 variants:")
    for row in shortlist.head(5).itertuples():
        print(
            f"  {row.label}: flag={row.flag_rate_pct}% "
            f"jaccard={row.jaccard_vs_threat_m4} d_xp={row.cohens_d_xp} d_prog={row.cohens_d_prog}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
