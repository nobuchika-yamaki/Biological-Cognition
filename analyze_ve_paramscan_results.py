#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze parameter-scan results for the viability-preserving controllable-future expansion project.

Input directory:
    ~/Desktop/ve_paramscan_only_fast

Expected structure:
    ve_paramscan_only_fast/
        parameter_scan/
            parameter_scan_summary.csv
            <setting>/
                summary_by_model.csv
                primary_pairwise_tests.csv
                family_vs_incomplete_tests.csv
                pareto_summary.csv
                accepted_environment_diagnostics.csv

Outputs:
    ve_paramscan_only_fast/analysis/
        robustness_overview.csv
        setting_level_interpretation.csv
        model_rankings_by_setting.csv
        weak_or_failed_settings.csv
        pairwise_primary_all_settings.csv
        family_tests_all_settings.csv
        parameter_scan_interpretation.txt
        figures/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FAMILY_MODELS = [
    "viability_constrained_endpoint",
    "joint_viability_expansion",
]

INCOMPLETE_MODELS = [
    "random_admissible",
    "homeostasis_only",
    "uncertainty_minimizing",
    "novelty_seeking",
    "unconstrained_controllability",
]

MAIN_MODEL = "joint_viability_expansion"
PRIMARY = "joint_viable_expansion_score"

DEFAULT_OUTDIR = Path("~/Desktop/ve_paramscan_only_fast").expanduser()


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return pd.DataFrame()


def find_scan_dir(root: Path) -> Path:
    if (root / "parameter_scan").exists():
        return root / "parameter_scan"
    if (root / "parameter_scan_summary.csv").exists():
        return root
    raise FileNotFoundError(
        f"Could not find parameter_scan directory or parameter_scan_summary.csv under {root}"
    )


def setting_dirs(scan_dir: Path) -> List[Path]:
    dirs = []
    for p in sorted(scan_dir.iterdir()):
        if p.is_dir() and (p / "summary_by_model.csv").exists():
            dirs.append(p)
    return dirs


def extract_setting_type(label: str) -> str:
    if label == "default":
        return "default"
    if label.startswith("Hpi_"):
        return "planning_horizon"
    if label.startswith("Hc_"):
        return "controllability_horizon"
    if label.startswith("hmin_"):
        return "viability_threshold"
    if label.startswith("alpha_"):
        return "weight_grid"
    if label.startswith("thr_"):
        return "screening_thresholds"
    return "other"


def load_setting_details(scan_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranking_rows: List[Dict] = []
    pairwise_rows: List[pd.DataFrame] = []
    family_rows: List[pd.DataFrame] = []
    pareto_rows: List[pd.DataFrame] = []
    diagnostics_rows: List[pd.DataFrame] = []

    for d in setting_dirs(scan_dir):
        label = d.name

        summary = safe_read_csv(d / "summary_by_model.csv")
        if not summary.empty:
            score_col = f"{PRIMARY}_mean"
            if score_col in summary.columns:
                tmp = summary.copy()
                tmp["analysis_label"] = label
                tmp["setting_type"] = extract_setting_type(label)
                tmp["rank_by_primary"] = tmp[score_col].rank(ascending=False, method="min")
                tmp["is_family_model"] = tmp["model"].isin(FAMILY_MODELS)
                ranking_rows.extend(tmp.to_dict("records"))

        pw = safe_read_csv(d / "primary_pairwise_tests.csv")
        if not pw.empty:
            pw["analysis_label"] = label
            pw["setting_type"] = extract_setting_type(label)
            pairwise_rows.append(pw)

        ft = safe_read_csv(d / "family_vs_incomplete_tests.csv")
        if not ft.empty:
            ft["analysis_label"] = label
            ft["setting_type"] = extract_setting_type(label)
            family_rows.append(ft)

        pareto = safe_read_csv(d / "pareto_summary.csv")
        if not pareto.empty:
            pareto["analysis_label"] = label
            pareto["setting_type"] = extract_setting_type(label)
            pareto_rows.append(pareto)

        diag = safe_read_csv(d / "accepted_environment_diagnostics.csv")
        if not diag.empty:
            diag["analysis_label"] = label
            diag["setting_type"] = extract_setting_type(label)
            diagnostics_rows.append(diag)

    rankings = pd.DataFrame(ranking_rows)
    pairwise = pd.concat(pairwise_rows, ignore_index=True) if pairwise_rows else pd.DataFrame()
    family = pd.concat(family_rows, ignore_index=True) if family_rows else pd.DataFrame()
    pareto_all = pd.concat(pareto_rows, ignore_index=True) if pareto_rows else pd.DataFrame()
    diagnostics_all = pd.concat(diagnostics_rows, ignore_index=True) if diagnostics_rows else pd.DataFrame()

    return rankings, pairwise, family, pareto_all, diagnostics_all


def build_overview(scan_summary: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    df = scan_summary.copy()

    if "analysis_label" not in df.columns:
        raise ValueError("parameter_scan_summary.csv lacks analysis_label column.")

    df["setting_type"] = df["analysis_label"].map(extract_setting_type)

    df["family_advantage_positive"] = df["family_minus_incomplete_mean_psi"] > 0
    if "family_vs_incomplete_p_one_sided" in df.columns:
        df["family_advantage_p_lt_05"] = df["family_vs_incomplete_p_one_sided"] < 0.05
    else:
        df["family_advantage_p_lt_05"] = np.nan

    df["joint_lower_than_endpoint"] = df["joint_minus_endpoint_mean_psi"] < 0

    if not rankings.empty and f"{PRIMARY}_mean" in rankings.columns:
        top = (
            rankings.sort_values(["analysis_label", f"{PRIMARY}_mean"], ascending=[True, False])
            .groupby("analysis_label")
            .first()
            .reset_index()[["analysis_label", "model", f"{PRIMARY}_mean", "is_family_model"]]
        )
        top = top.rename(
            columns={
                "model": "top_model",
                f"{PRIMARY}_mean": "top_model_psi",
                "is_family_model": "top_model_is_family",
            }
        )
        df = df.merge(top, on="analysis_label", how="left")

        family_rank_stats = []
        for label, g in rankings.groupby("analysis_label"):
            fam = g[g["model"].isin(FAMILY_MODELS)]
            inc = g[g["model"].isin(INCOMPLETE_MODELS)]
            fam_min_rank = float(fam["rank_by_primary"].min()) if len(fam) else np.nan
            inc_best_rank = float(inc["rank_by_primary"].min()) if len(inc) else np.nan
            family_rank_stats.append({
                "analysis_label": label,
                "best_family_rank": fam_min_rank,
                "best_incomplete_rank": inc_best_rank,
                "family_has_top_rank": fam_min_rank == 1.0,
                "best_family_beats_best_incomplete": fam_min_rank < inc_best_rank,
            })
        rank_df = pd.DataFrame(family_rank_stats)
        df = df.merge(rank_df, on="analysis_label", how="left")

    def classify(row: pd.Series) -> str:
        if not bool(row.get("family_advantage_positive", False)):
            return "failed_family_advantage"
        if row.get("best_family_beats_best_incomplete", True) is False:
            return "weak_best_incomplete_matches_or_exceeds_family"
        if row.get("family_advantage_p_lt_05", True) is False:
            return "positive_but_not_significant_in_small_scan"
        return "supported"

    df["interpretation_status"] = df.apply(classify, axis=1)

    diff = df["family_minus_incomplete_mean_psi"].astype(float)
    df["family_advantage_band"] = pd.cut(
        diff,
        bins=[-np.inf, 0.0, 0.05, 0.15, np.inf],
        labels=["none_or_negative", "small", "moderate", "large"],
    )

    return df


def summarize_overview(overview: pd.DataFrame) -> str:
    n = len(overview)
    n_pos = int(overview["family_advantage_positive"].sum()) if n else 0
    n_sig = int(overview["family_advantage_p_lt_05"].fillna(False).sum()) if n else 0
    n_supported = int((overview["interpretation_status"] == "supported").sum()) if n else 0
    n_top_family = int(overview.get("family_has_top_rank", pd.Series(False, index=overview.index)).fillna(False).sum()) if n else 0

    lines: List[str] = []
    lines.append("PARAMETER-SCAN INTERPRETATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Number of completed settings: {n}")
    lines.append(f"Family advantage positive: {n_pos}/{n}")
    lines.append(f"Family advantage p < .05: {n_sig}/{n}")
    lines.append(f"Supported settings by scan criterion: {n_supported}/{n}")
    lines.append(f"At least one family model had the top primary score: {n_top_family}/{n}")
    lines.append("")

    if "family_minus_incomplete_mean_psi" in overview.columns:
        lines.append("Family-minus-incomplete mean Ψ:")
        lines.append(f"  mean   = {overview['family_minus_incomplete_mean_psi'].mean():.6f}")
        lines.append(f"  median = {overview['family_minus_incomplete_mean_psi'].median():.6f}")
        lines.append(f"  min    = {overview['family_minus_incomplete_mean_psi'].min():.6f}")
        lines.append(f"  max    = {overview['family_minus_incomplete_mean_psi'].max():.6f}")
        lines.append("")

    by_type = (
        overview.groupby("setting_type")
        .agg(
            n=("analysis_label", "count"),
            positive=("family_advantage_positive", "sum"),
            significant=("family_advantage_p_lt_05", "sum"),
            mean_advantage=("family_minus_incomplete_mean_psi", "mean"),
            min_advantage=("family_minus_incomplete_mean_psi", "min"),
        )
        .reset_index()
    )

    lines.append("By setting type:")
    for _, r in by_type.iterrows():
        lines.append(
            f"  {r['setting_type']}: "
            f"positive {int(r['positive'])}/{int(r['n'])}, "
            f"p<.05 {int(r['significant'])}/{int(r['n'])}, "
            f"mean advantage {r['mean_advantage']:.6f}, "
            f"min {r['min_advantage']:.6f}"
        )
    lines.append("")

    weak = overview[overview["interpretation_status"] != "supported"].copy()
    if len(weak):
        lines.append("Weak or failed settings:")
        for _, r in weak.sort_values("family_minus_incomplete_mean_psi").iterrows():
            pval = r.get("family_vs_incomplete_p_one_sided", np.nan)
            lines.append(
                f"  {r['analysis_label']}: {r['interpretation_status']}, "
                f"family-minus-incomplete Ψ = {r['family_minus_incomplete_mean_psi']:.6f}, "
                f"p = {pval:.4g}"
            )
    else:
        lines.append("Weak or failed settings: none under the implemented criterion.")
    lines.append("")

    lines.append("Interpretation rule:")
    lines.append(
        "  The scan supports the central proposition if viability-constrained expansion "
        "family models consistently exceed incomplete single-principle models. "
        "The joint model does not need to uniquely beat the endpoint family member."
    )

    return "\n".join(lines)


def save_figures(overview: pd.DataFrame, rankings: pd.DataFrame, outdir: Path) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    tmp = overview.sort_values("family_minus_incomplete_mean_psi", ascending=False).reset_index(drop=True)
    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(tmp)), tmp["family_minus_incomplete_mean_psi"].to_numpy(dtype=float))
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(np.arange(len(tmp)), tmp["analysis_label"].tolist(), rotation=90, fontsize=6)
    plt.ylabel("Family - incomplete models, mean Ψ")
    plt.title("Parameter scan: family advantage across settings")
    plt.tight_layout()
    plt.savefig(figdir / "family_advantage_all_settings.png", dpi=300)
    plt.close()

    by_type = (
        overview.groupby("setting_type")["family_minus_incomplete_mean_psi"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    plt.figure(figsize=(8, 4))
    x = np.arange(len(by_type))
    plt.bar(x, by_type["mean"].to_numpy(dtype=float))
    plt.errorbar(
        x,
        by_type["mean"].to_numpy(dtype=float),
        yerr=[
            by_type["mean"].to_numpy(dtype=float) - by_type["min"].to_numpy(dtype=float),
            by_type["max"].to_numpy(dtype=float) - by_type["mean"].to_numpy(dtype=float),
        ],
        fmt="none",
        capsize=3,
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, by_type["setting_type"].tolist(), rotation=30, ha="right")
    plt.ylabel("Family - incomplete models, mean Ψ")
    plt.title("Family advantage by parameter class")
    plt.tight_layout()
    plt.savefig(figdir / "family_advantage_by_setting_type.png", dpi=300)
    plt.close()

    if "joint_minus_endpoint_mean_psi" in overview.columns:
        tmp2 = overview.sort_values("joint_minus_endpoint_mean_psi", ascending=False).reset_index(drop=True)
        plt.figure(figsize=(12, 5))
        plt.bar(np.arange(len(tmp2)), tmp2["joint_minus_endpoint_mean_psi"].to_numpy(dtype=float))
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.xticks(np.arange(len(tmp2)), tmp2["analysis_label"].tolist(), rotation=90, fontsize=6)
        plt.ylabel("Joint - endpoint, mean Ψ")
        plt.title("Diagnostic: joint model versus endpoint family member")
        plt.tight_layout()
        plt.savefig(figdir / "joint_vs_endpoint_all_settings.png", dpi=300)
        plt.close()

    if not rankings.empty and f"{PRIMARY}_mean" in rankings.columns:
        rank_piv = rankings.pivot(index="analysis_label", columns="model", values="rank_by_primary")
        rank_piv = rank_piv.reindex(overview["analysis_label"].tolist())
        plt.figure(figsize=(10, max(5, 0.22 * len(rank_piv))))
        plt.imshow(rank_piv.to_numpy(dtype=float), aspect="auto")
        plt.colorbar(label="Rank by primary Ψ; lower is better")
        plt.xticks(np.arange(rank_piv.shape[1]), rank_piv.columns.tolist(), rotation=45, ha="right")
        plt.yticks(np.arange(rank_piv.shape[0]), rank_piv.index.tolist(), fontsize=6)
        plt.title("Model ranks across parameter settings")
        plt.tight_layout()
        plt.savefig(figdir / "model_rank_matrix.png", dpi=300)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_OUTDIR), help="Root output directory, e.g. ~/Desktop/ve_paramscan_only_fast")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    scan_dir = find_scan_dir(root)
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    scan_summary_path = scan_dir / "parameter_scan_summary.csv"
    if not scan_summary_path.exists():
        scan_summary_path = scan_dir / "parameter_scan_summary_incremental.csv"

    scan_summary = safe_read_csv(scan_summary_path)
    if scan_summary.empty:
        raise FileNotFoundError(f"No usable parameter_scan_summary found under {scan_dir}")

    rankings, pairwise, family_tests, pareto_all, diagnostics_all = load_setting_details(scan_dir)

    overview = build_overview(scan_summary, rankings)

    overview.to_csv(analysis_dir / "robustness_overview.csv", index=False)
    rankings.to_csv(analysis_dir / "model_rankings_by_setting.csv", index=False)
    pairwise.to_csv(analysis_dir / "pairwise_primary_all_settings.csv", index=False)
    family_tests.to_csv(analysis_dir / "family_tests_all_settings.csv", index=False)
    pareto_all.to_csv(analysis_dir / "pareto_all_settings.csv", index=False)
    diagnostics_all.to_csv(analysis_dir / "diagnostics_all_settings.csv", index=False)

    weak = overview[overview["interpretation_status"] != "supported"].copy()
    weak.to_csv(analysis_dir / "weak_or_failed_settings.csv", index=False)

    compact_cols = [
        "analysis_label",
        "setting_type",
        "n_env",
        "H_pi",
        "H_c",
        "hmin",
        "alpha",
        "eta",
        "threshold_rho_omega",
        "threshold_rho_delta",
        "threshold_kappa",
        "family_mean_psi",
        "incomplete_mean_psi",
        "family_minus_incomplete_mean_psi",
        "family_vs_incomplete_p_one_sided",
        "joint_mean_psi",
        "endpoint_mean_psi",
        "joint_minus_endpoint_mean_psi",
        "top_model",
        "top_model_is_family",
        "best_family_beats_best_incomplete",
        "interpretation_status",
    ]
    compact_cols = [c for c in compact_cols if c in overview.columns]
    overview[compact_cols].to_csv(analysis_dir / "setting_level_interpretation.csv", index=False)

    report = summarize_overview(overview)
    (analysis_dir / "parameter_scan_interpretation.txt").write_text(report, encoding="utf-8")

    save_figures(overview, rankings, analysis_dir)

    print(report)
    print(f"\n[done] analysis outputs saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
