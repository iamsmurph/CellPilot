import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


CONF_ORDER = {"A": 4, "B": 3, "C": 2, "D": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify agreement between TF–TF Jaccard (ChIP) and DoRothEA-derived TF–TF similarity."
    )
    parser.add_argument(
        "--jaccard-csv",
        default="/data/rbg/users/seanmurphy/CellPilot/data/datasets/supplementary/chip_tf_jaccard_top500_pairs.csv",
        help="CSV with columns: tf_a, tf_b, jaccard",
    )
    parser.add_argument(
        "--dorothea-csv",
        default="/data/rbg/users/seanmurphy/CellPilot/data/datasets/supplementary/dorothea_net.csv",
        help='DoRothEA edge list CSV with columns: "source","confidence","target","mor"',
    )
    parser.add_argument(
        "--min-confidence",
        choices=["A", "B", "C", "D"],
        default="C",
        help="Keep DoRothEA edges with confidence ≥ this letter (A strongest).",
    )
    parser.add_argument(
        "--mor-corr-min-overlap",
        type=int,
        default=3,
        help="Minimum shared targets to compute MOR correlation (otherwise NaN).",
    )
    parser.add_argument(
        "--out-csv",
        default="/data/rbg/users/seanmurphy/CellPilot/data/datasets/supplementary/dorothea_jaccard_agreement.csv",
        help="Output CSV of merged pairwise metrics.",
    )
    parser.add_argument(
        "--out-summary",
        default="/data/rbg/users/seanmurphy/CellPilot/data/datasets/supplementary/dorothea_jaccard_agreement.txt",
        help="Output text file with summary stats (also printed).",
    )
    return parser.parse_args()


def filter_dorothea(df: pd.DataFrame, min_conf: str) -> pd.DataFrame:
    min_rank = CONF_ORDER[min_conf]
    df = df.copy()
    df["confidence_rank"] = df["confidence"].map(CONF_ORDER)
    df = df[df["confidence_rank"] >= min_rank]
    return df.drop(columns=["confidence_rank"])  # not needed further


def build_tf_target_views(df: pd.DataFrame) -> Tuple[Dict[str, set], Dict[str, set], Dict[str, set], Dict[str, pd.Series]]:
    # Returns: (targets_any, targets_pos, targets_neg, mor_series)
    targets_any: Dict[str, set] = {}
    targets_pos: Dict[str, set] = {}
    targets_neg: Dict[str, set] = {}
    mor_series: Dict[str, pd.Series] = {}

    for tf, sub in df.groupby("source"):
        t_any = set(sub["target"].astype(str))
        t_pos = set(sub.loc[sub["mor"] > 0, "target"].astype(str))
        t_neg = set(sub.loc[sub["mor"] < 0, "target"].astype(str))
        targets_any[tf] = t_any
        targets_pos[tf] = t_pos
        targets_neg[tf] = t_neg
        # Keep MOR as a Series indexed by target for correlation
        s = sub.set_index("target")["mor"].astype(float)
        mor_series[tf] = s

    return targets_any, targets_pos, targets_neg, mor_series


def jaccard_of_sets(a: set, b: set) -> float:
    if not a and not b:
        return np.nan
    u = len(a | b)
    if u == 0:
        return np.nan
    return len(a & b) / float(u)


def same_sign_overlap_jaccard(pos_a: set, neg_a: set, pos_b: set, neg_b: set) -> float:
    same_pos = len(pos_a & pos_b)
    same_neg = len(neg_a & neg_b)
    union = len((pos_a | pos_b) | (neg_a | neg_b))
    if union == 0:
        return np.nan
    return (same_pos + same_neg) / float(union)


def mor_correlation(
    s_a: pd.Series, s_b: pd.Series, min_overlap: int = 3
) -> Tuple[float, int]:
    common = s_a.index.intersection(s_b.index)
    n = int(len(common))
    if n < min_overlap:
        return (np.nan, n)
    x = s_a.loc[common].astype(float).to_numpy()
    y = s_b.loc[common].astype(float).to_numpy()
    # Spearman is robust to scale and nonlinearity
    rho, _ = spearmanr(x, y)
    if isinstance(rho, float) and (not math.isnan(rho)):
        return (float(rho), n)
    return (np.nan, n)


def compute_pair_metrics(
    pairs: pd.DataFrame,
    tf_targets_any: Dict[str, set],
    tf_targets_pos: Dict[str, set],
    tf_targets_neg: Dict[str, set],
    tf_mor: Dict[str, pd.Series],
    min_overlap: int,
) -> pd.DataFrame:
    recs = []
    for tf_a, tf_b, chip_j in pairs[["tf_a", "tf_b", "jaccard"]].itertuples(index=False):
        any_a = tf_targets_any.get(tf_a, set())
        any_b = tf_targets_any.get(tf_b, set())
        pos_a = tf_targets_pos.get(tf_a, set())
        pos_b = tf_targets_pos.get(tf_b, set())
        neg_a = tf_targets_neg.get(tf_a, set())
        neg_b = tf_targets_neg.get(tf_b, set())

        dor_j_bin = jaccard_of_sets(any_a, any_b)
        dor_j_signed = same_sign_overlap_jaccard(pos_a, neg_a, pos_b, neg_b)

        s_a = tf_mor.get(tf_a)
        s_b = tf_mor.get(tf_b)
        mor_r, mor_n = (np.nan, 0)
        if s_a is not None and s_b is not None:
            mor_r, mor_n = mor_correlation(s_a, s_b, min_overlap=min_overlap)

        recs.append(
            {
                "tf_a": tf_a,
                "tf_b": tf_b,
                "chip_jaccard": float(chip_j),
                "dorothea_jaccard_binary": dor_j_bin,
                "dorothea_jaccard_signed": dor_j_signed,
                "dorothea_mor_spearman": mor_r,
                "dorothea_mor_overlap": mor_n,
                "n_targets_tf_a": len(any_a),
                "n_targets_tf_b": len(any_b),
                "n_same_sign": len((pos_a & pos_b)) + len((neg_a & neg_b)),
                "n_union_signed": len((pos_a | pos_b) | (neg_a | neg_b)),
                "n_overlap_any": len(any_a & any_b),
                "n_union_any": len(any_a | any_b),
            }
        )

    return pd.DataFrame.from_records(recs)


def spearman_safe(x: pd.Series, y: pd.Series) -> float:
    s = x.combine(y, lambda a, b: not (pd.isna(a) or pd.isna(b)))
    mask = s.astype(bool)
    if mask.sum() < 3:
        return np.nan
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho) if isinstance(rho, float) else np.nan


def summarize(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Total merged TF–TF pairs: {len(df):,}")

    rho_bin = spearman_safe(df["chip_jaccard"], df["dorothea_jaccard_binary"]) 
    rho_signed = spearman_safe(df["chip_jaccard"], df["dorothea_jaccard_signed"]) 
    rho_mor = spearman_safe(df["chip_jaccard"], df["dorothea_mor_spearman"]) 

    lines.append(f"Spearman(chip_jaccard, dorothea_jaccard_binary): {rho_bin:.3f} (NaNs dropped)")
    lines.append(f"Spearman(chip_jaccard, dorothea_jaccard_signed): {rho_signed:.3f} (NaNs dropped)")
    lines.append(f"Spearman(chip_jaccard, dorothea_mor_spearman): {rho_mor:.3f} (NaNs dropped)")

    # Simple enrichment check: mean chip_jaccard for pairs with any Dorothea overlap vs none
    has_any = df["n_overlap_any"] > 0
    mean_has = df.loc[has_any, "chip_jaccard"].mean()
    mean_not = df.loc[~has_any, "chip_jaccard"].mean()
    lines.append(
        f"Mean chip_jaccard | Dorothea overlap>0: {mean_has:.4f}; overlap==0: {mean_not:.4f}"
    )

    # And by MOR correlation availability
    has_mor = df["dorothea_mor_overlap"] >= 3
    mean_has_mor = df.loc[has_mor, "chip_jaccard"].mean()
    mean_no_mor = df.loc[~has_mor, "chip_jaccard"].mean()
    lines.append(
        f"Mean chip_jaccard | MOR overlap≥3: {mean_has_mor:.4f}; <3: {mean_no_mor:.4f}"
    )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    jaccard_df = pd.read_csv(args.jaccard_csv)
    dorothea_df = pd.read_csv(args.dorothea_csv)

    # Coerce types / names
    jaccard_df = jaccard_df.rename(columns={"tf_a": "tf_a", "tf_b": "tf_b", "jaccard": "jaccard"})
    dorothea_df = dorothea_df.rename(
        columns={"source": "source", "confidence": "confidence", "target": "target", "mor": "mor"}
    )
    dorothea_df["source"] = dorothea_df["source"].astype(str)
    dorothea_df["target"] = dorothea_df["target"].astype(str)

    # Filter Dorothea by confidence
    dorothea_df = filter_dorothea(dorothea_df, args.min_confidence)

    # Only consider pairs where both TFs exist in Dorothea as sources (TFs)
    dor_tfs = set(dorothea_df["source"].unique().tolist())
    filtered_pairs = jaccard_df[(jaccard_df["tf_a"].isin(dor_tfs)) & (jaccard_df["tf_b"].isin(dor_tfs))].copy()

    # Build TF→targets views
    tf_any, tf_pos, tf_neg, tf_mor = build_tf_target_views(dorothea_df)

    # Compute metrics
    merged = compute_pair_metrics(
        filtered_pairs,
        tf_targets_any=tf_any,
        tf_targets_pos=tf_pos,
        tf_targets_neg=tf_neg,
        tf_mor=tf_mor,
        min_overlap=args.mor_corr_min_overlap,
    )

    # Save outputs
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    summary = summarize(merged)
    print(summary)
    out_txt = Path(args.out_summary)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(summary + "\n")


if __name__ == "__main__":
    main()


