import argparse
import os
from typing import Tuple

import pandas as pd

def canonicalize_pair(tf_a: str, tf_b: str) -> Tuple[str, str]:
    """
    Return a deterministic ordering of a TF pair to ensure A,B == B,A.
    """
    a, b = (tf_a or "").strip(), (tf_b or "").strip()
    return tuple(sorted([a, b]))  # type: ignore[return-value]

def build_template_text(tf_a: str, tf_b: str, include_task_hint: bool = True) -> str:
    a_norm, b_norm = canonicalize_pair(tf_a, tf_b)
    lines = []
    if include_task_hint:
        lines.append(
            "Instruction: Represent the propensity that two human transcription factors co-bind (appear in overlapping ChIP-seq peaks). Encode signal sources such as binding motifs, known PPIs, and chromatin context. Map to the concept of expected Jaccard overlap (0–1)."
        )
    lines.append("Input:")
    lines.append("species: Homo sapiens (hg38)")
    lines.append("assay: ChIP-seq (IDR peaks; Jaccard on peak sets)")
    lines.append("pair_order: unordered")
    lines.append(f"TF_A: {a_norm}")
    lines.append(f"TF_B: {b_norm}")
    if include_task_hint:
        lines.append("")
        lines.append("Scale anchors:")
        lines.append("~0.0: little/no known co-binding or context overlap")
        lines.append("~0.5: moderate overlap in some contexts or cell types")
        lines.append("~1.0: frequent co-binding across contexts with strong evidence")
    return "\n".join(lines)

def generate_templates(
    input_csv: str,
    output_csv: str,
    include_task_hint: bool = True,
) -> str:
    """
    Read the TF pair CSV and write a new CSV with only 'template' and 'jaccard' columns.

    The source CSV must include columns: "tf_a", "tf_b", "jaccard".
    The output CSV will include only "template" and "jaccard" columns.
    """
    required_cols = {"tf_a", "tf_b", "jaccard"}
    df = pd.read_csv(input_csv)

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {sorted(missing)}")

    df["template"] = df.apply(
        lambda r: build_template_text(str(r["tf_a"]), str(r["tf_b"]), include_task_hint),
        axis=1,
    )

    output_df = df[["template", "jaccard"]]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    return output_csv

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate templated input texts for TF pairs CSV."
    )
    parser.add_argument(
        "--input",
        required=False,
        default="/data/rbg/users/seanmurphy/CellPilot/data/datasets/train/chip_tf_jaccard_top150_pairs.csv",
        help="Path to chip_tf_jaccard_top150_pairs.csv",
    )
    parser.add_argument(
        "--output",
        required=False,
        default="/data/rbg/users/seanmurphy/CellPilot/data/datasets/train/tf_pairs_templates_jaccard.csv",
        help="Output CSV path (will include only 'template' and 'jaccard' columns)",
    )
    parser.add_argument(
        "--no-task-hint",
        action="store_true",
        help="Exclude the task hint from the template text",
    )

    args = parser.parse_args()

    output_path = generate_templates(
        input_csv=args.input,
        output_csv=args.output,
        include_task_hint=not args.no_task_hint,
    )
    print(f"Wrote templated CSV to: {output_path}")

if __name__ == "__main__":
    main()
