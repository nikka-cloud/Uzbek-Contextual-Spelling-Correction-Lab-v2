from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


SRC_DIR = Path("outputs/05_synthetic_typo_generation/v11_realistic_multivariant")
SRC_OUTPUT = SRC_DIR / "token_generator_v11_output.csv"
SRC_AUDIT = SRC_DIR / "token_generator_v11_generation_audit.csv"

OUT_DIR = Path("outputs/05_synthetic_typo_generation/v11_1_realistic_core")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORE_OUT = OUT_DIR / "token_generator_v11_1_core_output.csv"
REMOVED_OUT = OUT_DIR / "removed_normalization_apostrophe_rows_v11_1.csv"
CORE_AUDIT_OUT = OUT_DIR / "token_generator_v11_1_core_generation_audit.csv"
READABLE_OUT = OUT_DIR / "token_generator_v11_1_core_readable_review.csv"
SUMMARY_OUT = OUT_DIR / "token_generator_v11_1_core_summary.md"


REMOVE_RULES = {
    "apostrophe_ascii_to_curly_left",
    "apostrophe_ascii_to_backtick",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not SRC_OUTPUT.exists():
        raise FileNotFoundError(SRC_OUTPUT)
    if not SRC_AUDIT.exists():
        raise FileNotFoundError(SRC_AUDIT)

    df = pd.read_csv(SRC_OUTPUT)
    audit = pd.read_csv(SRC_AUDIT)

    before_rows = len(df)

    removed = df[df["typo_rule"].isin(REMOVE_RULES)].copy()
    core = df[~df["typo_rule"].isin(REMOVE_RULES)].copy()

    # Token-position key: one clean sentence + one token position.
    key_cols = ["review_id", "target", "token_index", "clean_token"]

    original_positions = df[key_cols].drop_duplicates()
    core_positions = core[key_cols].drop_duplicates()

    original_position_count = len(original_positions)
    core_position_count = len(core_positions)

    missing_after_filter = (
        original_positions
        .merge(core_positions, on=key_cols, how="left", indicator=True)
        .query("_merge == 'left_only'")
        .drop(columns=["_merge"])
    )

    # Core variant count per token position.
    core_counts = (
        core.groupby(key_cols, dropna=False)
        .size()
        .reset_index(name="generated_variants_core")
    )

    core_audit = (
        original_positions
        .merge(core_counts, on=key_cols, how="left")
        .fillna({"generated_variants_core": 0})
    )

    core_audit["generated_variants_core"] = core_audit["generated_variants_core"].astype(int)
    core_audit["covered_after_filter"] = core_audit["generated_variants_core"] >= 1

    core.to_csv(CORE_OUT, index=False)
    removed.to_csv(REMOVED_OUT, index=False)
    core_audit.to_csv(CORE_AUDIT_OUT, index=False)

    readable_cols = [
        "row_id",
        "target",
        "input",
        "token_index",
        "clean_token",
        "corrupted_token",
        "typo_family",
        "typo_rule",
        "variant_rank",
    ]

    readable = core[readable_cols].copy()
    readable["change"] = (
        "TOKEN "
        + readable["token_index"].astype(str)
        + ": "
        + readable["clean_token"].astype(str)
        + " → "
        + readable["corrupted_token"].astype(str)
    )
    readable.to_csv(READABLE_OUT, index=False)

    family_counts = core["typo_family"].value_counts()
    rule_counts = core["typo_rule"].value_counts()

    removed_rule_counts = removed["typo_rule"].value_counts()

    summary = []
    summary.append("# Token Generator v11.1 Realistic Core")
    summary.append("")
    summary.append("## Purpose")
    summary.append("")
    summary.append("Create model-facing realistic typo data from v11 by removing deterministic apostrophe-normalization variants.")
    summary.append("")
    summary.append("Removed rules:")
    for r in sorted(REMOVE_RULES):
        summary.append(f"- `{r}`")
    summary.append("")
    summary.append("Reason:")
    summary.append("- Frozen clean targets are already normalized.")
    summary.append("- Curly/backtick apostrophe variants should be handled by deterministic normalization, not learned as contextual spelling errors.")
    summary.append("- Apostrophe-drop rows are kept because missing apostrophes are real spelling/corpus corruption.")
    summary.append("")
    summary.append("## Counts")
    summary.append(f"- original v11 rows: `{before_rows}`")
    summary.append(f"- removed rows: `{len(removed)}`")
    summary.append(f"- v11.1 core rows: `{len(core)}`")
    summary.append(f"- original eligible token positions: `{original_position_count}`")
    summary.append(f"- covered token positions after filtering: `{core_position_count}`")
    summary.append(f"- missing token positions after filtering: `{len(missing_after_filter)}`")
    summary.append("")
    summary.append("## Removed rule counts")
    for k, v in removed_rule_counts.items():
        summary.append(f"- {k}: `{v}`")
    summary.append("")
    summary.append("## Core typo family counts")
    for k, v in family_counts.items():
        summary.append(f"- {k}: `{v}`")
    summary.append("")
    summary.append("## Core typo rule counts")
    for k, v in rule_counts.items():
        summary.append(f"- {k}: `{v}`")
    summary.append("")
    summary.append("## Files")
    summary.append(f"- core output: `{CORE_OUT}`")
    summary.append(f"- removed rows: `{REMOVED_OUT}`")
    summary.append(f"- core audit: `{CORE_AUDIT_OUT}`")
    summary.append(f"- readable review: `{READABLE_OUT}`")
    summary.append("")
    summary.append("## SHA256")
    summary.append(f"- core output: `{sha256_file(CORE_OUT)}`")
    summary.append(f"- removed rows: `{sha256_file(REMOVED_OUT)}`")
    summary.append(f"- core audit: `{sha256_file(CORE_AUDIT_OUT)}`")
    summary.append(f"- readable review: `{sha256_file(READABLE_OUT)}`")

    SUMMARY_OUT.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n".join(summary))

    if len(missing_after_filter) == 0:
        print("\nV11.1 CORE COVERAGE AFTER FILTER: PASS")
    else:
        print("\nV11.1 CORE COVERAGE AFTER FILTER: FAIL")
        print(missing_after_filter.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
