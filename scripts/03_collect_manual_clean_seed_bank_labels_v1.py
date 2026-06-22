from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1/"
    "strict_clean_seed_extractor_v1_3_lexical_gate/"
    "manual_clean_seed_bank_v1"
)

BATCH_DIR = BASE / "manual_review_batches_v1"

OUT_ALL = BASE / "manual_clean_seed_bank_labeled_all_v1.csv"
OUT_CLEAN = BASE / "manual_clean_seed_bank_clean_targets_v1.csv"
OUT_BAD = BASE / "manual_clean_seed_bank_not_clean_targets_v1.csv"
OUT_REPORT = BASE / "manual_clean_seed_bank_label_report_v1.md"

valid_labels = {"", "CLEAN_TARGET", "NOT_CLEAN_TARGET"}

batch_paths = sorted(BATCH_DIR.glob("manual_seed_bank_batch_*.csv"))

if not batch_paths:
    raise SystemExit(f"No batch files found in: {BATCH_DIR}")

frames = []

for path in batch_paths:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["batch_file"] = path.name

    if "manual_clean_label" not in df.columns:
        raise SystemExit(f"Missing manual_clean_label in: {path}")

    bad_labels = sorted(set(df["manual_clean_label"]) - valid_labels)
    if bad_labels:
        raise SystemExit(f"Invalid labels in {path}: {bad_labels}")

    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)

# Deduplicate by review_id just in case.
before = len(all_df)
all_df = all_df.drop_duplicates(subset=["review_id"], keep="last").copy()
after = len(all_df)

clean_df = all_df[all_df["manual_clean_label"] == "CLEAN_TARGET"].copy()
bad_df = all_df[all_df["manual_clean_label"] == "NOT_CLEAN_TARGET"].copy()
pending_df = all_df[all_df["manual_clean_label"] == ""].copy()

all_df.to_csv(OUT_ALL, index=False)
clean_df.to_csv(OUT_CLEAN, index=False)
bad_df.to_csv(OUT_BAD, index=False)

lines = []
lines.append("# Manual Clean Seed Bank Label Report v1")
lines.append("")
lines.append("## Purpose")
lines.append("")
lines.append("Collect manual labels from seed-bank review batches.")
lines.append("")
lines.append("## Counts")
lines.append("")
lines.append(f"- batch_files: `{len(batch_paths)}`")
lines.append(f"- rows_before_dedup: `{before}`")
lines.append(f"- rows_after_dedup: `{after}`")
lines.append(f"- CLEAN_TARGET: `{len(clean_df)}`")
lines.append(f"- NOT_CLEAN_TARGET: `{len(bad_df)}`")
lines.append(f"- pending_unlabeled: `{len(pending_df)}`")
lines.append("")
lines.append("## Output files")
lines.append("")
lines.append(f"- all_labeled_file: `{OUT_ALL}`")
lines.append(f"- clean_targets_file: `{OUT_CLEAN}`")
lines.append(f"- not_clean_targets_file: `{OUT_BAD}`")
lines.append("")
lines.append("## Decision rule")
lines.append("")
if len(pending_df) > 0:
    lines.append("Status: `PARTIAL_LABELING_IN_PROGRESS`")
    lines.append("")
    lines.append("Only labeled rows are safe to use. Pending rows must not be used yet.")
else:
    lines.append("Status: `FULLY_LABELED`")
    lines.append("")
    lines.append("All rows have manual labels.")

OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))

print("\nFIRST CLEAN TARGETS")
print(
    clean_df.head(30)[
        ["review_id", "batch_file", "source_preview", "source_record_id", "part_number", "sentence_text"]
    ].to_string(index=False)
)

print("\nFIRST NOT CLEAN TARGETS")
print(
    bad_df.head(30)[
        ["review_id", "batch_file", "source_preview", "source_record_id", "part_number", "sentence_text"]
    ].to_string(index=False)
)
