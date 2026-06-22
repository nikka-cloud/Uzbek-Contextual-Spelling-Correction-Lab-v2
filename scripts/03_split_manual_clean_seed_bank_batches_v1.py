from __future__ import annotations

from pathlib import Path
import math
import pandas as pd


BASE = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1/"
    "strict_clean_seed_extractor_v1_3_lexical_gate/"
    "manual_clean_seed_bank_v1"
)

IN = BASE / "manual_clean_seed_bank_review_v1.csv"
OUT = BASE / "manual_review_batches_v1"
OUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 100

df = pd.read_csv(IN, dtype=str).fillna("")

required = [
    "review_id",
    "source_preview",
    "manual_clean_label",
    "manual_notes",
    "source_record_id",
    "part_number",
    "sentence_id",
    "sentence_text",
]

for col in required:
    if col not in df.columns:
        raise SystemExit(f"Missing required column: {col}")

# Keep the review surface clean and simple.
review_cols = [
    "review_id",
    "manual_clean_label",
    "manual_notes",
    "source_preview",
    "source_record_id",
    "part_number",
    "sentence_id",
    "sentence_text",
]

df = df[review_cols].copy()

n_batches = math.ceil(len(df) / BATCH_SIZE)

paths = []

for i in range(n_batches):
    start = i * BATCH_SIZE
    end = min((i + 1) * BATCH_SIZE, len(df))
    batch = df.iloc[start:end].copy()

    out_path = OUT / f"manual_seed_bank_batch_{i + 1:03d}.csv"
    batch.to_csv(out_path, index=False)
    paths.append(out_path)

summary = []
summary.append("# Manual Clean Seed Bank Batches v1")
summary.append("")
summary.append(f"- input_rows: `{len(df)}`")
summary.append(f"- batch_size: `{BATCH_SIZE}`")
summary.append(f"- batches_written: `{len(paths)}`")
summary.append("")
summary.append("## Labels")
summary.append("")
summary.append("- `CLEAN_TARGET`")
summary.append("- `NOT_CLEAN_TARGET`")
summary.append("")
summary.append("## Batch files")
summary.append("")

for path in paths:
    summary.append(f"- `{path}`")

summary_path = BASE / "manual_review_batches_summary_v1.md"
summary_path.write_text("\n".join(summary), encoding="utf-8")

print("\n".join(summary))
print("\nFIRST BATCH PREVIEW")
print(
    pd.read_csv(paths[0], dtype=str).fillna("")
    .head(40)[
        [
            "review_id",
            "manual_clean_label",
            "manual_notes",
            "source_preview",
            "source_record_id",
            "part_number",
            "sentence_text",
        ]
    ]
    .to_string(index=False)
)
