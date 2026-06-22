from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd


if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python3 scripts/03_check_manual_seed_bank_batch_v1.py "
        "<path/to/manual_seed_bank_batch_XXX.csv>"
    )

path = Path(sys.argv[1])

if not path.exists():
    raise SystemExit(f"Missing batch file: {path}")

df = pd.read_csv(path, dtype=str).fillna("")

required = [
    "review_id",
    "manual_clean_label",
    "manual_notes",
    "source_preview",
    "source_record_id",
    "part_number",
    "sentence_id",
    "sentence_text",
]

for col in required:
    if col not in df.columns:
        raise SystemExit(f"Missing required column: {col}")

valid_labels = {"", "CLEAN_TARGET", "NOT_CLEAN_TARGET"}

bad_labels = sorted(set(df["manual_clean_label"]) - valid_labels)

if bad_labels:
    print("BAD LABEL VALUES FOUND:")
    for label in bad_labels:
        print("-", repr(label))
    raise SystemExit("Fix invalid labels first.")

counts = df["manual_clean_label"].value_counts(dropna=False).to_dict()

clean = int((df["manual_clean_label"] == "CLEAN_TARGET").sum())
bad = int((df["manual_clean_label"] == "NOT_CLEAN_TARGET").sum())
pending = int((df["manual_clean_label"] == "").sum())
total = len(df)

precision_like = clean / (clean + bad) if (clean + bad) else 0.0

print("\n# Manual Seed Bank Batch Check")
print()
print(f"- file: `{path}`")
print(f"- rows: `{total}`")
print(f"- CLEAN_TARGET: `{clean}`")
print(f"- NOT_CLEAN_TARGET: `{bad}`")
print(f"- pending: `{pending}`")
print(f"- clean_share_among_labeled: `{precision_like:.2%}`")

print("\n## Label distribution")
for k, v in counts.items():
    print(f"- {repr(k)}: {v}")

if pending:
    print("\nDECISION: PENDING — finish labels before using this batch.")
elif clean == 0:
    print("\nDECISION: NO CLEAN SEEDS IN THIS BATCH.")
elif precision_like < 0.20:
    print("\nDECISION: MOSTLY BAD BANK — useful mainly for negative/risk examples.")
else:
    print("\nDECISION: MIXED BANK — extract CLEAN_TARGET rows into seed bank after review.")

print("\n## First 30 CLEAN_TARGET rows")
print(
    df[df["manual_clean_label"] == "CLEAN_TARGET"]
    .head(30)[["review_id", "source_preview", "source_record_id", "part_number", "sentence_text"]]
    .to_string(index=False)
)

print("\n## First 30 NOT_CLEAN_TARGET rows")
print(
    df[df["manual_clean_label"] == "NOT_CLEAN_TARGET"]
    .head(30)[["review_id", "source_preview", "source_record_id", "part_number", "sentence_text"]]
    .to_string(index=False)
)
