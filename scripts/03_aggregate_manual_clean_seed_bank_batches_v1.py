from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


BASE = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1/"
    "strict_clean_seed_extractor_v1_3_lexical_gate/"
    "manual_clean_seed_bank_v1"
)

BATCH_DIR = BASE / "manual_review_batches_v1"
OUT = BASE / "manual_labeled_outputs_v1"
OUT.mkdir(parents=True, exist_ok=True)

batch_paths = sorted(BATCH_DIR.glob("manual_seed_bank_batch_*.csv"))

if not batch_paths:
    raise SystemExit(f"No batch files found in: {BATCH_DIR}")

frames = []

for path in batch_paths:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["batch_file"] = path.name
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)

required = [
    "review_id",
    "manual_clean_label",
    "manual_notes",
    "source_preview",
    "source_record_id",
    "part_number",
    "sentence_id",
    "sentence_text",
    "batch_file",
]

for col in required:
    if col not in all_df.columns:
        raise SystemExit(f"Missing required column: {col}")

valid_labels = {"", "CLEAN_TARGET", "NOT_CLEAN_TARGET"}
bad_labels = sorted(set(all_df["manual_clean_label"]) - valid_labels)

if bad_labels:
    raise SystemExit(f"Invalid labels found: {bad_labels}")

# Safety: no duplicate review IDs.
dup_review_ids = all_df[all_df["review_id"].duplicated(keep=False)]
if len(dup_review_ids):
    dup_path = OUT / "duplicate_review_ids_v1.csv"
    dup_review_ids.to_csv(dup_path, index=False)
    raise SystemExit(f"Duplicate review_id values found. See: {dup_path}")

clean_df = all_df[all_df["manual_clean_label"] == "CLEAN_TARGET"].copy()
bad_df = all_df[all_df["manual_clean_label"] == "NOT_CLEAN_TARGET"].copy()
pending_df = all_df[all_df["manual_clean_label"] == ""].copy()

all_path = OUT / "manual_clean_seed_bank_labeled_all_v1.csv"
clean_path = OUT / "manual_clean_seed_bank_clean_targets_v1.csv"
bad_path = OUT / "manual_clean_seed_bank_not_clean_targets_v1.csv"
pending_path = OUT / "manual_clean_seed_bank_pending_v1.csv"

all_df.to_csv(all_path, index=False)
clean_df.to_csv(clean_path, index=False)
bad_df.to_csv(bad_path, index=False)
pending_df.to_csv(pending_path, index=False)

# Lightweight sanity flags for CLEAN_TARGET rows.
# These do not automatically reject. They only identify rows worth a second look.
risk_patterns = {
    "lowercase_start": re.compile(r"^[a-zʻʼ'`‘’]"),
    "merged_camelcase": re.compile(r"[a-zʻʼ'`‘’][A-Z][a-z]"),
    "many_repeated_chars": re.compile(r"([A-Za-z])\1\1"),
    "number_word_conversion": re.compile(
        r"\b(bir|ikki|uch|to'rt|besh|olti|yetti|sakkiz|to'qqiz|o'n|yigirma|"
        r"o'ttiz|qirq|ellik|oltmish|yetmish|sakson|to'qson|yuz|ming|million)\b",
        re.IGNORECASE,
    ),
    "figure_or_rasm_fragment": re.compile(r"\brasm\b", re.IGNORECASE),
    "known_ocr_fragment": re.compile(
        r"(minginchi|GEObir|IChKI|UYeFA|Nnnom|O rt|xoslngini|imkonyat|"
        r"kelibchiqqan|ilmfan|boshdanoyoq|sa'yharakat|uyjoy|kiyimbosh)",
        re.IGNORECASE,
    ),
    "very_long_token": re.compile(r"\b[A-Za-zʻʼ'`‘’]{22,}\b"),
}

flag_rows = []

for _, row in clean_df.iterrows():
    text = row["sentence_text"]
    flags = [name for name, pat in risk_patterns.items() if pat.search(text)]
    if flags:
        item = row.to_dict()
        item["clean_sanity_flags"] = "|".join(flags)
        flag_rows.append(item)

flags_df = pd.DataFrame(flag_rows)
flags_path = OUT / "manual_clean_targets_sanity_flags_v1.csv"
flags_df.to_csv(flags_path, index=False)

summary = []
summary.append("# Manual Clean Seed Bank Labeled Outputs v1")
summary.append("")
summary.append("## Overall counts")
summary.append("")
summary.append(f"- total_rows: `{len(all_df)}`")
summary.append(f"- CLEAN_TARGET: `{len(clean_df)}`")
summary.append(f"- NOT_CLEAN_TARGET: `{len(bad_df)}`")
summary.append(f"- pending: `{len(pending_df)}`")
summary.append("")
summary.append("## Counts by batch")
summary.append("")

batch_counts = (
    all_df
    .pivot_table(
        index="batch_file",
        columns="manual_clean_label",
        values="review_id",
        aggfunc="count",
        fill_value=0,
    )
    .reset_index()
)

# Normalize blank column name for markdown display.
batch_counts.columns = ["PENDING" if c == "" else c for c in batch_counts.columns]

summary.append(batch_counts.to_markdown(index=False))
summary.append("")
summary.append("## Outputs")
summary.append("")
summary.append(f"- all_labeled_file: `{all_path}`")
summary.append(f"- clean_targets_file: `{clean_path}`")
summary.append(f"- not_clean_targets_file: `{bad_path}`")
summary.append(f"- pending_file: `{pending_path}`")
summary.append(f"- clean_sanity_flags_file: `{flags_path}`")
summary.append("")
summary.append("## Sanity note")
summary.append("")
summary.append(
    "Rows in `manual_clean_targets_sanity_flags_v1.csv` are not automatically wrong. "
    "They are CLEAN_TARGET rows that should be double-checked because they match risky surface patterns."
)

summary_path = OUT / "manual_clean_seed_bank_labeled_summary_v1.md"
summary_path.write_text("\n".join(summary), encoding="utf-8")

print("\n".join(summary))

print("\nFIRST 30 CLEAN_TARGET ROWS")
print(
    clean_df.head(30)[
        ["review_id", "batch_file", "source_record_id", "part_number", "sentence_text"]
    ].to_string(index=False)
)

print("\nFIRST 30 FLAGGED CLEAN_TARGET ROWS")
if len(flags_df):
    print(
        flags_df.head(30)[
            ["review_id", "batch_file", "clean_sanity_flags", "sentence_text"]
        ].to_string(index=False)
    )
else:
    print("No sanity flags found among CLEAN_TARGET rows.")
