from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1/"
    "strict_clean_seed_extractor_v1_3_lexical_gate"
)

SOURCES = [
    ("v1_3_fresh_failed_preview", BASE / "fresh_blind_review_sample_v1_3" / "manual_review_fresh_blind_200_v1_3.csv"),
    ("v1_3b_failed_preview", BASE / "fresh_blind_review_sample_v1_3b" / "manual_review_fresh_blind_200_v1_3b.csv"),
    ("v1_3c_failed_preview", BASE / "safe_modern_prose_seed_v1_3c" / "manual_review_safe_modern_prose_200_v1_3c.csv"),
    ("v1_4a_failed_preview", BASE / "document_safe_clean_seed_v1_4a" / "manual_review_document_safe_200_v1_4a.csv"),
    ("v1_4b_failed_preview", BASE / "document_safe_clean_seed_v1_4b" / "manual_review_document_safe_200_v1_4b.csv"),
    ("v1_4c_failed_preview", BASE / "news_style_clean_seed_v1_4c" / "manual_review_news_style_200_v1_4c.csv"),
]

OUT = BASE / "manual_clean_seed_bank_v1"
OUT.mkdir(parents=True, exist_ok=True)

rows = []

for source_name, path in SOURCES:
    if not path.exists():
        print("MISSING:", path)
        continue

    df = pd.read_csv(path, dtype=str).fillna("")
    df["source_preview"] = source_name

    if "sentence_text" not in df.columns:
        print("SKIP no sentence_text:", path)
        continue

    keep_cols = [
        "source_preview",
        "manual_clean_label",
        "manual_notes",
        "source_record_id",
        "document_sample_id",
        "part_number",
        "sentence_id",
        "word_count",
        "char_count",
        "anchor_count",
        "news_anchor_score_v1_4c",
        "sentence_text",
        "local_blockers_v1_4c",
        "candidate_block_reasons_v1_4c",
    ]

    for col in keep_cols:
        if col not in df.columns:
            df[col] = ""

    rows.append(df[keep_cols])

if not rows:
    raise SystemExit("No preview files found.")

all_df = pd.concat(rows, ignore_index=True)

all_df["normalized_sentence_text"] = (
    all_df["sentence_text"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    .str.lower()
)

all_df = all_df.drop_duplicates(subset=["normalized_sentence_text"], keep="first").copy()

# Reset labels because this is a new purpose: not validation, manual seed curation.
all_df["manual_clean_label"] = ""
all_df["manual_notes"] = ""

all_df.insert(0, "review_id", [f"seedbank-v1-{i:05d}" for i in range(1, len(all_df) + 1)])

out_path = OUT / "manual_clean_seed_bank_review_v1.csv"
summary_path = OUT / "manual_clean_seed_bank_summary_v1.md"

all_df.to_csv(out_path, index=False)

summary = []
summary.append("# Manual Clean Seed Bank v1")
summary.append("")
summary.append("## Purpose")
summary.append("")
summary.append("This is not extractor validation. This file is for manually collecting true clean target sentences and known bad examples after heuristic extraction failed.")
summary.append("")
summary.append("## Counts")
summary.append("")
summary.append(f"- review_rows_after_dedup: `{len(all_df)}`")
summary.append("")
summary.append("## Labels")
summary.append("")
summary.append("- `CLEAN_TARGET` = sentence is clean enough to be a gold target for synthetic typo generation.")
summary.append("- `NOT_CLEAN_TARGET` = sentence has OCR, spelling, spacing, merged-word, foreign, archaic, domain-risk, fragment, or grammar damage.")
summary.append("")
summary.append("## Output")
summary.append("")
summary.append(f"- review_file: `{out_path}`")
summary.append("")
summary.append("## Rule")
summary.append("")
summary.append("Use this to build a manually trusted seed bank. Do not call any failed extractor validated.")

summary_path.write_text("\n".join(summary), encoding="utf-8")

print("\n".join(summary))
print("\nFIRST 80 ROWS")
print(
    all_df.head(80)[
        [
            "review_id",
            "source_preview",
            "manual_clean_label",
            "manual_notes",
            "source_record_id",
            "part_number",
            "sentence_text",
        ]
    ].to_string(index=False)
)
