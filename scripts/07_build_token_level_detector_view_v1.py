from __future__ import annotations

import csv
import re
from pathlib import Path


INPUT_PATH = Path("outputs/07_training_views/detector_training_view_v1/detector_training_view_all_v1.csv")

OUT_DIR = Path("outputs/07_training_views/token_level_detector_view_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_OUT = OUT_DIR / "token_level_detector_all_v1.csv"
TRAIN_OUT = OUT_DIR / "token_level_detector_train_v1.csv"
DEV_OUT = OUT_DIR / "token_level_detector_dev_v1.csv"
TEST_OUT = OUT_DIR / "token_level_detector_test_v1.csv"
SUMMARY_OUT = OUT_DIR / "token_level_detector_summary_v1.md"

TOKEN_RE = re.compile(r"\S+")


def tokenize_with_spans(text: str):
    return [
        {
            "token_index": i,
            "token": m.group(0),
            "start_char": m.start(),
            "end_char": m.end(),
        }
        for i, m in enumerate(TOKEN_RE.finditer(text))
    ]


def main():
    with INPUT_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []

    for r in rows:
        tokens = tokenize_with_spans(r["input"])

        corrupted_index = None
        if r["label"] == "CORRECT_THIS":
            corrupted_index = int(r["token_index"])

        for t in tokens:
            token_label = "KEEP"

            if r["label"] == "CORRECT_THIS" and t["token_index"] == corrupted_index:
                token_label = "CORRECT_THIS"

            out_rows.append({
                "parent_row_id": r["row_id"],
                "group_id": r["group_id"],
                "split": r["split"],
                "input": r["input"],
                "target": r["target"],
                "token_index": t["token_index"],
                "token": t["token"],
                "token_start_char": t["start_char"],
                "token_end_char": t["end_char"],
                "label": token_label,
                "sentence_label": r["label"],
                "clean_token": r["clean_token"],
                "observed_token": r["observed_token"],
                "correction": r["correction"],
                "typo_family": r["typo_family"],
                "typo_rule": r["typo_rule"],
                "source": r["source"],
            })

    fieldnames = list(out_rows[0].keys())

    def write(path, data):
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(data)

    write(ALL_OUT, out_rows)
    write(TRAIN_OUT, [r for r in out_rows if r["split"] == "train"])
    write(DEV_OUT, [r for r in out_rows if r["split"] == "dev"])
    write(TEST_OUT, [r for r in out_rows if r["split"] == "test"])

    total = len(out_rows)
    correct = sum(1 for r in out_rows if r["label"] == "CORRECT_THIS")
    keep = sum(1 for r in out_rows if r["label"] == "KEEP")

    summary = f"""# Token-Level Detector View v1

## Counts

- total token rows: `{total}`
- KEEP token rows: `{keep}`
- CORRECT_THIS token rows: `{correct}`

## Files

- all: `{ALL_OUT}`
- train: `{TRAIN_OUT}`
- dev: `{DEV_OUT}`
- test: `{TEST_OUT}`
"""

    SUMMARY_OUT.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
