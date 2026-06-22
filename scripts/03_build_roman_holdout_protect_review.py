#!/usr/bin/env python3
"""
Build complete review batches for every holdout event predicted as
PROTECT_BOUNDARY by the conservative Roman-expansion candidate rule.

This script does not modify the corpus, inventories or segmenter.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(
    "outputs/03_sentence_inventory/"
    "roman_rule_holdout_10k_v1"
)

INPUT_PATH = ROOT / "all_candidates.csv"
METRICS_PATH = ROOT / "metrics.json"

OUTPUT_PATH = (
    ROOT / "roman_holdout_protect_full_audit_v1.csv"
)

BATCH_ROOT = (
    ROOT / "roman_holdout_protect_review_batches_v1"
)

BATCH_SIZE = 20

INITIAL_RE = re.compile(
    r"\b(?:[A-Z]|Sh|SH|Ch|CH)\."
)

ORDINAL_DAMAGE_RE = re.compile(
    r"(?:"
    r"birinchi|beshinchi|o'ninchi|"
    r"yuzinchi|minginchi"
    r")[A-Za-zʻʼ‘’']+",
    re.IGNORECASE,
)

REFERENCE_HINT_RE = re.compile(
    r"\b(?:"
    r"professor|akademik|akademigi|"
    r"muallif|mualliflar|taqrizchi|"
    r"redaktor|nashriyot|adabiyot|"
    r"metodika|darslik|kitob|"
    r"tom|bet|sahifa|formula|jadval"
    r")\b",
    re.IGNORECASE,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    if not rows:
        raise RuntimeError(
            f"Refusing to write empty CSV: {path}"
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def build_review_hints(row: dict[str, str]) -> str:
    combined = (
        f"{row['left_context']} "
        f"{row['right_context']}"
    )

    hints: list[str] = []

    initial_count = len(
        INITIAL_RE.findall(combined)
    )

    comma_count = combined.count(",")

    if initial_count >= 3:
        hints.append("MANY_PERSON_INITIALS")

    if comma_count >= 5:
        hints.append("LIST_LIKE_PUNCTUATION")

    if REFERENCE_HINT_RE.search(combined):
        hints.append("REFERENCE_OR_METADATA_TERM")

    if ORDINAL_DAMAGE_RE.search(combined):
        hints.append("JOINED_ORDINAL_DAMAGE_ANYWHERE")

    if int(row["right_token_count"]) < 15:
        hints.append("SHORT_RIGHT_CONTEXT")

    return "|".join(hints)


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}"
        )

    if BATCH_ROOT.exists():
        raise FileExistsError(
            f"Batch directory already exists: "
            f"{BATCH_ROOT}"
        )

    metrics = json.loads(
        METRICS_PATH.read_text(encoding="utf-8")
    )

    rows = read_csv(INPUT_PATH)

    protect_rows = [
        row
        for row in rows
        if row["rule_prediction"]
        == "PROTECT_BOUNDARY"
    ]

    expected_count = int(
        metrics["prediction_counts"][
            "PROTECT_BOUNDARY"
        ]
    )

    if len(protect_rows) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} protect rows, "
            f"found {len(protect_rows)}"
        )

    protect_rows.sort(
        key=lambda row: (
            int(row["source_record_id"]),
            int(
                row[
                    "boundary_parent_char_position"
                ]
            ),
            row["event_id"],
        )
    )

    review_rows: list[dict[str, str]] = []

    for row in protect_rows:
        review_rows.append(
            {
                "review_status": "PENDING",
                "human_decision": "",
                "review_note": "",
                "review_hints": build_review_hints(
                    row
                ),
                **row,
            }
        )

    event_ids = [
        row["event_id"]
        for row in review_rows
    ]

    if len(event_ids) != len(set(event_ids)):
        raise RuntimeError(
            "Duplicate event IDs found."
        )

    write_csv(OUTPUT_PATH, review_rows)

    BATCH_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    batch_sizes: list[int] = []

    for batch_index, start in enumerate(
        range(0, len(review_rows), BATCH_SIZE),
        start=1,
    ):
        batch = review_rows[
            start:start + BATCH_SIZE
        ]

        batch_path = (
            BATCH_ROOT
            / f"batch-{batch_index:02d}.csv"
        )

        write_csv(batch_path, batch)
        batch_sizes.append(len(batch))

    hint_counts = Counter()

    for row in review_rows:
        for hint in row["review_hints"].split("|"):
            if hint:
                hint_counts[hint] += 1

    print(
        "\n--- complete holdout protect audit ---"
    )
    print(f"total_rows: {len(review_rows)}")
    print(f"unique_event_ids: {len(set(event_ids))}")
    print(f"batch_sizes: {batch_sizes}")
    print(f"review_hint_counts: {dict(hint_counts)}")
    print(f"full_audit: {OUTPUT_PATH}")
    print(f"review_batches: {BATCH_ROOT}")

    print("\n--- Batch 1 preview ---")

    for number, row in enumerate(
        review_rows[:12],
        start=1,
    ):
        print("\n" + "=" * 105)
        print(
            f"ROW: {number} | "
            f"source={row['source_record_id']} | "
            f"position="
            f"{row['boundary_parent_char_position']}"
        )
        print(
            "review_hints:",
            row["review_hints"] or "NONE",
        )
        print(
            f"{row['left_context']} "
            f"[.] {row['right_context']}"
        )

    print("\nAllowed human decisions:")
    print("PROTECT_BOUNDARY")
    print("KEEP_SPLIT")
    print("SOURCE_TOO_CORRUPTED")
    print("UNCERTAIN")

    print("\nNo data or segmenter was modified.")


if __name__ == "__main__":
    main()
