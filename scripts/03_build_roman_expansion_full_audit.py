#!/usr/bin/env python3
"""
Build a complete manual-review inventory for every detected
ROMAN_EXPANSION_NEAR_INITIAL boundary in the Phase 3 10K experiment.

This script does not modify:
- the source corpus;
- the Phase 2 normalized corpus;
- the sentence inventory;
- the sentence splitter.

It only combines existing evidence and creates deterministic review
files for all 84 detected Roman-expansion boundary candidates.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(
    "outputs/03_sentence_inventory/segment_10k_v1"
)

FAMILY_AUDIT_PATH = (
    ROOT / "lowercase_split_family_audit_v1.csv"
)

FOCUSED_REVIEW_PATH = (
    ROOT / "focused_rule_review_v1.csv"
)

FULL_OUTPUT_PATH = (
    ROOT / "roman_expansion_full_audit_v1.csv"
)

BATCH_OUTPUT_DIR = (
    ROOT / "roman_expansion_review_batches_v1"
)

TEMP_BATCH_DIR = (
    ROOT / "roman_expansion_review_batches_v1.tmp"
)

EXPECTED_TOTAL = 84
EXPECTED_ALREADY_REVIEWED = 25
BATCH_SIZE = 20


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def event_key(row: dict[str, str]) -> tuple[int, int]:
    return (
        int(row["source_record_id"]),
        int(row["sentence_index_in_record"]),
    )


def assign_triage_group(
    family_labels: set[str],
) -> str:
    """
    Group pending rows for easier human inspection.

    These groups are organizational only.
    They are not automatic linguistic labels.
    """
    if "SHORT_OR_METADATA_CONTEXT" in family_labels:
        return "SHORT_OR_METADATA_CONTEXT"

    if "NUMBER_LED_NEXT_SEGMENT" in family_labels:
        return "NUMBER_LED_CONTEXT"

    if "LOWERCASE_WORD_PAIR_CANDIDATE" in family_labels:
        return "LOWERCASE_PROSE_CONTINUATION"

    return "OTHER_CONTEXT"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    family_rows = read_csv(FAMILY_AUDIT_PATH)
    focused_rows = read_csv(FOCUSED_REVIEW_PATH)

    roman_rows = []

    for row in family_rows:
        families = set(
            row["family_labels"].split("|")
        )

        if "ROMAN_EXPANSION_NEAR_INITIAL" in families:
            roman_rows.append(row)

    if len(roman_rows) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} Roman-expansion events, "
            f"found {len(roman_rows)}."
        )

    focused_roman_rows = [
        row
        for row in focused_rows
        if (
            row["review_group"]
            == "ROMAN_EXPANSION_NEAR_INITIAL"
        )
    ]

    if (
        len(focused_roman_rows)
        != EXPECTED_ALREADY_REVIEWED
    ):
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_ALREADY_REVIEWED} existing Roman reviews, "
            f"found {len(focused_roman_rows)}."
        )

    focused_by_key: dict[
        tuple[int, int],
        dict[str, str],
    ] = {}

    for row in focused_roman_rows:
        key = event_key(row)

        if key in focused_by_key:
            raise RuntimeError(
                f"Duplicate focused-review key: {key}"
            )

        focused_by_key[key] = row

    full_rows: list[dict[str, Any]] = []

    for row in roman_rows:
        key = event_key(row)
        families = set(
            row["family_labels"].split("|")
        )
        focused = focused_by_key.get(key)

        if focused is None:
            review_status = "PENDING"
            review_decision = ""
            rule_eligible = ""
            review_note = ""
        else:
            review_status = "REVIEWED"
            review_decision = focused[
                "review_decision"
            ]
            rule_eligible = focused[
                "rule_eligible"
            ]
            review_note = focused[
                "review_note"
            ]

        full_rows.append(
            {
                "review_status": review_status,
                "triage_group": assign_triage_group(
                    families
                ),
                "source_record_id": (
                    row["source_record_id"]
                ),
                "sentence_index_in_record": (
                    row["sentence_index_in_record"]
                ),
                "sentence_id": row["sentence_id"],
                "family_labels": row["family_labels"],
                "boundary_marker": (
                    row["boundary_marker"]
                ),
                "left_last_token": (
                    row["left_last_token"]
                ),
                "right_first_token": (
                    row["right_first_token"]
                ),
                "parent_review_flags": (
                    row["parent_review_flags"]
                ),
                "left_context": row["left_context"],
                "right_context": row["right_context"],
                "review_decision": review_decision,
                "rule_eligible": rule_eligible,
                "review_note": review_note,
            }
        )

    full_rows.sort(
        key=lambda row: (
            row["review_status"] != "PENDING",
            row["triage_group"],
            int(row["source_record_id"]),
            int(row["sentence_index_in_record"]),
        )
    )

    fieldnames = [
        "review_status",
        "triage_group",
        "source_record_id",
        "sentence_index_in_record",
        "sentence_id",
        "family_labels",
        "boundary_marker",
        "left_last_token",
        "right_first_token",
        "parent_review_flags",
        "left_context",
        "right_context",
        "review_decision",
        "rule_eligible",
        "review_note",
    ]

    temporary_full_path = (
        FULL_OUTPUT_PATH.with_suffix(".tmp")
    )

    write_csv(
        temporary_full_path,
        full_rows,
        fieldnames,
    )

    temporary_full_path.replace(
        FULL_OUTPUT_PATH
    )

    pending_rows = [
        row
        for row in full_rows
        if row["review_status"] == "PENDING"
    ]

    reviewed_rows = [
        row
        for row in full_rows
        if row["review_status"] == "REVIEWED"
    ]

    if (
        len(reviewed_rows)
        != EXPECTED_ALREADY_REVIEWED
    ):
        raise RuntimeError(
            "Reviewed-row accounting failed."
        )

    if (
        len(pending_rows)
        != EXPECTED_TOTAL
        - EXPECTED_ALREADY_REVIEWED
    ):
        raise RuntimeError(
            "Pending-row accounting failed."
        )

    if TEMP_BATCH_DIR.exists():
        shutil.rmtree(TEMP_BATCH_DIR)

    TEMP_BATCH_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    batch_paths = []

    for start in range(
        0,
        len(pending_rows),
        BATCH_SIZE,
    ):
        batch_rows = pending_rows[
            start:start + BATCH_SIZE
        ]

        batch_number = (
            start // BATCH_SIZE
        ) + 1

        batch_path = (
            TEMP_BATCH_DIR
            / f"batch-{batch_number:02d}.csv"
        )

        write_csv(
            batch_path,
            batch_rows,
            fieldnames,
        )

        batch_paths.append(batch_path)

    if BATCH_OUTPUT_DIR.exists():
        shutil.rmtree(BATCH_OUTPUT_DIR)

    TEMP_BATCH_DIR.rename(
        BATCH_OUTPUT_DIR
    )

    review_status_counts = Counter(
        row["review_status"]
        for row in full_rows
    )

    triage_counts = Counter(
        row["triage_group"]
        for row in pending_rows
    )

    existing_decision_counts = Counter(
        row["review_decision"]
        for row in reviewed_rows
    )

    print("\n--- Roman-expansion full audit ---")
    print(f"total_events: {len(full_rows)}")
    print(
        "review_status_counts:",
        dict(review_status_counts),
    )
    print(
        "pending_triage_counts:",
        dict(sorted(triage_counts.items())),
    )
    print(
        "existing_decision_counts:",
        dict(existing_decision_counts),
    )
    print(
        "batch_sizes:",
        [
            sum(
                1
                for _ in path.open(
                    "r",
                    encoding="utf-8",
                )
            ) - 1
            for path in sorted(
                BATCH_OUTPUT_DIR.glob(
                    "batch-*.csv"
                )
            )
        ],
    )

    print("\n--- outputs ---")
    print(f"full_audit: {FULL_OUTPUT_PATH}")
    print(f"review_batches: {BATCH_OUTPUT_DIR}")
    print(
        "full_audit_sha256:",
        file_sha256(FULL_OUTPUT_PATH),
    )

    print("\n--- Batch 1 preview ---")

    first_batch_path = (
        BATCH_OUTPUT_DIR / "batch-01.csv"
    )
    first_batch_rows = read_csv(
        first_batch_path
    )

    for number, row in enumerate(
        first_batch_rows[:12],
        start=1,
    ):
        print("\n" + "=" * 110)
        print(f"ROW: {number}")
        print(
            f"source={row['source_record_id']} "
            f"sentence_index="
            f"{row['sentence_index_in_record']}"
        )
        print(
            f"triage_group={row['triage_group']}"
        )
        print(
            f"families={row['family_labels']}"
        )
        print("\nBOUNDARY")
        print(
            row["left_context"]
            + " "
            + f"[{row['boundary_marker']}]"
            + " "
            + row["right_context"]
        )

    print("\nAllowed review decisions:")
    print("PROTECT_BOUNDARY")
    print("KEEP_SPLIT")
    print("SOURCE_TOO_CORRUPTED")
    print("UNCERTAIN")

    print("\nNo segmentation rule was changed.")


if __name__ == "__main__":
    main()
