#!/usr/bin/env python3
"""
Compare Phase 3 sentence inventories by exact character boundaries.

The comparison verifies that v2 differs from v1 only through approved
boundary protections. It does not modify either inventory.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


V1_ROOT = Path(
    "outputs/03_sentence_inventory/segment_10k_v1"
)

V2_ROOT = Path(
    "outputs/03_sentence_inventory/segment_10k_v2"
)

OUTPUT_PATH = (
    V2_ROOT / "v1_v2_boundary_changes.csv"
)

APPROVED_NEW_FLAGS = {
    "DOTTED_NAME_OR_DOMAIN_PROTECTED",
    "ROMAN_EXPANSION_INITIAL_PROTECTED",
}


def load_inventory(
    path: Path,
) -> dict[int, list[dict[str, Any]]]:
    """Load sentence rows grouped by source document."""
    documents: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            row = json.loads(line)

            source_id = row["source_record_id"]
            documents[source_id].append(row)

    for source_id, rows in documents.items():
        rows.sort(
            key=lambda row: (
                row["sentence_start_char"],
                row["sentence_end_char"],
            )
        )

        expected_indices = list(range(len(rows)))
        actual_indices = [
            row["sentence_index_in_record"]
            for row in rows
        ]

        if actual_indices != expected_indices:
            raise RuntimeError(
                "Non-sequential sentence indices for "
                f"source {source_id}: {actual_indices}"
            )

    return dict(documents)


def boundary_positions(
    rows: list[dict[str, Any]],
) -> set[int]:
    """
    Return internal sentence-end positions.

    The final document end is excluded because it is not a boundary
    between two sentence candidates.
    """
    if not rows:
        return set()

    final_end = max(
        row["sentence_end_char"]
        for row in rows
    )

    return {
        row["sentence_end_char"]
        for row in rows
        if row["sentence_end_char"] != final_end
    }


def find_v1_left_row(
    rows: list[dict[str, Any]],
    boundary: int,
) -> dict[str, Any]:
    """Find the v1 sentence ending at the removed boundary."""
    matches = [
        row
        for row in rows
        if row["sentence_end_char"] == boundary
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one v1 sentence ending at "
            f"boundary {boundary}; found {len(matches)}."
        )

    return matches[0]


def find_v1_right_row(
    rows: list[dict[str, Any]],
    boundary: int,
) -> dict[str, Any]:
    """Find the first v1 sentence beginning after the boundary."""
    candidates = [
        row
        for row in rows
        if row["sentence_start_char"] >= boundary
    ]

    if not candidates:
        raise RuntimeError(
            "Could not find the v1 sentence after "
            f"boundary {boundary}."
        )

    return min(
        candidates,
        key=lambda row: row["sentence_start_char"],
    )


def find_v2_spanning_row(
    rows: list[dict[str, Any]],
    boundary: int,
) -> dict[str, Any]:
    """Find the v2 sentence that spans a removed v1 boundary."""
    matches = [
        row
        for row in rows
        if (
            row["sentence_start_char"] < boundary
            < row["sentence_end_char"]
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected one v2 sentence spanning "
            f"boundary {boundary}; found {len(matches)}."
        )

    return matches[0]


def compact(text: str, limit: int = 500) -> str:
    """Create a compact display preview."""
    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    return text[:limit] + "…"


def main() -> None:
    v1_sentences_path = V1_ROOT / "sentences.jsonl"
    v2_sentences_path = V2_ROOT / "sentences.jsonl"

    for path in [
        v1_sentences_path,
        v2_sentences_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    v1 = load_inventory(v1_sentences_path)
    v2 = load_inventory(v2_sentences_path)

    v1_source_ids = set(v1)
    v2_source_ids = set(v2)

    missing_in_v2 = sorted(
        v1_source_ids - v2_source_ids
    )
    extra_in_v2 = sorted(
        v2_source_ids - v1_source_ids
    )

    if missing_in_v2 or extra_in_v2:
        raise RuntimeError(
            "Source-document coverage differs.\n"
            f"Missing in v2: {missing_in_v2[:20]}\n"
            f"Extra in v2: {extra_in_v2[:20]}"
        )

    removed_boundary_count = 0
    added_boundary_count = 0
    changed_document_count = 0
    unexplained_removed_count = 0

    trigger_counts = Counter()
    removed_per_document = Counter()
    changed_rows: list[dict[str, Any]] = []

    for source_id in sorted(v1_source_ids):
        v1_rows = v1[source_id]
        v2_rows = v2[source_id]

        if not v1_rows or not v2_rows:
            raise RuntimeError(
                f"Zero-sentence document found: {source_id}"
            )

        v1_document_span = (
            v1_rows[0]["sentence_start_char"],
            v1_rows[-1]["sentence_end_char"],
        )
        v2_document_span = (
            v2_rows[0]["sentence_start_char"],
            v2_rows[-1]["sentence_end_char"],
        )

        if v1_document_span != v2_document_span:
            raise RuntimeError(
                "Document coverage changed for source "
                f"{source_id}: "
                f"v1={v1_document_span}, "
                f"v2={v2_document_span}"
            )

        v1_boundaries = boundary_positions(v1_rows)
        v2_boundaries = boundary_positions(v2_rows)

        removed = sorted(
            v1_boundaries - v2_boundaries
        )
        added = sorted(
            v2_boundaries - v1_boundaries
        )

        if removed or added:
            changed_document_count += 1

        removed_boundary_count += len(removed)
        added_boundary_count += len(added)
        removed_per_document[source_id] = len(removed)

        for boundary in removed:
            left_row = find_v1_left_row(
                v1_rows,
                boundary,
            )
            right_row = find_v1_right_row(
                v1_rows,
                boundary,
            )
            merged_row = find_v2_spanning_row(
                v2_rows,
                boundary,
            )

            flags = set(
                merged_row["boundary_flags"]
            )

            approved_flags = sorted(
                flags & APPROVED_NEW_FLAGS
            )

            if not approved_flags:
                unexplained_removed_count += 1
                trigger = "UNEXPLAINED"
            else:
                trigger = "|".join(approved_flags)

            trigger_counts[trigger] += 1

            changed_rows.append(
                {
                    "source_record_id": source_id,
                    "removed_boundary_position": (
                        boundary
                    ),
                    "trigger": trigger,
                    "v1_left_sentence_index": (
                        left_row[
                            "sentence_index_in_record"
                        ]
                    ),
                    "v1_right_sentence_index": (
                        right_row[
                            "sentence_index_in_record"
                        ]
                    ),
                    "v2_sentence_index": (
                        merged_row[
                            "sentence_index_in_record"
                        ]
                    ),
                    "v2_boundary_flags": "|".join(
                        merged_row["boundary_flags"]
                    ),
                    "parent_review_flags": "|".join(
                        merged_row[
                            "parent_review_flags"
                        ]
                    ),
                    "v1_left_text": compact(
                        left_row["sentence_text"]
                    ),
                    "v1_right_text": compact(
                        right_row["sentence_text"]
                    ),
                    "v2_merged_text": compact(
                        merged_row["sentence_text"]
                    ),
                }
            )

        for boundary in added:
            changed_rows.append(
                {
                    "source_record_id": source_id,
                    "removed_boundary_position": (
                        boundary
                    ),
                    "trigger": (
                        "UNEXPECTED_ADDED_BOUNDARY"
                    ),
                    "v1_left_sentence_index": "",
                    "v1_right_sentence_index": "",
                    "v2_sentence_index": "",
                    "v2_boundary_flags": "",
                    "parent_review_flags": "",
                    "v1_left_text": "",
                    "v1_right_text": "",
                    "v2_merged_text": "",
                }
            )

    v1_sentence_count = sum(
        len(rows)
        for rows in v1.values()
    )
    v2_sentence_count = sum(
        len(rows)
        for rows in v2.values()
    )

    expected_removed_difference = (
        v1_sentence_count - v2_sentence_count
    )

    fieldnames = [
        "source_record_id",
        "removed_boundary_position",
        "trigger",
        "v1_left_sentence_index",
        "v1_right_sentence_index",
        "v2_sentence_index",
        "v2_boundary_flags",
        "parent_review_flags",
        "v1_left_text",
        "v1_right_text",
        "v2_merged_text",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(changed_rows)

    print("\n--- Phase 3 v1 versus v2 comparison ---")
    print(f"source_documents: {len(v1_source_ids):,}")
    print(f"v1_sentences: {v1_sentence_count:,}")
    print(f"v2_sentences: {v2_sentence_count:,}")
    print(
        "sentence_count_difference: "
        f"{expected_removed_difference:,}"
    )
    print(
        "changed_documents: "
        f"{changed_document_count:,}"
    )
    print(
        "removed_boundaries: "
        f"{removed_boundary_count:,}"
    )
    print(
        "added_boundaries: "
        f"{added_boundary_count:,}"
    )
    print(
        "unexplained_removed_boundaries: "
        f"{unexplained_removed_count:,}"
    )
    print(
        "trigger_counts:",
        dict(sorted(trigger_counts.items())),
    )
    print(f"output_csv: {OUTPUT_PATH}")

    print("\n--- documents with most removed boundaries ---")
    for source_id, count in removed_per_document.most_common(15):
        if count == 0:
            break

        print(
            f"source={source_id:<10} "
            f"removed={count}"
        )

    print("\n--- first changed boundaries ---")
    for row in changed_rows[:12]:
        print("\n" + "=" * 100)
        print(
            f"source={row['source_record_id']} "
            f"position="
            f"{row['removed_boundary_position']} "
            f"trigger={row['trigger']}"
        )
        print(
            "V1 LEFT:  "
            + row["v1_left_text"]
        )
        print(
            "V1 RIGHT: "
            + row["v1_right_text"]
        )
        print(
            "V2 MERGED:"
            + row["v2_merged_text"]
        )

    problems = []

    if added_boundary_count != 0:
        problems.append(
            "V2 introduced new boundaries."
        )

    if (
        removed_boundary_count
        != expected_removed_difference
    ):
        problems.append(
            "Removed-boundary count does not match "
            "the sentence-count difference."
        )

    if unexplained_removed_count != 0:
        problems.append(
            "Some removed boundaries lack an approved "
            "new protection flag."
        )

    if problems:
        print("\nCOMPARISON STATUS: FAIL")

        for problem in problems:
            print(f"- {problem}")

        raise SystemExit(1)

    print("\nCOMPARISON STATUS: PASS")


if __name__ == "__main__":
    main()
