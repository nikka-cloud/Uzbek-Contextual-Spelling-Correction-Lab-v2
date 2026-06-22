#!/usr/bin/env python3

"""
Build a deterministic, population-proportional document-sampling
manifest across the complete curated corpus.

This script does not read document text and does not modify the corpus.

The selected document positions will later be extracted and segmented
for an unseen global corpus-quality validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import math
import random


ROOT = Path(__file__).resolve().parents[1]

CENSUS_PATH = (
    ROOT
    / "outputs"
    / "03_sentence_inventory"
    / "full_corpus_part_census_v1"
    / "full_corpus_part_census_v1.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "03_sentence_inventory"
    / "global_validation_document_sample_v1"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "global_validation_document_manifest_v1.csv"
)

ALLOCATION_PATH = (
    OUTPUT_DIR
    / "global_validation_part_allocation_v1.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "global_validation_document_summary_v1.json"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "global_validation_document_report_v1.md"
)


RANDOM_SEED = 20260617
TARGET_DOCUMENTS = 6_950

# The previous segment_10k_v2 pilot used local rows 0–999
# from these ten parts. Excluding those rows guarantees that
# the new global sample does not reuse pilot documents.
OLD_PILOT_PARTS = {
    0,
    2,
    5,
    10,
    20,
    50,
    80,
    110,
    130,
    138,
}

OLD_PILOT_ROWS_PER_PART = 1_000


def percentage(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return round(
        numerator / denominator * 100.0,
        8,
    )


def read_census() -> list[dict[str, Any]]:
    if not CENSUS_PATH.exists():
        raise FileNotFoundError(
            f"Census file not found: {CENSUS_PATH}"
        )

    with CENSUS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        raw_rows = list(csv.DictReader(handle))

    if len(raw_rows) != 139:
        raise ValueError(
            "Expected 139 census parts, "
            f"found {len(raw_rows)}."
        )

    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        part_number = int(raw["part_number"])
        record_count = int(
            raw["nonempty_record_rows"]
        )

        exclusion_start = (
            OLD_PILOT_ROWS_PER_PART
            if part_number in OLD_PILOT_PARTS
            else 0
        )

        eligible_count = (
            record_count - exclusion_start
        )

        if eligible_count <= 0:
            raise ValueError(
                f"Part {part_number} has no eligible rows."
            )

        rows.append(
            {
                "part_number": part_number,
                "filename": raw["filename"],
                "record_count": record_count,
                "excluded_pilot_rows": (
                    exclusion_start
                ),
                "eligible_start_row": (
                    exclusion_start
                ),
                "eligible_record_count": (
                    eligible_count
                ),
            }
        )

    rows.sort(
        key=lambda row: row["part_number"]
    )

    observed_parts = [
        row["part_number"]
        for row in rows
    ]

    if observed_parts != list(range(139)):
        raise ValueError(
            "Expected continuous parts 0–138."
        )

    return rows


def allocate_samples(
    rows: list[dict[str, Any]],
) -> None:
    total_eligible = sum(
        row["eligible_record_count"]
        for row in rows
    )

    allocated = 0

    for row in rows:
        exact = (
            TARGET_DOCUMENTS
            * row["eligible_record_count"]
            / total_eligible
        )

        floor_value = math.floor(exact)

        row["exact_allocation"] = exact
        row["allocated_documents"] = (
            floor_value
        )
        row["allocation_remainder"] = (
            exact - floor_value
        )

        allocated += floor_value

    remaining = (
        TARGET_DOCUMENTS - allocated
    )

    remainder_order = sorted(
        rows,
        key=lambda row: (
            -row["allocation_remainder"],
            row["part_number"],
        ),
    )

    for row in remainder_order[:remaining]:
        row["allocated_documents"] += 1

    final_total = sum(
        row["allocated_documents"]
        for row in rows
    )

    if final_total != TARGET_DOCUMENTS:
        raise ValueError(
            "Allocation total mismatch: "
            f"{final_total} != {TARGET_DOCUMENTS}"
        )

    parts_without_samples = [
        row["part_number"]
        for row in rows
        if row["allocated_documents"] < 1
    ]

    if parts_without_samples:
        raise ValueError(
            "Some parts received no sample: "
            f"{parts_without_samples}"
        )


def build_manifest(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []

    global_number = 0

    for row in rows:
        part = row["part_number"]
        eligible_start = (
            row["eligible_start_row"]
        )
        record_count = row["record_count"]
        sample_count = (
            row["allocated_documents"]
        )

        rng = random.Random(
            RANDOM_SEED + part
        )

        selected_local_rows = sorted(
            rng.sample(
                range(
                    eligible_start,
                    record_count,
                ),
                sample_count,
            )
        )

        inclusion_probability = (
            sample_count
            / row["eligible_record_count"]
        )

        sampling_weight = (
            row["eligible_record_count"]
            / sample_count
        )

        for part_sample_number, local_row in enumerate(
            selected_local_rows,
            start=1,
        ):
            global_number += 1

            source_record_id = (
                part * 100_000
                + local_row
            )

            manifest.append(
                {
                    "global_sample_number": (
                        global_number
                    ),
                    "document_sample_id": (
                        f"gvd-v1-p{part:03d}-"
                        f"r{local_row:05d}"
                    ),
                    "part_number": part,
                    "part_sample_number": (
                        part_sample_number
                    ),
                    "filename": row["filename"],
                    "local_row_number": (
                        local_row
                    ),
                    "source_record_id": (
                        source_record_id
                    ),
                    "part_record_count": (
                        record_count
                    ),
                    "part_eligible_record_count": (
                        row[
                            "eligible_record_count"
                        ]
                    ),
                    "part_allocated_documents": (
                        sample_count
                    ),
                    "inclusion_probability": round(
                        inclusion_probability,
                        10,
                    ),
                    "sampling_weight": round(
                        sampling_weight,
                        6,
                    ),
                    "old_pilot_part": (
                        part in OLD_PILOT_PARTS
                    ),
                    "old_pilot_document_excluded": (
                        part in OLD_PILOT_PARTS
                        and local_row
                        < OLD_PILOT_ROWS_PER_PART
                    ),
                    "random_seed": (
                        RANDOM_SEED
                    ),
                }
            )

    return manifest


def validate_manifest(
    rows: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> None:
    if len(manifest) != TARGET_DOCUMENTS:
        raise ValueError(
            f"Expected {TARGET_DOCUMENTS} rows, "
            f"found {len(manifest)}."
        )

    sample_ids = [
        row["document_sample_id"]
        for row in manifest
    ]

    source_ids = [
        row["source_record_id"]
        for row in manifest
    ]

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            "Duplicate document sample IDs found."
        )

    if len(source_ids) != len(set(source_ids)):
        raise ValueError(
            "Duplicate source record IDs found."
        )

    selected_parts = {
        row["part_number"]
        for row in manifest
    }

    if selected_parts != set(range(139)):
        missing = sorted(
            set(range(139))
            - selected_parts
        )

        raise ValueError(
            f"Missing sampled parts: {missing}"
        )

    pilot_overlap = [
        row
        for row in manifest
        if row[
            "old_pilot_document_excluded"
        ]
    ]

    if pilot_overlap:
        raise ValueError(
            "Old pilot documents entered the "
            "new sample."
        )

    source_id_errors = [
        row
        for row in manifest
        if (
            row["source_record_id"]
            // 100_000
            != row["part_number"]
        )
    ]

    if source_id_errors:
        raise ValueError(
            "Source record ID and part mismatch."
        )

    allocation_lookup = {
        row["part_number"]:
        row["allocated_documents"]
        for row in rows
    }

    observed_counts: dict[int, int] = {}

    for row in manifest:
        part = row["part_number"]

        observed_counts[part] = (
            observed_counts.get(part, 0)
            + 1
        )

    if observed_counts != allocation_lookup:
        raise ValueError(
            "Manifest counts do not match "
            "the allocation table."
        )


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
    census_rows = read_census()
    allocate_samples(census_rows)

    manifest = build_manifest(
        census_rows
    )

    validate_manifest(
        census_rows,
        manifest,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    allocation_rows = []

    for row in census_rows:
        sample_count = (
            row["allocated_documents"]
        )

        allocation_rows.append(
            {
                "part_number": (
                    row["part_number"]
                ),
                "filename": (
                    row["filename"]
                ),
                "part_record_count": (
                    row["record_count"]
                ),
                "excluded_old_pilot_rows": (
                    row["excluded_pilot_rows"]
                ),
                "eligible_start_row": (
                    row["eligible_start_row"]
                ),
                "eligible_record_count": (
                    row[
                        "eligible_record_count"
                    ]
                ),
                "allocated_documents": (
                    sample_count
                ),
                "part_sampling_fraction_pct": (
                    percentage(
                        sample_count,
                        row[
                            "eligible_record_count"
                        ],
                    )
                ),
                "sampling_weight": round(
                    row[
                        "eligible_record_count"
                    ]
                    / sample_count,
                    6,
                ),
            }
        )

    write_csv(
        MANIFEST_PATH,
        manifest,
        list(manifest[0]),
    )

    write_csv(
        ALLOCATION_PATH,
        allocation_rows,
        list(allocation_rows[0]),
    )

    total_records = sum(
        row["record_count"]
        for row in census_rows
    )

    total_eligible = sum(
        row["eligible_record_count"]
        for row in census_rows
    )

    allocated_counts = [
        row["allocated_documents"]
        for row in census_rows
    ]

    summary = {
        "status": (
            "GLOBAL_VALIDATION_DOCUMENT_MANIFEST_READY"
        ),
        "random_seed": RANDOM_SEED,
        "target_documents": (
            TARGET_DOCUMENTS
        ),
        "selected_documents": (
            len(manifest)
        ),
        "parts_available": (
            len(census_rows)
        ),
        "parts_sampled": (
            len(
                {
                    row["part_number"]
                    for row in manifest
                }
            )
        ),
        "full_corpus_records": (
            total_records
        ),
        "eligible_records_after_old_pilot_exclusion": (
            total_eligible
        ),
        "old_pilot_documents_excluded": (
            len(OLD_PILOT_PARTS)
            * OLD_PILOT_ROWS_PER_PART
        ),
        "sample_fraction_of_eligible_records_pct": (
            percentage(
                len(manifest),
                total_eligible,
            )
        ),
        "minimum_documents_per_part": (
            min(allocated_counts)
        ),
        "maximum_documents_per_part": (
            max(allocated_counts)
        ),
        "duplicate_sample_ids": 0,
        "duplicate_source_record_ids": 0,
        "old_pilot_overlap": 0,
        "manifest_path": str(
            MANIFEST_PATH.relative_to(ROOT)
        ),
        "allocation_path": str(
            ALLOCATION_PATH.relative_to(ROOT)
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# Global Validation Document Sample v1",
        "",
        "## Status",
        "",
        "`GLOBAL_VALIDATION_DOCUMENT_MANIFEST_READY`",
        "",
        "## Purpose",
        "",
        "Create an unseen, all-part document sample "
        "for the final corpus-quality validation.",
        "",
        "## Sampling design",
        "",
        f"- Full curated records: "
        f"{total_records:,}",
        f"- Eligible records after excluding old "
        f"pilot documents: {total_eligible:,}",
        f"- Selected documents: "
        f"{len(manifest):,}",
        f"- Parts sampled: "
        f"{len(census_rows)} / {len(census_rows)}",
        f"- Documents per part: "
        f"{min(allocated_counts)}–"
        f"{max(allocated_counts)}",
        f"- Random seed: {RANDOM_SEED}",
        f"- Old pilot overlap: 0",
        "",
        "The manifest contains row positions only. "
        "Document text has not yet been extracted or segmented.",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(
        "--- GLOBAL VALIDATION DOCUMENT MANIFEST BUILT ---"
    )
    print(
        "full_corpus_records:",
        total_records,
    )
    print(
        "eligible_records:",
        total_eligible,
    )
    print(
        "selected_documents:",
        len(manifest),
    )
    print(
        "parts_sampled:",
        len(census_rows),
    )
    print(
        "documents_per_part_range:",
        (
            min(allocated_counts),
            max(allocated_counts),
        ),
    )
    print(
        "old_pilot_overlap:",
        0,
    )
    print(
        "duplicate_source_record_ids:",
        0,
    )
    print()
    print("Created files:")
    print(
        "-",
        MANIFEST_PATH.relative_to(ROOT),
    )
    print(
        "-",
        ALLOCATION_PATH.relative_to(ROOT),
    )
    print(
        "-",
        SUMMARY_PATH.relative_to(ROOT),
    )
    print(
        "-",
        REPORT_PATH.relative_to(ROOT),
    )
    print()
    print(
        "GLOBAL VALIDATION DOCUMENT MANIFEST: PASS"
    )
    print(
        "STATUS: READY_FOR_DOCUMENT_EXTRACTION"
    )


if __name__ == "__main__":
    main()
