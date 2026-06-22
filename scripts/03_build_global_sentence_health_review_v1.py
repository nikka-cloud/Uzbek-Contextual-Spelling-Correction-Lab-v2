#!/usr/bin/env python3

"""
Build an unseen, stratified random sentence-health review sample.

The sample is drawn proportionally across all 139 corpus parts from
the global validation sentence pool.

This script does not route, repair, delete or modify corpus sentences.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import csv
import hashlib
import json
import math
import random


ROOT = Path(__file__).resolve().parents[1]

SENTENCE_POOL_PATH = (
    ROOT
    / "outputs"
    / "03_sentence_inventory"
    / "global_validation_sentence_pool_v1"
    / "global_validation_sentences_v1.jsonl"
)

SEGMENTATION_METRICS_PATH = (
    ROOT
    / "outputs"
    / "03_sentence_inventory"
    / "global_validation_sentence_pool_v1"
    / "segmentation_metrics_v1.json"
)

OLD_REVIEW_PATH = (
    ROOT
    / "outputs"
    / "03_sentence_inventory"
    / "corpus_viability_audit_v1"
    / "estimation_random_review_master_v1.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "03_sentence_inventory"
    / "global_sentence_health_review_v1"
)

BATCH_DIR = OUTPUT_DIR / "review_batches_v1"

MASTER_PATH = (
    OUTPUT_DIR
    / "global_sentence_health_review_master_v1.csv"
)

ALLOCATION_PATH = (
    OUTPUT_DIR
    / "global_sentence_health_part_allocation_v1.csv"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "global_sentence_health_sampling_metrics_v1.json"
)

INSTRUCTIONS_PATH = (
    OUTPUT_DIR
    / "global_sentence_health_review_instructions_v1.md"
)


EXPECTED_POOL_ROWS = 62_827
EXPECTED_PARTS = 139
TARGET_REVIEW_ROWS = 1_000
BATCH_SIZE = 25
RANDOM_SEED = 20260618

ALLOWED_LABELS = [
    "HEALTHY",
    "MINOR_DAMAGE_BUT_USABLE",
    "FORMAT_OR_LIST",
    "FRAGMENT",
    "HEAVY_CORRUPTION",
    "MIXED_OR_FOREIGN_TEXT",
    "UNCERTAIN",
]


FIELDNAMES = [
    "review_row_number",
    "review_sample_id",
    "sample_role",
    "sample_group",
    "batch_number",
    "sampling_seed",
    "part_number",
    "part_sentence_population",
    "part_review_sample_size",
    "within_part_sampling_weight",
    "document_sampling_weight",
    "two_stage_estimation_weight",
    "global_sample_number",
    "document_sample_id",
    "local_row_number",
    "source_record_id",
    "sentence_id",
    "sentence_index_in_record",
    "sentence_start_char",
    "sentence_end_char",
    "sentence_character_count",
    "sentence_token_count",
    "segmentation_method",
    "terminal_punctuation",
    "boundary_flags",
    "segmentation_review_required",
    "parent_review_flags",
    "exact_text_group_size",
    "exact_duplicate_text",
    "previous_sentence",
    "sentence_text",
    "next_sentence",
    "human_health_label",
    "human_health_notes",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def percentage(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return round(
        numerator / denominator * 100.0,
        6,
    )


def normalize_flags(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, str):
        return [
            item.strip()
            for item in value.split("|")
            if item.strip()
        ]

    raise TypeError(
        f"Unsupported flag value: {type(value).__name__}"
    )


def read_sentence_pool() -> list[dict[str, Any]]:
    if not SENTENCE_POOL_PATH.exists():
        raise FileNotFoundError(
            f"Sentence pool not found: {SENTENCE_POOL_PATH}"
        )

    rows: list[dict[str, Any]] = []

    with SENTENCE_POOL_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error

            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} is not a JSON object."
                )

            rows.append(row)

    if len(rows) != EXPECTED_POOL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_POOL_ROWS} sentence rows, "
            f"found {len(rows)}."
        )

    sentence_ids = [
        str(row["sentence_id"])
        for row in rows
    ]

    if len(sentence_ids) != len(set(sentence_ids)):
        raise ValueError(
            "Duplicate sentence IDs found in sentence pool."
        )

    parts = {
        int(row["part_number"])
        for row in rows
    }

    if parts != set(range(EXPECTED_PARTS)):
        missing_parts = sorted(
            set(range(EXPECTED_PARTS)) - parts
        )

        raise ValueError(
            f"Sentence pool is missing parts: {missing_parts}"
        )

    old_overlap = sum(
        bool(row.get("old_pilot_overlap", False))
        for row in rows
    )

    if old_overlap:
        raise ValueError(
            f"Sentence pool contains {old_overlap} old-pilot rows."
        )

    return rows


def read_old_review_ids() -> tuple[set[int], set[str]]:
    if not OLD_REVIEW_PATH.exists():
        raise FileNotFoundError(
            f"Old review file not found: {OLD_REVIEW_PATH}"
        )

    source_ids: set[int] = set()
    sentence_ids: set[str] = set()

    with OLD_REVIEW_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            source_ids.add(
                int(row["source_record_id"])
            )

            sentence_id = str(
                row.get("sentence_id", "")
            ).strip()

            if sentence_id:
                sentence_ids.add(sentence_id)

    return source_ids, sentence_ids


def allocate_by_part(
    rows_by_part: dict[int, list[dict[str, Any]]],
) -> dict[int, int]:
    total_rows = sum(
        len(rows)
        for rows in rows_by_part.values()
    )

    allocation: dict[int, int] = {}
    remainders: list[tuple[float, int]] = []
    allocated = 0

    for part in range(EXPECTED_PARTS):
        population = len(rows_by_part[part])

        exact = (
            TARGET_REVIEW_ROWS
            * population
            / total_rows
        )

        floor_value = math.floor(exact)

        if floor_value < 1:
            floor_value = 1

        allocation[part] = floor_value
        allocated += floor_value

        remainders.append(
            (
                exact - math.floor(exact),
                part,
            )
        )

    if allocated > TARGET_REVIEW_ROWS:
        removable = sorted(
            (
                (
                    remainders[part][0],
                    part,
                )
                for part in range(EXPECTED_PARTS)
                if allocation[part] > 1
            ),
            key=lambda item: (
                item[0],
                item[1],
            ),
        )

        difference = allocated - TARGET_REVIEW_ROWS

        for _, part in removable[:difference]:
            allocation[part] -= 1

    elif allocated < TARGET_REVIEW_ROWS:
        remaining = TARGET_REVIEW_ROWS - allocated

        for _, part in sorted(
            remainders,
            key=lambda item: (
                -item[0],
                item[1],
            ),
        )[:remaining]:
            allocation[part] += 1

    final_total = sum(allocation.values())

    if final_total != TARGET_REVIEW_ROWS:
        raise ValueError(
            "Part allocation failed: "
            f"{final_total} != {TARGET_REVIEW_ROWS}"
        )

    for part, sample_count in allocation.items():
        population = len(rows_by_part[part])

        if sample_count > population:
            raise ValueError(
                f"Part {part} allocation exceeds population."
            )

    return allocation


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
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def write_instructions() -> None:
    content = """# Global Sentence Health Review v1

## Purpose

This is the unseen global sentence-quality review.

Label only the CURRENT SENTENCE.

Previous and next sentences are context only. Do not label the whole
document.

## Allowed labels

### HEALTHY

Normal, meaningful Uzbek sentence-like prose that is usable without
important repair.

### MINOR_DAMAGE_BUT_USABLE

Meaning and sentence structure remain recoverable, but there is limited
damage such as missing apostrophes, one merged word, an isolated typo,
minor punctuation damage or a small OCR-like problem.

### FORMAT_OR_LIST

Heading, table row, bibliography item, score, metadata, list entry,
standalone label, URL, reference or other non-prose structure.

### FRAGMENT

Only part of a sentence, with an incomplete beginning, ending or
grammatical structure.

### HEAVY_CORRUPTION

Reliable reconstruction would require substantial guessing.

### MIXED_OR_FOREIGN_TEXT

Primarily foreign-language or too mixed to be ordinary Uzbek prose.

### UNCERTAIN

Use only when a confident decision cannot be made.

## Important rules

- Do not repair text while reviewing.
- Do not change sentence text.
- Do not infer that an unflagged sentence is healthy.
- Do not infer that a flagged sentence is harmful.
- Short sentences may be healthy.
- Literary, historical, religious or technical language may be healthy.
- Label the current sentence independently of whether it is duplicated.
"""

    INSTRUCTIONS_PATH.write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    if OUTPUT_DIR.exists():
        raise RuntimeError(
            "Output directory already exists. "
            "Refusing to overwrite a versioned review set: "
            f"{OUTPUT_DIR}"
        )

    metrics = json.loads(
        SEGMENTATION_METRICS_PATH.read_text(
            encoding="utf-8"
        )
    )

    if int(metrics["sentences_written"]) != EXPECTED_POOL_ROWS:
        raise ValueError(
            "Segmentation metrics and sentence pool disagree."
        )

    all_rows = read_sentence_pool()
    old_source_ids, old_sentence_ids = read_old_review_ids()

    rows_by_part: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    context_lookup: dict[
        tuple[str, int],
        dict[str, Any],
    ] = {}

    text_group_sizes = Counter(
        str(row["sentence_text_sha256"])
        for row in all_rows
    )

    for row in all_rows:
        part = int(row["part_number"])

        rows_by_part[part].append(row)

        context_lookup[
            (
                str(row["document_sample_id"]),
                int(
                    row[
                        "sentence_index_in_record"
                    ]
                ),
            )
        ] = row

    allocation = allocate_by_part(
        rows_by_part
    )

    selected: list[dict[str, Any]] = []

    for part in range(EXPECTED_PARTS):
        rng = random.Random(
            RANDOM_SEED + part * 1009
        )

        selected.extend(
            rng.sample(
                rows_by_part[part],
                allocation[part],
            )
        )

    if len(selected) != TARGET_REVIEW_ROWS:
        raise ValueError(
            "Selected review-row count mismatch."
        )

    shuffle_rng = random.Random(
        RANDOM_SEED
    )

    shuffle_rng.shuffle(selected)

    selected_sentence_ids = [
        str(row["sentence_id"])
        for row in selected
    ]

    if (
        len(selected_sentence_ids)
        != len(set(selected_sentence_ids))
    ):
        raise ValueError(
            "Duplicate sentence IDs in selected review sample."
        )

    selected_old_source_overlap = {
        int(row["source_record_id"])
        for row in selected
    } & old_source_ids

    selected_old_sentence_overlap = (
        set(selected_sentence_ids)
        & old_sentence_ids
    )

    if selected_old_source_overlap:
        raise ValueError(
            "Selected sample overlaps old reviewed source records."
        )

    if selected_old_sentence_overlap:
        raise ValueError(
            "Selected sample overlaps old reviewed sentence IDs."
        )

    review_rows: list[dict[str, Any]] = []

    for review_number, row in enumerate(
        selected,
        start=1,
    ):
        part = int(row["part_number"])
        part_population = len(
            rows_by_part[part]
        )
        part_sample_size = allocation[part]

        within_part_weight = (
            part_population
            / part_sample_size
        )

        document_weight = float(
            row["document_sampling_weight"]
        )

        sentence_index = int(
            row["sentence_index_in_record"]
        )

        document_sample_id = str(
            row["document_sample_id"]
        )

        previous_row = context_lookup.get(
            (
                document_sample_id,
                sentence_index - 1,
            )
        )

        next_row = context_lookup.get(
            (
                document_sample_id,
                sentence_index + 1,
            )
        )

        boundary_flags = normalize_flags(
            row.get("boundary_flags")
        )

        parent_flags = normalize_flags(
            row.get("parent_review_flags")
        )

        text_hash = str(
            row["sentence_text_sha256"]
        )

        duplicate_group_size = int(
            text_group_sizes[text_hash]
        )

        batch_number = (
            (review_number - 1)
            // BATCH_SIZE
            + 1
        )

        review_rows.append(
            {
                "review_row_number": (
                    review_number
                ),
                "review_sample_id": (
                    f"gshv1-{review_number:04d}-"
                    f"{row['sentence_id']}"
                ),
                "sample_role": (
                    "GLOBAL_ESTIMATION_RANDOM"
                ),
                "sample_group": (
                    "GLOBAL_UNSEEN_SENTENCE_HEALTH"
                ),
                "batch_number": batch_number,
                "sampling_seed": RANDOM_SEED,
                "part_number": part,
                "part_sentence_population": (
                    part_population
                ),
                "part_review_sample_size": (
                    part_sample_size
                ),
                "within_part_sampling_weight": round(
                    within_part_weight,
                    8,
                ),
                "document_sampling_weight": round(
                    document_weight,
                    8,
                ),
                "two_stage_estimation_weight": round(
                    document_weight
                    * within_part_weight,
                    8,
                ),
                "global_sample_number": (
                    row["global_sample_number"]
                ),
                "document_sample_id": (
                    document_sample_id
                ),
                "local_row_number": (
                    row["local_row_number"]
                ),
                "source_record_id": (
                    row["source_record_id"]
                ),
                "sentence_id": (
                    row["sentence_id"]
                ),
                "sentence_index_in_record": (
                    sentence_index
                ),
                "sentence_start_char": (
                    row["sentence_start_char"]
                ),
                "sentence_end_char": (
                    row["sentence_end_char"]
                ),
                "sentence_character_count": (
                    row["sentence_character_count"]
                ),
                "sentence_token_count": (
                    row["sentence_token_count"]
                ),
                "segmentation_method": (
                    row["segmentation_method"]
                ),
                "terminal_punctuation": (
                    row["terminal_punctuation"]
                ),
                "boundary_flags": "|".join(
                    boundary_flags
                ),
                "segmentation_review_required": (
                    bool(
                        row[
                            "segmentation_review_required"
                        ]
                    )
                ),
                "parent_review_flags": "|".join(
                    parent_flags
                ),
                "exact_text_group_size": (
                    duplicate_group_size
                ),
                "exact_duplicate_text": (
                    duplicate_group_size > 1
                ),
                "previous_sentence": (
                    previous_row["sentence_text"]
                    if previous_row
                    else ""
                ),
                "sentence_text": (
                    row["sentence_text"]
                ),
                "next_sentence": (
                    next_row["sentence_text"]
                    if next_row
                    else ""
                ),
                "human_health_label": "",
                "human_health_notes": "",
            }
        )

    if TARGET_REVIEW_ROWS % BATCH_SIZE != 0:
        raise ValueError(
            "Target review rows must divide evenly "
            "by batch size."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    BATCH_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        MASTER_PATH,
        review_rows,
        FIELDNAMES,
    )

    allocation_rows = []

    for part in range(EXPECTED_PARTS):
        population = len(
            rows_by_part[part]
        )
        sample_count = allocation[part]

        allocation_rows.append(
            {
                "part_number": part,
                "sentence_population": population,
                "review_sample_size": sample_count,
                "sampling_fraction_pct": (
                    percentage(
                        sample_count,
                        population,
                    )
                ),
                "within_part_sampling_weight": round(
                    population / sample_count,
                    8,
                ),
            }
        )

    write_csv(
        ALLOCATION_PATH,
        allocation_rows,
        [
            "part_number",
            "sentence_population",
            "review_sample_size",
            "sampling_fraction_pct",
            "within_part_sampling_weight",
        ],
    )

    batch_count = (
        TARGET_REVIEW_ROWS
        // BATCH_SIZE
    )

    for batch_number in range(
        1,
        batch_count + 1,
    ):
        start = (
            batch_number - 1
        ) * BATCH_SIZE

        end = start + BATCH_SIZE

        batch_rows = review_rows[start:end]

        batch_path = (
            BATCH_DIR
            / f"batch-{batch_number:02d}.csv"
        )

        write_csv(
            batch_path,
            batch_rows,
            FIELDNAMES,
        )

    write_instructions()

    selected_document_counts = Counter(
        row["document_sample_id"]
        for row in review_rows
    )

    selected_part_counts = Counter(
        int(row["part_number"])
        for row in review_rows
    )

    boundary_flagged = sum(
        bool(row["boundary_flags"])
        for row in review_rows
    )

    parent_flagged = sum(
        bool(row["parent_review_flags"])
        for row in review_rows
    )

    duplicate_rows = sum(
        bool(row["exact_duplicate_text"])
        for row in review_rows
    )

    metrics_output = {
        "status": (
            "READY_FOR_MANUAL_GLOBAL_HEALTH_REVIEW"
        ),
        "source_sentence_pool": str(
            SENTENCE_POOL_PATH.relative_to(ROOT)
        ),
        "source_sentence_rows": len(all_rows),
        "parts_available": len(rows_by_part),
        "parts_sampled": len(selected_part_counts),
        "target_review_rows": (
            TARGET_REVIEW_ROWS
        ),
        "selected_review_rows": (
            len(review_rows)
        ),
        "batch_size": BATCH_SIZE,
        "batch_count": batch_count,
        "random_seed": RANDOM_SEED,
        "minimum_rows_per_part": min(
            selected_part_counts.values()
        ),
        "maximum_rows_per_part": max(
            selected_part_counts.values()
        ),
        "unique_documents_in_review": (
            len(selected_document_counts)
        ),
        "maximum_review_rows_from_one_document": (
            max(selected_document_counts.values())
        ),
        "boundary_flagged_review_rows": (
            boundary_flagged
        ),
        "boundary_flagged_review_rate_pct": (
            percentage(
                boundary_flagged,
                len(review_rows),
            )
        ),
        "parent_flagged_review_rows": (
            parent_flagged
        ),
        "exact_duplicate_review_rows": (
            duplicate_rows
        ),
        "old_source_record_overlap": 0,
        "old_sentence_id_overlap": 0,
        "duplicate_selected_sentence_ids": 0,
        "allowed_labels": ALLOWED_LABELS,
        "master_csv": str(
            MASTER_PATH.relative_to(ROOT)
        ),
        "master_csv_sha256": (
            sha256_file(MASTER_PATH)
        ),
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics_output,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "--- GLOBAL SENTENCE HEALTH REVIEW BUILT ---"
    )
    print(
        "source_sentence_rows:",
        len(all_rows),
    )
    print(
        "parts_sampled:",
        len(selected_part_counts),
    )
    print(
        "selected_review_rows:",
        len(review_rows),
    )
    print(
        "rows_per_part_range:",
        (
            min(selected_part_counts.values()),
            max(selected_part_counts.values()),
        ),
    )
    print(
        "unique_documents_in_review:",
        len(selected_document_counts),
    )
    print(
        "maximum_rows_from_one_document:",
        max(selected_document_counts.values()),
    )
    print(
        "boundary_flagged_review_rows:",
        boundary_flagged,
    )
    print(
        "parent_flagged_review_rows:",
        parent_flagged,
    )
    print(
        "exact_duplicate_review_rows:",
        duplicate_rows,
    )
    print(
        "old_source_record_overlap:",
        0,
    )
    print(
        "old_sentence_id_overlap:",
        0,
    )
    print(
        "batch_count:",
        batch_count,
    )
    print()
    print("Created files:")
    print("-", MASTER_PATH.relative_to(ROOT))
    print("-", ALLOCATION_PATH.relative_to(ROOT))
    print("-", METRICS_PATH.relative_to(ROOT))
    print("-", INSTRUCTIONS_PATH.relative_to(ROOT))
    print("-", BATCH_DIR.relative_to(ROOT))
    print()
    print(
        "GLOBAL SENTENCE HEALTH REVIEW: PASS"
    )
    print(
        "STATUS: READY_FOR_MANUAL_GLOBAL_HEALTH_REVIEW"
    )


if __name__ == "__main__":
    main()
