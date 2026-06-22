#!/usr/bin/env python3

"""
Segment the globally sampled validation documents using the existing
Phase 3 v2 sentence segmenter.

This script:

- does not modify source documents;
- does not correct text;
- preserves source offsets and sampling lineage;
- independently validates each segmented document;
- measures warning flags, lengths and exact sentence duplicates.

It does not assign linguistic-health labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import csv
import hashlib
import importlib.util
import json
import math


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "outputs"
    / "03_sentence_inventory"
    / "global_validation_document_sample_v1"
    / "extracted_documents_v1.jsonl"
)

SEGMENTER_PATH = (
    ROOT
    / "scripts"
    / "03_sentence_segmenter_core.py"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "03_sentence_inventory"
    / "global_validation_sentence_pool_v1"
)

SENTENCES_PATH = (
    OUTPUT_DIR
    / "global_validation_sentences_v1.jsonl"
)

DOCUMENT_RESULTS_PATH = (
    OUTPUT_DIR
    / "document_segmentation_results_v1.csv"
)

PART_SUMMARY_PATH = (
    OUTPUT_DIR
    / "per_part_segmentation_summary_v1.csv"
)

BOUNDARY_COUNTS_PATH = (
    OUTPUT_DIR
    / "boundary_flag_counts_v1.csv"
)

PARENT_FLAG_COUNTS_PATH = (
    OUTPUT_DIR
    / "parent_review_flag_counts_v1.csv"
)

DUPLICATE_TEXT_PATH = (
    OUTPUT_DIR
    / "duplicate_sentence_text_groups_v1.csv"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "segmentation_metrics_v1.json"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "segmentation_report_v1.md"
)


EXPECTED_DOCUMENTS = 6_950
EXPECTED_PARTS = 139


DOCUMENT_RESULT_FIELDS = [
    "global_sample_number",
    "document_sample_id",
    "part_number",
    "local_row_number",
    "source_record_id",
    "normalized_text_character_count",
    "normalized_text_token_count",
    "sentences_produced",
    "segmentation_review_sentences",
    "boundary_flagged_sentences",
    "parent_review_required",
    "normalized_text_hash_matches",
    "validation_problem_count",
    "validation_problems",
    "segmentation_status",
    "failure_reason",
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"

        value /= 1024

    return str(size_bytes)


def percentage(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return round(
        numerator / denominator * 100.0,
        4,
    )


def percentile(
    values: list[int],
    fraction: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    position = (
        len(ordered) - 1
    ) * fraction

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(ordered[lower])

    weight = position - lower

    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def numeric_summary(
    values: list[int],
) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "minimum": 0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "maximum": 0,
        }

    return {
        "count": len(values),
        "minimum": min(values),
        "p25": round(
            percentile(values, 0.25),
            4,
        ),
        "median": round(
            percentile(values, 0.50),
            4,
        ),
        "mean": round(
            sum(values) / len(values),
            4,
        ),
        "p75": round(
            percentile(values, 0.75),
            4,
        ),
        "p95": round(
            percentile(values, 0.95),
            4,
        ),
        "maximum": max(values),
    }


def load_segmenter():
    spec = importlib.util.spec_from_file_location(
        "phase_03_sentence_segmenter_core",
        SEGMENTER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load segmenter: "
            f"{SEGMENTER_PATH}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    if not hasattr(module, "segment_document"):
        raise RuntimeError(
            "Loaded segmenter has no "
            "segment_document function."
        )

    return module


def normalize_flags(
    value: Any,
) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, tuple):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        return [
            item.strip()
            for item in text.split("|")
            if item.strip()
        ]

    raise TypeError(
        "Review flags must be a list, tuple, "
        "string or null."
    )


def verify_sentence_rows(
    parent_text: str,
    rows: list[dict[str, Any]],
) -> list[str]:
    """
    Independently validate one segmented document.

    Checks:
    - sequential sentence indices;
    - valid and non-overlapping offsets;
    - exact source-text reconstruction;
    - character-count agreement;
    - complete accounting of non-whitespace source characters;
    - unique sentence IDs inside the document.
    """

    problems: list[str] = []

    expected_indices = list(
        range(len(rows))
    )

    actual_indices = [
        row["sentence_index_in_record"]
        for row in rows
    ]

    if actual_indices != expected_indices:
        problems.append(
            "NON_SEQUENTIAL_SENTENCE_INDICES"
        )

    sentence_ids = [
        row["sentence_id"]
        for row in rows
    ]

    if (
        len(sentence_ids)
        != len(set(sentence_ids))
    ):
        problems.append(
            "DUPLICATE_SENTENCE_ID_IN_DOCUMENT"
        )

    previous_end = 0

    covered_non_whitespace_positions: set[
        int
    ] = set()

    for row in rows:
        start = row[
            "sentence_start_char"
        ]
        end = row[
            "sentence_end_char"
        ]
        sentence_text = row[
            "sentence_text"
        ]

        if not (
            0 <= start < end <= len(parent_text)
        ):
            problems.append(
                "INVALID_OFFSET_RANGE"
            )
            continue

        if start < previous_end:
            problems.append(
                "OVERLAPPING_OFFSETS"
            )

        if (
            parent_text[start:end]
            != sentence_text
        ):
            problems.append(
                "OFFSET_RECONSTRUCTION_FAILURE"
            )

        if (
            len(sentence_text)
            != row[
                "sentence_character_count"
            ]
        ):
            problems.append(
                "CHARACTER_COUNT_MISMATCH"
            )

        for position in range(start, end):
            if not parent_text[
                position
            ].isspace():
                covered_non_whitespace_positions.add(
                    position
                )

        previous_end = end

    expected_non_whitespace_positions = {
        position
        for position, character
        in enumerate(parent_text)
        if not character.isspace()
    }

    if (
        covered_non_whitespace_positions
        != expected_non_whitespace_positions
    ):
        problems.append(
            "NON_WHITESPACE_PARENT_CONTENT_"
            "NOT_ACCOUNTED_FOR"
        )

    return sorted(set(problems))


def read_documents() -> list[dict[str, Any]]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: "
            f"{INPUT_PATH}"
        )

    documents: list[
        dict[str, Any]
    ] = []

    with INPUT_PATH.open(
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
                    f"Invalid JSON at input line "
                    f"{line_number}: {error}"
                ) from error

            if not isinstance(row, dict):
                raise ValueError(
                    f"Input line {line_number} "
                    "is not a JSON object."
                )

            documents.append(row)

    if len(documents) != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"Expected {EXPECTED_DOCUMENTS} "
            f"documents, found "
            f"{len(documents)}."
        )

    sample_ids = [
        row["document_sample_id"]
        for row in documents
    ]

    source_ids = [
        int(
            row["source_record"][
                "source_record_id"
            ]
        )
        for row in documents
    ]

    if (
        len(sample_ids)
        != len(set(sample_ids))
    ):
        raise ValueError(
            "Duplicate document sample IDs "
            "found."
        )

    if (
        len(source_ids)
        != len(set(source_ids))
    ):
        raise ValueError(
            "Duplicate source record IDs found."
        )

    parts = {
        int(
            row["sample_metadata"][
                "part_number"
            ]
        )
        for row in documents
    }

    if parts != set(range(EXPECTED_PARTS)):
        raise ValueError(
            "Extracted document sample does not "
            "cover all parts 0–138."
        )

    return documents


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


def main() -> None:
    documents = read_documents()
    segmenter = load_segmenter()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_results: list[
        dict[str, Any]
    ] = []

    boundary_flag_counts = Counter()
    parent_flag_counts = Counter()
    segmentation_method_counts = Counter()
    terminal_punctuation_counts = Counter()

    per_part_documents = Counter()
    per_part_sentences = Counter()
    per_part_review_sentences = Counter()
    per_part_zero_sentence_documents = Counter()

    sentence_character_counts: list[int] = []
    sentence_token_counts: list[int] = []
    sentences_per_document: list[int] = []

    all_sentence_ids: set[str] = set()
    cross_document_duplicate_sentence_ids = 0

    sentence_text_groups: dict[
        str,
        dict[str, Any],
    ] = {}

    total_sentences = 0
    total_segmentation_review_sentences = 0
    total_boundary_flagged_sentences = 0
    documents_with_parent_review = 0
    documents_with_zero_sentences = 0

    validation_problem_counts = Counter()
    segmentation_failure_counts = Counter()

    print(
        "--- BUILDING GLOBAL VALIDATION "
        "SENTENCE POOL ---"
    )

    print(
        "documents_to_process:",
        len(documents),
    )

    with SENTENCES_PATH.open(
        "w",
        encoding="utf-8",
    ) as sentence_handle:

        for progress, wrapper in enumerate(
            documents,
            start=1,
        ):
            metadata = wrapper[
                "sample_metadata"
            ]

            source_record = wrapper[
                "source_record"
            ]

            document_sample_id = str(
                wrapper["document_sample_id"]
            )

            global_sample_number = int(
                wrapper[
                    "global_sample_number"
                ]
            )

            part_number = int(
                metadata["part_number"]
            )

            local_row_number = int(
                metadata["local_row_number"]
            )

            source_record_id = int(
                source_record[
                    "source_record_id"
                ]
            )

            normalized_text = source_record[
                "normalized_text"
            ]

            stored_text_hash = str(
                source_record.get(
                    "normalized_text_sha256",
                    "",
                )
            )

            wrapper_text_hash = str(
                wrapper.get(
                    "normalized_text_sha256",
                    "",
                )
            )

            calculated_text_hash = (
                sha256_text(normalized_text)
            )

            normalized_text_hash_matches = (
                stored_text_hash
                == calculated_text_hash
                and wrapper_text_hash
                == calculated_text_hash
            )

            parent_flags = normalize_flags(
                source_record.get(
                    "review_flags",
                    [],
                )
            )

            per_part_documents[
                part_number
            ] += 1

            if parent_flags:
                documents_with_parent_review += 1

            parent_flag_counts.update(
                parent_flags
            )

            rows: list[
                dict[str, Any]
            ] = []

            problems: list[str] = []
            failure_reason = ""

            try:
                if (
                    source_record_id
                    != int(
                        metadata[
                            "source_record_id"
                        ]
                    )
                ):
                    raise ValueError(
                        "SOURCE_ID_METADATA_MISMATCH"
                    )

                if (
                    source_record_id
                    // 100_000
                    != part_number
                ):
                    raise ValueError(
                        "SOURCE_PART_MISMATCH"
                    )

                if not isinstance(
                    normalized_text,
                    str,
                ):
                    raise TypeError(
                        "NORMALIZED_TEXT_NOT_STRING"
                    )

                if not normalized_text.strip():
                    raise ValueError(
                        "NORMALIZED_TEXT_EMPTY"
                    )

                if not normalized_text_hash_matches:
                    raise ValueError(
                        "NORMALIZED_TEXT_HASH_MISMATCH"
                    )

                rows = (
                    segmenter.segment_document(
                        source_record_id=(
                            source_record_id
                        ),
                        normalized_text=(
                            normalized_text
                        ),
                        parent_normalized_text_sha256=(
                            calculated_text_hash
                        ),
                        parent_review_flags=(
                            parent_flags
                        ),
                    )
                )

                problems = verify_sentence_rows(
                    normalized_text,
                    rows,
                )

            except Exception as error:
                failure_reason = (
                    f"{type(error).__name__}:"
                    f"{error}"
                )

                segmentation_failure_counts[
                    type(error).__name__
                ] += 1

            for problem in problems:
                validation_problem_counts[
                    problem
                ] += 1

            segmentation_status = (
                "PASS"
                if (
                    not failure_reason
                    and not problems
                )
                else "FAIL"
            )

            sentence_count = len(rows)

            if sentence_count == 0:
                documents_with_zero_sentences += 1
                per_part_zero_sentence_documents[
                    part_number
                ] += 1

            review_sentence_count = sum(
                bool(
                    row[
                        "segmentation_review_required"
                    ]
                )
                for row in rows
            )

            boundary_flagged_count = sum(
                bool(
                    row["boundary_flags"]
                )
                for row in rows
            )

            sentences_per_document.append(
                sentence_count
            )

            per_part_sentences[
                part_number
            ] += sentence_count

            per_part_review_sentences[
                part_number
            ] += review_sentence_count

            total_sentences += sentence_count

            total_segmentation_review_sentences += (
                review_sentence_count
            )

            total_boundary_flagged_sentences += (
                boundary_flagged_count
            )

            for sentence_row in rows:
                sentence_id = sentence_row[
                    "sentence_id"
                ]

                if sentence_id in all_sentence_ids:
                    cross_document_duplicate_sentence_ids += 1

                all_sentence_ids.add(
                    sentence_id
                )

                boundary_flags = normalize_flags(
                    sentence_row[
                        "boundary_flags"
                    ]
                )

                row_parent_flags = normalize_flags(
                    sentence_row[
                        "parent_review_flags"
                    ]
                )

                boundary_flag_counts.update(
                    boundary_flags
                )

                segmentation_method_counts[
                    sentence_row[
                        "segmentation_method"
                    ]
                ] += 1

                terminal_punctuation_counts[
                    sentence_row[
                        "terminal_punctuation"
                    ]
                ] += 1

                character_count = int(
                    sentence_row[
                        "sentence_character_count"
                    ]
                )

                token_count = int(
                    sentence_row[
                        "sentence_token_count"
                    ]
                )

                sentence_character_counts.append(
                    character_count
                )

                sentence_token_counts.append(
                    token_count
                )

                sentence_text = sentence_row[
                    "sentence_text"
                ]

                sentence_text_hash = (
                    sha256_text(sentence_text)
                )

                text_group = (
                    sentence_text_groups.setdefault(
                        sentence_text_hash,
                        {
                            "sentence_text_sha256": (
                                sentence_text_hash
                            ),
                            "sentence_text": (
                                sentence_text
                            ),
                            "sentence_count": 0,
                            "document_sample_ids": set(),
                            "source_record_ids": set(),
                            "sentence_ids": [],
                        },
                    )
                )

                text_group[
                    "sentence_count"
                ] += 1

                text_group[
                    "document_sample_ids"
                ].add(document_sample_id)

                text_group[
                    "source_record_ids"
                ].add(source_record_id)

                if (
                    len(
                        text_group[
                            "sentence_ids"
                        ]
                    )
                    < 20
                ):
                    text_group[
                        "sentence_ids"
                    ].append(sentence_id)

                output_row = {
                    **sentence_row,
                    "boundary_flags": (
                        boundary_flags
                    ),
                    "parent_review_flags": (
                        row_parent_flags
                    ),
                    "sentence_text_sha256": (
                        sentence_text_hash
                    ),
                    "global_sample_number": (
                        global_sample_number
                    ),
                    "document_sample_id": (
                        document_sample_id
                    ),
                    "part_number": (
                        part_number
                    ),
                    "local_row_number": (
                        local_row_number
                    ),
                    "document_sampling_weight": (
                        float(
                            metadata[
                                "sampling_weight"
                            ]
                        )
                    ),
                    "document_inclusion_probability": (
                        float(
                            metadata[
                                "inclusion_probability"
                            ]
                        )
                    ),
                    "old_pilot_overlap": (
                        bool(
                            metadata[
                                "old_pilot_overlap"
                            ]
                        )
                    ),
                }

                sentence_handle.write(
                    json.dumps(
                        output_row,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )

                sentence_handle.write("\n")

            document_results.append(
                {
                    "global_sample_number": (
                        global_sample_number
                    ),
                    "document_sample_id": (
                        document_sample_id
                    ),
                    "part_number": (
                        part_number
                    ),
                    "local_row_number": (
                        local_row_number
                    ),
                    "source_record_id": (
                        source_record_id
                    ),
                    "normalized_text_character_count": (
                        len(normalized_text)
                    ),
                    "normalized_text_token_count": (
                        len(
                            normalized_text.split()
                        )
                    ),
                    "sentences_produced": (
                        sentence_count
                    ),
                    "segmentation_review_sentences": (
                        review_sentence_count
                    ),
                    "boundary_flagged_sentences": (
                        boundary_flagged_count
                    ),
                    "parent_review_required": (
                        bool(parent_flags)
                    ),
                    "normalized_text_hash_matches": (
                        normalized_text_hash_matches
                    ),
                    "validation_problem_count": (
                        len(problems)
                    ),
                    "validation_problems": (
                        "|".join(problems)
                    ),
                    "segmentation_status": (
                        segmentation_status
                    ),
                    "failure_reason": (
                        failure_reason
                    ),
                }
            )

            if (
                progress % 500 == 0
                or progress == len(documents)
            ):
                print(
                    f"processed={progress}/"
                    f"{len(documents)} "
                    f"sentences={total_sentences} "
                    f"failures="
                    f"{sum(segmentation_failure_counts.values())} "
                    f"validation_problems="
                    f"{sum(validation_problem_counts.values())}"
                )

    document_results.sort(
        key=lambda row: int(
            row["global_sample_number"]
        )
    )

    write_csv(
        DOCUMENT_RESULTS_PATH,
        document_results,
        DOCUMENT_RESULT_FIELDS,
    )

    part_rows: list[
        dict[str, Any]
    ] = []

    for part_number in range(
        EXPECTED_PARTS
    ):
        document_count = (
            per_part_documents[
                part_number
            ]
        )

        sentence_count = (
            per_part_sentences[
                part_number
            ]
        )

        review_count = (
            per_part_review_sentences[
                part_number
            ]
        )

        part_rows.append(
            {
                "part_number": (
                    part_number
                ),
                "documents_processed": (
                    document_count
                ),
                "sentences_produced": (
                    sentence_count
                ),
                "sentences_per_document": round(
                    (
                        sentence_count
                        / document_count
                    )
                    if document_count
                    else 0.0,
                    4,
                ),
                "segmentation_review_sentences": (
                    review_count
                ),
                "segmentation_review_rate_pct": (
                    percentage(
                        review_count,
                        sentence_count,
                    )
                ),
                "zero_sentence_documents": (
                    per_part_zero_sentence_documents[
                        part_number
                    ]
                ),
            }
        )

    write_csv(
        PART_SUMMARY_PATH,
        part_rows,
        list(part_rows[0]),
    )

    boundary_rows = [
        {
            "boundary_flag": flag,
            "sentence_count": count,
            "percentage_of_all_sentences": (
                percentage(
                    count,
                    total_sentences,
                )
            ),
        }
        for flag, count in sorted(
            boundary_flag_counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    ]

    if not boundary_rows:
        boundary_rows = [
            {
                "boundary_flag": "",
                "sentence_count": 0,
                "percentage_of_all_sentences": 0.0,
            }
        ]

    write_csv(
        BOUNDARY_COUNTS_PATH,
        boundary_rows,
        [
            "boundary_flag",
            "sentence_count",
            "percentage_of_all_sentences",
        ],
    )

    parent_rows = [
        {
            "parent_review_flag": flag,
            "document_or_sentence_count": (
                count
            ),
        }
        for flag, count in sorted(
            parent_flag_counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    ]

    if not parent_rows:
        parent_rows = [
            {
                "parent_review_flag": "",
                "document_or_sentence_count": 0,
            }
        ]

    write_csv(
        PARENT_FLAG_COUNTS_PATH,
        parent_rows,
        [
            "parent_review_flag",
            "document_or_sentence_count",
        ],
    )

    duplicate_groups = []

    for group in sentence_text_groups.values():
        if group["sentence_count"] <= 1:
            continue

        duplicate_groups.append(
            {
                "sentence_text_sha256": (
                    group[
                        "sentence_text_sha256"
                    ]
                ),
                "sentence_count": (
                    group["sentence_count"]
                ),
                "distinct_document_count": (
                    len(
                        group[
                            "document_sample_ids"
                        ]
                    )
                ),
                "distinct_source_record_count": (
                    len(
                        group[
                            "source_record_ids"
                        ]
                    )
                ),
                "sentence_ids_preview": "|".join(
                    group["sentence_ids"]
                ),
                "sentence_text": (
                    group["sentence_text"]
                ),
            }
        )

    duplicate_groups.sort(
        key=lambda row: (
            -int(row["sentence_count"]),
            row["sentence_text_sha256"],
        )
    )

    write_csv(
        DUPLICATE_TEXT_PATH,
        duplicate_groups,
        [
            "sentence_text_sha256",
            "sentence_count",
            "distinct_document_count",
            "distinct_source_record_count",
            "sentence_ids_preview",
            "sentence_text",
        ],
    )

    failed_documents = [
        row
        for row in document_results
        if row["segmentation_status"]
        != "PASS"
    ]

    duplicate_sentence_text_rows = sum(
        int(row["sentence_count"])
        for row in duplicate_groups
    )

    output_size = (
        SENTENCES_PATH.stat().st_size
    )

    output_hash = sha256_file(
        SENTENCES_PATH
    )

    validation_errors: list[str] = []

    if len(document_results) != EXPECTED_DOCUMENTS:
        validation_errors.append(
            "DOCUMENT_RESULT_COUNT_MISMATCH"
        )

    if failed_documents:
        validation_errors.append(
            "FAILED_DOCUMENT_SEGMENTATIONS_PRESENT"
        )

    if (
        cross_document_duplicate_sentence_ids
        != 0
    ):
        validation_errors.append(
            "CROSS_DOCUMENT_DUPLICATE_"
            "SENTENCE_IDS_PRESENT"
        )

    if (
        len(all_sentence_ids)
        != total_sentences
    ):
        validation_errors.append(
            "UNIQUE_SENTENCE_ID_COUNT_MISMATCH"
        )

    if len(per_part_documents) != EXPECTED_PARTS:
        validation_errors.append(
            "PART_DOCUMENT_COVERAGE_MISMATCH"
        )

    metrics = {
        "status": (
            "PASS"
            if not validation_errors
            else "FAIL"
        ),
        "source_documents": str(
            INPUT_PATH.relative_to(ROOT)
        ),
        "segmenter": str(
            SEGMENTER_PATH.relative_to(ROOT)
        ),
        "documents_processed": (
            len(document_results)
        ),
        "parts_processed": (
            len(per_part_documents)
        ),
        "documents_with_zero_sentences": (
            documents_with_zero_sentences
        ),
        "documents_with_parent_review_flags": (
            documents_with_parent_review
        ),
        "sentences_written": (
            total_sentences
        ),
        "unique_sentence_ids": (
            len(all_sentence_ids)
        ),
        "cross_document_duplicate_sentence_ids": (
            cross_document_duplicate_sentence_ids
        ),
        "segmentation_review_sentences": (
            total_segmentation_review_sentences
        ),
        "segmentation_review_rate_pct": (
            percentage(
                total_segmentation_review_sentences,
                total_sentences,
            )
        ),
        "boundary_flagged_sentences": (
            total_boundary_flagged_sentences
        ),
        "boundary_flagged_sentence_rate_pct": (
            percentage(
                total_boundary_flagged_sentences,
                total_sentences,
            )
        ),
        "document_segmentation_failures": (
            len(failed_documents)
        ),
        "segmentation_failure_counts": dict(
            sorted(
                segmentation_failure_counts.items()
            )
        ),
        "validation_problem_counts": dict(
            sorted(
                validation_problem_counts.items()
            )
        ),
        "boundary_flag_counts": dict(
            sorted(
                boundary_flag_counts.items()
            )
        ),
        "parent_review_flag_counts": dict(
            sorted(
                parent_flag_counts.items()
            )
        ),
        "segmentation_method_counts": dict(
            sorted(
                segmentation_method_counts.items()
            )
        ),
        "terminal_punctuation_counts": dict(
            sorted(
                terminal_punctuation_counts.items()
            )
        ),
        "sentence_character_count": (
            numeric_summary(
                sentence_character_counts
            )
        ),
        "sentence_token_count": (
            numeric_summary(
                sentence_token_counts
            )
        ),
        "sentences_per_document": (
            numeric_summary(
                sentences_per_document
            )
        ),
        "exact_duplicate_sentence_text_groups": (
            len(duplicate_groups)
        ),
        "rows_in_exact_duplicate_text_groups": (
            duplicate_sentence_text_rows
        ),
        "exact_duplicate_sentence_row_rate_pct": (
            percentage(
                duplicate_sentence_text_rows,
                total_sentences,
            )
        ),
        "output_jsonl": str(
            SENTENCES_PATH.relative_to(ROOT)
        ),
        "output_jsonl_size_bytes": (
            output_size
        ),
        "output_jsonl_size_human": (
            human_size(output_size)
        ),
        "output_jsonl_sha256": (
            output_hash
        ),
        "validation_errors": (
            validation_errors
        ),
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# Global Validation Sentence Pool v1",
        "",
        "## Status",
        "",
        f"`{metrics['status']}`",
        "",
        "## Processing",
        "",
        f"- Documents processed: "
        f"{len(document_results):,}",
        f"- Parts processed: "
        f"{len(per_part_documents)} / "
        f"{EXPECTED_PARTS}",
        f"- Sentences produced: "
        f"{total_sentences:,}",
        f"- Documents with zero sentences: "
        f"{documents_with_zero_sentences:,}",
        f"- Segmentation failures: "
        f"{len(failed_documents):,}",
        f"- Validation problems: "
        f"{sum(validation_problem_counts.values()):,}",
        f"- Duplicate sentence IDs: "
        f"{cross_document_duplicate_sentence_ids:,}",
        "",
        "## Warning signals",
        "",
        f"- Segmentation-review sentences: "
        f"{total_segmentation_review_sentences:,} "
        f"({metrics['segmentation_review_rate_pct']}%)",
        f"- Boundary-flagged sentences: "
        f"{total_boundary_flagged_sentences:,} "
        f"({metrics['boundary_flagged_sentence_rate_pct']}%)",
        f"- Documents with parent review flags: "
        f"{documents_with_parent_review:,}",
        "",
        "## Exact sentence-text duplicates",
        "",
        f"- Duplicate text groups: "
        f"{len(duplicate_groups):,}",
        f"- Rows in duplicate groups: "
        f"{duplicate_sentence_text_rows:,}",
        f"- Duplicate-row rate: "
        f"{metrics['exact_duplicate_sentence_row_rate_pct']}%",
        "",
        "## Important interpretation",
        "",
        "A segmentation PASS proves structural integrity only. "
        "It does not prove that the sentences are linguistically "
        "healthy or suitable for training.",
        "",
        "The next step is to draw an unseen manual-review sample "
        "from this sentence pool.",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print(
        "--- GLOBAL VALIDATION SENTENCE "
        "POOL BUILT ---"
    )

    print(
        "documents_processed:",
        len(document_results),
    )

    print(
        "parts_processed:",
        len(per_part_documents),
    )

    print(
        "sentences_written:",
        total_sentences,
    )

    print(
        "documents_with_zero_sentences:",
        documents_with_zero_sentences,
    )

    print(
        "segmentation_review_sentences:",
        total_segmentation_review_sentences,
    )

    print(
        "segmentation_review_rate_pct:",
        metrics[
            "segmentation_review_rate_pct"
        ],
    )

    print(
        "boundary_flagged_sentences:",
        total_boundary_flagged_sentences,
    )

    print(
        "boundary_flagged_sentence_rate_pct:",
        metrics[
            "boundary_flagged_sentence_rate_pct"
        ],
    )

    print(
        "document_segmentation_failures:",
        len(failed_documents),
    )

    print(
        "validation_problem_counts:",
        dict(
            validation_problem_counts
        ),
    )

    print(
        "cross_document_duplicate_sentence_ids:",
        cross_document_duplicate_sentence_ids,
    )

    print(
        "exact_duplicate_sentence_text_groups:",
        len(duplicate_groups),
    )

    print(
        "rows_in_exact_duplicate_text_groups:",
        duplicate_sentence_text_rows,
    )

    print(
        "sentence_character_count:",
        metrics[
            "sentence_character_count"
        ],
    )

    print(
        "sentence_token_count:",
        metrics[
            "sentence_token_count"
        ],
    )

    print(
        "sentences_per_document:",
        metrics[
            "sentences_per_document"
        ],
    )

    print(
        "output_jsonl_size:",
        metrics[
            "output_jsonl_size_human"
        ],
    )

    print(
        "output_jsonl_sha256:",
        output_hash,
    )

    print()
    print("Created files:")

    for path in [
        SENTENCES_PATH,
        DOCUMENT_RESULTS_PATH,
        PART_SUMMARY_PATH,
        BOUNDARY_COUNTS_PATH,
        PARENT_FLAG_COUNTS_PATH,
        DUPLICATE_TEXT_PATH,
        METRICS_PATH,
        REPORT_PATH,
    ]:
        print(
            "-",
            path.relative_to(ROOT),
        )

    print()

    if validation_errors:
        print(
            "GLOBAL VALIDATION SENTENCE POOL: "
            "FAIL"
        )

        print(
            "validation_errors:",
            validation_errors,
        )

        raise SystemExit(1)

    print(
        "GLOBAL VALIDATION SENTENCE POOL: PASS"
    )

    print(
        "STATUS: READY_FOR_GLOBAL_"
        "SENTENCE_REVIEW_SAMPLING"
    )


if __name__ == "__main__":
    main()
