#!/usr/bin/env python3
"""
Run a stratified 10,000-document Phase 3 sentence-inventory experiment.

This script:
- reads Phase 2 normalized documents;
- segments them without correcting text;
- independently validates offsets and content accounting;
- writes sentence-level JSONL;
- exports metrics and compact inspection samples.

The output is an experimental inventory, not a clean training corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import importlib.util
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATA_ROOT = Path(
    "outputs/02_curated_corpus/"
    "normalize_full_v1/normalized_records"
)

OUTPUT_BASE = Path(
    "outputs/03_sentence_inventory"
)

PROTECTED_EXPERIMENT_NAMES = {
    "segment_10k_v1",
}

SEGMENTER_PATH = Path(
    "scripts/03_sentence_segmenter_core.py"
)

PART_NUMBERS = [
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
]

ROWS_PER_PART = 1_000
MAX_FLAG_EXAMPLES = 5
MAX_INSPECTION_ROWS_PER_PART_AND_GROUP = 3
LONGEST_SENTENCE_COUNT = 30


def parse_arguments() -> argparse.Namespace:
    """Parse and validate the experiment output name."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a stratified Phase 3 sentence-inventory "
            "mini experiment without overwriting prior runs."
        )
    )

    parser.add_argument(
        "--experiment-name",
        required=True,
        help=(
            "Output directory name under "
            "outputs/03_sentence_inventory, for example "
            "segment_10k_v2."
        ),
    )

    arguments = parser.parse_args()
    experiment_name = arguments.experiment_name

    if not experiment_name:
        parser.error(
            "--experiment-name must not be empty."
        )

    allowed_characters = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_-."
    )

    if any(
        character not in allowed_characters
        for character in experiment_name
    ):
        parser.error(
            "--experiment-name may contain only letters, "
            "digits, underscores, hyphens, and periods."
        )

    if experiment_name in {".", ".."}:
        parser.error(
            "--experiment-name cannot be '.' or '..'."
        )

    if experiment_name.endswith(".tmp"):
        parser.error(
            "--experiment-name must not end in '.tmp'."
        )

    if experiment_name in PROTECTED_EXPERIMENT_NAMES:
        parser.error(
            f"{experiment_name!r} is a protected baseline "
            "and cannot be selected as a new output."
        )

    return arguments


def load_segmenter():
    """Load the Phase 3 core without requiring scripts to be a package."""
    spec = importlib.util.spec_from_file_location(
        "phase_03_sentence_segmenter_core",
        SEGMENTER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load segmenter: {SEGMENTER_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_sentence_rows(
    parent_text: str,
    rows: list[dict[str, Any]],
) -> list[str]:
    """
    Independently validate one segmented document.

    Checks:
    - sequential sentence indices;
    - valid and non-overlapping offsets;
    - exact text reconstruction;
    - character-count agreement;
    - complete accounting of all non-whitespace source characters;
    - unique sentence IDs inside the document.
    """
    problems: list[str] = []

    expected_indices = list(range(len(rows)))
    actual_indices = [
        row["sentence_index_in_record"]
        for row in rows
    ]

    if actual_indices != expected_indices:
        problems.append("NON_SEQUENTIAL_SENTENCE_INDICES")

    sentence_ids = [
        row["sentence_id"]
        for row in rows
    ]

    if len(sentence_ids) != len(set(sentence_ids)):
        problems.append("DUPLICATE_SENTENCE_ID_IN_DOCUMENT")

    previous_end = 0
    covered_non_whitespace_positions: set[int] = set()

    for row in rows:
        start = row["sentence_start_char"]
        end = row["sentence_end_char"]
        sentence_text = row["sentence_text"]

        if not (0 <= start < end <= len(parent_text)):
            problems.append("INVALID_OFFSET_RANGE")
            continue

        if start < previous_end:
            problems.append("OVERLAPPING_OFFSETS")

        if parent_text[start:end] != sentence_text:
            problems.append("OFFSET_RECONSTRUCTION_FAILURE")

        if len(sentence_text) != row["sentence_character_count"]:
            problems.append("CHARACTER_COUNT_MISMATCH")

        for position in range(start, end):
            if not parent_text[position].isspace():
                covered_non_whitespace_positions.add(position)

        previous_end = end

    expected_non_whitespace_positions = {
        position
        for position, char in enumerate(parent_text)
        if not char.isspace()
    }

    if (
        covered_non_whitespace_positions
        != expected_non_whitespace_positions
    ):
        problems.append(
            "NON_WHITESPACE_PARENT_CONTENT_NOT_ACCOUNTED_FOR"
        )

    return sorted(set(problems))


def percentile(values: list[int], percentage: float) -> int:
    """Return a deterministic nearest-rank percentile."""
    if not values:
        return 0

    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentage)
    return ordered[index]


def sha256_file(path: Path) -> str:
    """Calculate a file checksum without loading the file into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact JSONL with UTF-8 Uzbek text preserved."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def main() -> None:
    arguments = parse_arguments()
    experiment_name = arguments.experiment_name

    output_root = OUTPUT_BASE / experiment_name
    temp_root = OUTPUT_BASE / f"{experiment_name}.tmp"

    if output_root.exists():
        raise FileExistsError(
            f"Output already exists: {output_root}\n"
            "It was not overwritten."
        )

    if temp_root.exists():
        shutil.rmtree(temp_root)

    temp_root.mkdir(parents=True, exist_ok=False)

    segmenter = load_segmenter()

    sentences_path = temp_root / "sentences.jsonl"
    metrics_path = temp_root / "metrics.json"
    inspection_path = temp_root / "inspection_sample.jsonl"
    flag_examples_path = (
        temp_root / "boundary_flag_examples.jsonl"
    )
    longest_path = temp_root / "longest_sentences.csv"

    metrics = Counter()
    boundary_flag_counts = Counter()
    parent_review_flag_counts = Counter()
    terminal_punctuation_counts = Counter()
    segmentation_method_counts = Counter()
    per_part_document_counts = Counter()
    per_part_sentence_counts = Counter()

    sentence_character_counts: list[int] = []
    sentence_token_counts: list[int] = []
    sentences_per_document: list[int] = []

    all_sentence_ids: set[str] = set()
    problem_counts = Counter()

    flag_examples: dict[str, list[dict[str, Any]]] = (
        defaultdict(list)
    )
    inspection_rows: list[dict[str, Any]] = []
    inspection_group_counts = Counter()

    longest_heap: list[
        tuple[int, str, int, dict[str, Any]]
    ] = []
    heap_sequence = 0

    with sentences_path.open(
        "w",
        encoding="utf-8",
    ) as sentence_handle:

        for part_number in PART_NUMBERS:
            input_path = (
                DATA_ROOT
                / f"part-{part_number:05d}.jsonl"
            )

            if not input_path.exists():
                raise FileNotFoundError(input_path)

            with input_path.open(
                "r",
                encoding="utf-8",
            ) as input_handle:

                for local_row_number, line in enumerate(
                    input_handle
                ):
                    if local_row_number >= ROWS_PER_PART:
                        break

                    parent = json.loads(line)
                    source_record_id = parent[
                        "source_record_id"
                    ]
                    parent_text = parent["normalized_text"]

                    rows = segmenter.segment_document(
                        source_record_id=source_record_id,
                        normalized_text=parent_text,
                        parent_normalized_text_sha256=(
                            parent[
                                "normalized_text_sha256"
                            ]
                        ),
                        parent_review_flags=parent.get(
                            "review_flags",
                            [],
                        ),
                    )

                    problems = verify_sentence_rows(
                        parent_text,
                        rows,
                    )

                    for problem in problems:
                        problem_counts[problem] += 1

                    for row in rows:
                        sentence_id = row["sentence_id"]

                        if sentence_id in all_sentence_ids:
                            problem_counts[
                                "CROSS_DOCUMENT_DUPLICATE_SENTENCE_ID"
                            ] += 1

                        all_sentence_ids.add(sentence_id)

                    metrics["documents_scanned"] += 1
                    metrics["sentences_written"] += len(rows)
                    metrics["documents_with_parent_review"] += bool(
                        parent.get("review_required", False)
                    )

                    review_sentence_count = sum(
                        bool(
                            row[
                                "segmentation_review_required"
                            ]
                        )
                        for row in rows
                    )

                    metrics[
                        "segmentation_review_sentences"
                    ] += review_sentence_count

                    if review_sentence_count:
                        metrics[
                            "documents_with_segmentation_review"
                        ] += 1

                    if not rows:
                        metrics["documents_with_zero_sentences"] += 1

                    per_part_document_counts[part_number] += 1
                    per_part_sentence_counts[part_number] += len(
                        rows
                    )
                    sentences_per_document.append(len(rows))

                    parent_review_flag_counts.update(
                        parent.get("review_flags", [])
                    )

                    unflagged_added = False
                    review_added = False

                    for row in rows:
                        sentence_handle.write(
                            json.dumps(
                                row,
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )

                        character_count = row[
                            "sentence_character_count"
                        ]
                        token_count = row[
                            "sentence_token_count"
                        ]

                        sentence_character_counts.append(
                            character_count
                        )
                        sentence_token_counts.append(
                            token_count
                        )

                        boundary_flag_counts.update(
                            row["boundary_flags"]
                        )
                        terminal_punctuation_counts[
                            row["terminal_punctuation"]
                        ] += 1
                        segmentation_method_counts[
                            row["segmentation_method"]
                        ] += 1

                        for flag in row["boundary_flags"]:
                            if (
                                len(flag_examples[flag])
                                < MAX_FLAG_EXAMPLES
                            ):
                                flag_examples[flag].append(
                                    {
                                        "flag": flag,
                                        "source_record_id": (
                                            source_record_id
                                        ),
                                        "sentence_id": (
                                            sentence_id
                                        ),
                                        "sentence_text": row[
                                            "sentence_text"
                                        ],
                                        "all_boundary_flags": row[
                                            "boundary_flags"
                                        ],
                                    }
                                )

                        group = (
                            "REVIEW"
                            if row[
                                "segmentation_review_required"
                            ]
                            else "UNFLAGGED_BOUNDARY"
                        )

                        group_key = (
                            part_number,
                            group,
                        )

                        can_add_inspection = (
                            inspection_group_counts[
                                group_key
                            ]
                            < MAX_INSPECTION_ROWS_PER_PART_AND_GROUP
                        )

                        if group == "REVIEW" and not review_added:
                            if can_add_inspection:
                                inspection_rows.append(
                                    {
                                        "part_number": part_number,
                                        "inspection_group": group,
                                        "source_record_id": (
                                            source_record_id
                                        ),
                                        "parent_review_flags": (
                                            parent.get(
                                                "review_flags",
                                                [],
                                            )
                                        ),
                                        **row,
                                    }
                                )
                                inspection_group_counts[
                                    group_key
                                ] += 1
                                review_added = True

                        if (
                            group == "UNFLAGGED_BOUNDARY"
                            and not unflagged_added
                        ):
                            if can_add_inspection:
                                inspection_rows.append(
                                    {
                                        "part_number": part_number,
                                        "inspection_group": group,
                                        "source_record_id": (
                                            source_record_id
                                        ),
                                        "parent_review_flags": (
                                            parent.get(
                                                "review_flags",
                                                [],
                                            )
                                        ),
                                        **row,
                                    }
                                )
                                inspection_group_counts[
                                    group_key
                                ] += 1
                                unflagged_added = True

                        heap_sequence += 1

                        heap_item = (
                            character_count,
                            sentence_id,
                            heap_sequence,
                            {
                                "source_record_id": (
                                    source_record_id
                                ),
                                "sentence_id": sentence_id,
                                "sentence_character_count": (
                                    character_count
                                ),
                                "sentence_token_count": (
                                    token_count
                                ),
                                "segmentation_review_required": (
                                    row[
                                        "segmentation_review_required"
                                    ]
                                ),
                                "boundary_flags": "|".join(
                                    row["boundary_flags"]
                                ),
                                "sentence_text": row[
                                    "sentence_text"
                                ],
                            },
                        )

                        if (
                            len(longest_heap)
                            < LONGEST_SENTENCE_COUNT
                        ):
                            heapq.heappush(
                                longest_heap,
                                heap_item,
                            )
                        elif heap_item > longest_heap[0]:
                            heapq.heapreplace(
                                longest_heap,
                                heap_item,
                            )

            if (
                per_part_document_counts[part_number]
                != ROWS_PER_PART
            ):
                raise RuntimeError(
                    f"Expected {ROWS_PER_PART} records "
                    f"from part {part_number}, found "
                    f"{per_part_document_counts[part_number]}"
                )

    if problem_counts:
        raise RuntimeError(
            "Independent validation failed: "
            f"{dict(problem_counts)}"
        )

    expected_documents = (
        len(PART_NUMBERS) * ROWS_PER_PART
    )

    if metrics["documents_scanned"] != expected_documents:
        raise RuntimeError(
            f"Expected {expected_documents} documents, "
            f"found {metrics['documents_scanned']}"
        )

    flat_flag_examples = [
        example
        for flag in sorted(flag_examples)
        for example in flag_examples[flag]
    ]

    write_jsonl(
        inspection_path,
        inspection_rows,
    )
    write_jsonl(
        flag_examples_path,
        flat_flag_examples,
    )

    longest_rows = [
        item[3]
        for item in sorted(
            longest_heap,
            reverse=True,
        )
    ]

    with longest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "source_record_id",
            "sentence_id",
            "sentence_character_count",
            "sentence_token_count",
            "segmentation_review_required",
            "boundary_flags",
            "sentence_text",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(longest_rows)

    sentences_checksum = sha256_file(
        sentences_path
    )

    final_metrics = {
        "schema_version": (
            "phase_03_sentence_inventory_mini_metrics_v1"
        ),
        "experiment_name": experiment_name,
        "input_directory": str(DATA_ROOT),
        "parts_sampled": PART_NUMBERS,
        "rows_per_part": ROWS_PER_PART,
        "documents_scanned": metrics[
            "documents_scanned"
        ],
        "sentences_written": metrics[
            "sentences_written"
        ],
        "documents_with_zero_sentences": metrics[
            "documents_with_zero_sentences"
        ],
        "documents_with_parent_review": metrics[
            "documents_with_parent_review"
        ],
        "documents_with_segmentation_review": metrics[
            "documents_with_segmentation_review"
        ],
        "segmentation_review_sentences": metrics[
            "segmentation_review_sentences"
        ],
        "unique_sentence_ids": len(
            all_sentence_ids
        ),
        "validation_problem_counts": dict(
            problem_counts
        ),
        "boundary_flag_counts": dict(
            sorted(boundary_flag_counts.items())
        ),
        "parent_review_flag_counts": dict(
            sorted(parent_review_flag_counts.items())
        ),
        "terminal_punctuation_counts": dict(
            sorted(
                terminal_punctuation_counts.items()
            )
        ),
        "segmentation_method_counts": dict(
            sorted(
                segmentation_method_counts.items()
            )
        ),
        "per_part_document_counts": {
            str(part): per_part_document_counts[part]
            for part in PART_NUMBERS
        },
        "per_part_sentence_counts": {
            str(part): per_part_sentence_counts[part]
            for part in PART_NUMBERS
        },
        "sentence_character_count": {
            "minimum": min(
                sentence_character_counts,
                default=0,
            ),
            "median": percentile(
                sentence_character_counts,
                0.50,
            ),
            "p90": percentile(
                sentence_character_counts,
                0.90,
            ),
            "p95": percentile(
                sentence_character_counts,
                0.95,
            ),
            "p99": percentile(
                sentence_character_counts,
                0.99,
            ),
            "maximum": max(
                sentence_character_counts,
                default=0,
            ),
        },
        "sentence_token_count": {
            "minimum": min(
                sentence_token_counts,
                default=0,
            ),
            "median": percentile(
                sentence_token_counts,
                0.50,
            ),
            "p90": percentile(
                sentence_token_counts,
                0.90,
            ),
            "p95": percentile(
                sentence_token_counts,
                0.95,
            ),
            "p99": percentile(
                sentence_token_counts,
                0.99,
            ),
            "maximum": max(
                sentence_token_counts,
                default=0,
            ),
        },
        "sentences_per_document": {
            "minimum": min(
                sentences_per_document,
                default=0,
            ),
            "median": percentile(
                sentences_per_document,
                0.50,
            ),
            "p90": percentile(
                sentences_per_document,
                0.90,
            ),
            "p95": percentile(
                sentences_per_document,
                0.95,
            ),
            "p99": percentile(
                sentences_per_document,
                0.99,
            ),
            "maximum": max(
                sentences_per_document,
                default=0,
            ),
        },
        "sentences_jsonl_sha256": (
            sentences_checksum
        ),
        "important_note": (
            "This is a segmentation inventory, "
            "not a clean or gold sentence corpus."
        ),
    }

    metrics_path.write_text(
        json.dumps(
            final_metrics,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temp_root.rename(output_root)

    print("\nPhase 3 mini experiment completed.")
    print(f"Output: {output_root}")
    print(
        "Documents scanned: "
        f"{final_metrics['documents_scanned']:,}"
    )
    print(
        "Sentences written: "
        f"{final_metrics['sentences_written']:,}"
    )
    print(
        "Segmentation review sentences: "
        f"{final_metrics['segmentation_review_sentences']:,}"
    )
    print("Validation problems: 0")


if __name__ == "__main__":
    main()
