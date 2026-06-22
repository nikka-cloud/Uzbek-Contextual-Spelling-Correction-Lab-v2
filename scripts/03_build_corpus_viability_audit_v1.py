#!/usr/bin/env python3
"""
Build a sentence-level corpus viability audit.

Purpose
-------
Decide whether the current Uzbek corpus should be:

1. retained as the principal corpus;
2. filtered and supplemented with cleaner data; or
3. replaced as the principal clean-text source.

Sampling design
---------------
For each corpus part represented in segment_10k_v2:

- 70 ESTIMATION_RANDOM rows:
  unbiased rows used for the headline viability percentage;

- 30 DIAGNOSTIC_TARGETED rows:
  deliberately selected risk cases used only to understand damage.

Diagnostic rows must not be mixed into the headline corpus-health
percentage.

This script does not modify:
- the source corpus;
- Phase 2 normalized records;
- sentence inventories;
- the sentence segmenter.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


INVENTORY_ROOT = Path(
    "outputs/03_sentence_inventory/segment_10k_v2"
)

SENTENCES_PATH = INVENTORY_ROOT / "sentences.jsonl"
INVENTORY_METRICS_PATH = INVENTORY_ROOT / "metrics.json"

OUTPUT_ROOT = Path(
    "outputs/03_sentence_inventory/"
    "corpus_viability_audit_v1"
)

FULL_AUDIT_PATH = (
    OUTPUT_ROOT / "corpus_viability_audit_v1.csv"
)

SAMPLING_METRICS_PATH = (
    OUTPUT_ROOT / "sampling_metrics.json"
)

LABEL_GUIDE_PATH = OUTPUT_ROOT / "label_guide.md"
BATCH_ROOT = OUTPUT_ROOT / "review_batches"

RANDOM_SEED = 20260615

ESTIMATION_ROWS_PER_PART = 70
DIAGNOSTIC_ROWS_PER_PART = 30

TOTAL_ROWS_PER_PART = (
    ESTIMATION_ROWS_PER_PART
    + DIAGNOSTIC_ROWS_PER_PART
)

BATCH_SIZE = 50

DIAGNOSTIC_PLAN = [
    ("PARENT_REVIEW", 5),
    ("SEGMENTATION_REVIEW", 10),
    ("VERY_SHORT_CANDIDATE", 10),
    ("VERY_LONG_CANDIDATE", 5),
]

FIELDNAMES = [
    "audit_row_number",
    "sample_id",
    "sample_role",
    "sample_group",
    "part_number",
    "part_population_sentences",
    "sampling_weight",
    "source_record_id",
    "sentence_id",
    "sentence_index_in_record",
    "sentence_start_char",
    "sentence_end_char",
    "sentence_character_count",
    "sentence_token_count",
    "terminal_punctuation",
    "segmentation_method",
    "boundary_flags",
    "segmentation_review_required",
    "parent_review_flags",
    "previous_sentence",
    "sentence_text",
    "next_sentence",
    "human_health_label",
    "review_note",
]


def sentence_key(
    row: dict[str, Any],
) -> tuple[int, int]:
    """Stable sentence key within its parent document."""
    return (
        int(row["source_record_id"]),
        int(row["sentence_index_in_record"]),
    )


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 for one output file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def choose_unique_rows(
    *,
    candidates: list[dict[str, Any]],
    count: int,
    rng: random.Random,
    selected_keys: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Sample rows while preventing duplicate sentences."""
    available = [
        row
        for row in candidates
        if sentence_key(row) not in selected_keys
    ]

    if count <= 0 or not available:
        return []

    selected = rng.sample(
        available,
        min(count, len(available)),
    )

    for row in selected:
        selected_keys.add(sentence_key(row))

    return selected


def diagnostic_candidates(
    rows: list[dict[str, Any]],
    group_name: str,
) -> list[dict[str, Any]]:
    """Return candidates belonging to one diagnostic group."""
    if group_name == "PARENT_REVIEW":
        return [
            row
            for row in rows
            if row.get("parent_review_flags")
        ]

    if group_name == "SEGMENTATION_REVIEW":
        return [
            row
            for row in rows
            if row.get(
                "segmentation_review_required",
                False,
            )
        ]

    if group_name == "VERY_SHORT_CANDIDATE":
        return [
            row
            for row in rows
            if (
                int(row["sentence_token_count"]) <= 3
                or int(
                    row["sentence_character_count"]
                ) <= 20
            )
        ]

    if group_name == "VERY_LONG_CANDIDATE":
        return [
            row
            for row in rows
            if (
                int(row["sentence_token_count"]) >= 40
                or int(
                    row["sentence_character_count"]
                ) >= 300
            )
        ]

    raise ValueError(
        f"Unknown diagnostic group: {group_name}"
    )


def build_context_lookup(
    rows_by_source: dict[
        int,
        list[dict[str, Any]],
    ],
) -> dict[
    tuple[int, int],
    tuple[str, str],
]:
    """Store previous and next sentence text for each row."""
    lookup: dict[
        tuple[int, int],
        tuple[str, str],
    ] = {}

    for rows in rows_by_source.values():
        ordered = sorted(
            rows,
            key=lambda row: int(
                row["sentence_index_in_record"]
            ),
        )

        for position, row in enumerate(ordered):
            previous_text = ""

            if position > 0:
                previous_text = ordered[
                    position - 1
                ]["sentence_text"]

            next_text = ""

            if position + 1 < len(ordered):
                next_text = ordered[
                    position + 1
                ]["sentence_text"]

            lookup[sentence_key(row)] = (
                previous_text,
                next_text,
            )

    return lookup


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write audit rows with a fixed schema."""
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(rows)


def serialize_audit_row(
    *,
    row: dict[str, Any],
    audit_row_number: int,
    sample_role: str,
    sample_group: str,
    part_number: int,
    part_population: int,
    sampling_weight: float,
    context_lookup: dict[
        tuple[int, int],
        tuple[str, str],
    ],
) -> dict[str, Any]:
    """Convert an inventory sentence into an audit row."""
    previous_text, next_text = context_lookup[
        sentence_key(row)
    ]

    sample_id = (
        f"cva1-{audit_row_number:04d}-"
        f"{row['sentence_id']}"
    )

    return {
        "audit_row_number": audit_row_number,
        "sample_id": sample_id,
        "sample_role": sample_role,
        "sample_group": sample_group,
        "part_number": part_number,
        "part_population_sentences": (
            part_population
        ),
        "sampling_weight": (
            f"{sampling_weight:.8f}"
            if sampling_weight > 0
            else ""
        ),
        "source_record_id": (
            row["source_record_id"]
        ),
        "sentence_id": row["sentence_id"],
        "sentence_index_in_record": (
            row["sentence_index_in_record"]
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
        "terminal_punctuation": row.get(
            "terminal_punctuation",
            "",
        ),
        "segmentation_method": row.get(
            "segmentation_method",
            "",
        ),
        "boundary_flags": "|".join(
            row.get("boundary_flags", [])
        ),
        "segmentation_review_required": (
            row.get(
                "segmentation_review_required",
                False,
            )
        ),
        "parent_review_flags": "|".join(
            row.get("parent_review_flags", [])
        ),
        "previous_sentence": previous_text,
        "sentence_text": row["sentence_text"],
        "next_sentence": next_text,
        "human_health_label": "",
        "review_note": "",
    }


def main() -> None:
    if not SENTENCES_PATH.exists():
        raise FileNotFoundError(SENTENCES_PATH)

    if not INVENTORY_METRICS_PATH.exists():
        raise FileNotFoundError(
            INVENTORY_METRICS_PATH
        )

    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_ROOT}\n"
            "It was not overwritten."
        )

    inventory_metrics = json.loads(
        INVENTORY_METRICS_PATH.read_text(
            encoding="utf-8"
        )
    )

    parts = [
        int(part)
        for part in inventory_metrics[
            "parts_sampled"
        ]
    ]

    rows_by_part: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    rows_by_source: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    inventory_sentences_scanned = 0

    with SENTENCES_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            row = json.loads(line)
            inventory_sentences_scanned += 1

            source_record_id = int(
                row["source_record_id"]
            )

            part_number = (
                source_record_id // 100_000
            )

            if part_number not in parts:
                raise RuntimeError(
                    "Unexpected part inferred from "
                    f"source_record_id="
                    f"{source_record_id}: "
                    f"{part_number}"
                )

            rows_by_part[part_number].append(row)

            rows_by_source[
                source_record_id
            ].append(row)

    context_lookup = build_context_lookup(
        rows_by_source
    )

    part_population_counts = {
        part: len(rows_by_part[part])
        for part in parts
    }

    selected_records: list[
        tuple[
            dict[str, Any],
            str,
            str,
            int,
            int,
            float,
        ]
    ] = []

    sample_role_counts = Counter()
    sample_group_counts = Counter()
    part_sample_counts = Counter()

    for part_number in parts:
        part_rows = rows_by_part[part_number]

        if len(part_rows) < TOTAL_ROWS_PER_PART:
            raise RuntimeError(
                f"Part {part_number} has only "
                f"{len(part_rows)} sentences."
            )

        rng = random.Random(
            RANDOM_SEED + part_number
        )

        selected_keys: set[
            tuple[int, int]
        ] = set()

        estimation_rows = choose_unique_rows(
            candidates=part_rows,
            count=ESTIMATION_ROWS_PER_PART,
            rng=rng,
            selected_keys=selected_keys,
        )

        if (
            len(estimation_rows)
            != ESTIMATION_ROWS_PER_PART
        ):
            raise RuntimeError(
                "Could not create the estimation "
                f"sample for part {part_number}."
            )

        sampling_weight = (
            len(part_rows)
            / ESTIMATION_ROWS_PER_PART
        )

        for row in estimation_rows:
            selected_records.append(
                (
                    row,
                    "ESTIMATION_RANDOM",
                    "UNBIASED_RANDOM",
                    part_number,
                    len(part_rows),
                    sampling_weight,
                )
            )

            sample_role_counts[
                "ESTIMATION_RANDOM"
            ] += 1

            sample_group_counts[
                "UNBIASED_RANDOM"
            ] += 1

            part_sample_counts[
                part_number
            ] += 1

        diagnostic_selected = 0

        for (
            group_name,
            requested_count,
        ) in DIAGNOSTIC_PLAN:
            candidates = diagnostic_candidates(
                part_rows,
                group_name,
            )

            selected = choose_unique_rows(
                candidates=candidates,
                count=requested_count,
                rng=rng,
                selected_keys=selected_keys,
            )

            diagnostic_selected += len(selected)

            for row in selected:
                selected_records.append(
                    (
                        row,
                        "DIAGNOSTIC_TARGETED",
                        group_name,
                        part_number,
                        len(part_rows),
                        0.0,
                    )
                )

                sample_role_counts[
                    "DIAGNOSTIC_TARGETED"
                ] += 1

                sample_group_counts[
                    group_name
                ] += 1

                part_sample_counts[
                    part_number
                ] += 1

        missing_diagnostic_rows = (
            DIAGNOSTIC_ROWS_PER_PART
            - diagnostic_selected
        )

        if missing_diagnostic_rows > 0:
            fill_rows = choose_unique_rows(
                candidates=part_rows,
                count=missing_diagnostic_rows,
                rng=rng,
                selected_keys=selected_keys,
            )

            for row in fill_rows:
                selected_records.append(
                    (
                        row,
                        "DIAGNOSTIC_TARGETED",
                        "DIAGNOSTIC_FILL",
                        part_number,
                        len(part_rows),
                        0.0,
                    )
                )

                sample_role_counts[
                    "DIAGNOSTIC_TARGETED"
                ] += 1

                sample_group_counts[
                    "DIAGNOSTIC_FILL"
                ] += 1

                part_sample_counts[
                    part_number
                ] += 1

        if (
            part_sample_counts[part_number]
            != TOTAL_ROWS_PER_PART
        ):
            raise RuntimeError(
                f"Part {part_number} produced "
                f"{part_sample_counts[part_number]} "
                "rows instead of "
                f"{TOTAL_ROWS_PER_PART}."
            )

    expected_total_rows = (
        len(parts) * TOTAL_ROWS_PER_PART
    )

    if (
        len(selected_records)
        != expected_total_rows
    ):
        raise RuntimeError(
            f"Expected {expected_total_rows} rows, "
            f"found {len(selected_records)}."
        )

    unique_keys = {
        sentence_key(record[0])
        for record in selected_records
    }

    if len(unique_keys) != len(selected_records):
        raise RuntimeError(
            "Duplicate sentence found in audit."
        )

    shuffle_rng = random.Random(RANDOM_SEED)
    shuffle_rng.shuffle(selected_records)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    BATCH_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    audit_rows: list[dict[str, Any]] = []

    for audit_row_number, record in enumerate(
        selected_records,
        start=1,
    ):
        (
            row,
            sample_role,
            sample_group,
            part_number,
            part_population,
            sampling_weight,
        ) = record

        audit_rows.append(
            serialize_audit_row(
                row=row,
                audit_row_number=(
                    audit_row_number
                ),
                sample_role=sample_role,
                sample_group=sample_group,
                part_number=part_number,
                part_population=(
                    part_population
                ),
                sampling_weight=(
                    sampling_weight
                ),
                context_lookup=context_lookup,
            )
        )

    write_csv(
        FULL_AUDIT_PATH,
        audit_rows,
    )

    batch_sizes = []

    for batch_number, start in enumerate(
        range(
            0,
            len(audit_rows),
            BATCH_SIZE,
        ),
        start=1,
    ):
        batch_rows = audit_rows[
            start:start + BATCH_SIZE
        ]

        batch_path = (
            BATCH_ROOT
            / f"batch-{batch_number:02d}.csv"
        )

        write_csv(
            batch_path,
            batch_rows,
        )

        batch_sizes.append(len(batch_rows))

    label_guide = """# Corpus viability audit label guide

Only rows with `sample_role=ESTIMATION_RANDOM` are used for the
headline corpus viability percentage.

Rows with `sample_role=DIAGNOSTIC_TARGETED` are used only to understand
the types of damage present.

## Allowed `human_health_label` values

### HEALTHY
Coherent Uzbek prose with no material spelling, OCR, segmentation,
formatting, or language-mixing problem.

### MINOR_DAMAGE_BUT_USABLE
The meaning and sentence structure are reliable, but a small isolated
surface error exists. This row still requires safe repair or manual
verification before becoming a clean target.

### HEAVY_CORRUPTION
Several broken tokens, Roman-expansion corruption, severe OCR damage,
or text that cannot safely serve as a clean target.

### FORMAT_OR_LIST
Bibliography, table, index, heading, author list, metadata, glossary,
formula-like content, or another non-prose structure.

### FRAGMENT
The row is not a complete or independently useful sentence.

### MIXED_OR_FOREIGN_TEXT
A substantial amount of the row is non-Uzbek or mixed-language text.

### UNCERTAIN
The row and its neighbouring context are insufficient for a confident
decision.

Do not correct text in the audit file. Enter only the label and,
where useful, a brief review note.
"""

    LABEL_GUIDE_PATH.write_text(
        label_guide,
        encoding="utf-8",
    )

    sampling_metrics = {
        "experiment_name": (
            "corpus_viability_audit_v1"
        ),
        "source_inventory": str(
            INVENTORY_ROOT
        ),
        "inventory_sentences_scanned": (
            inventory_sentences_scanned
        ),
        "parts_sampled": parts,
        "part_population_sentences": (
            part_population_counts
        ),
        "estimation_rows_per_part": (
            ESTIMATION_ROWS_PER_PART
        ),
        "diagnostic_rows_per_part": (
            DIAGNOSTIC_ROWS_PER_PART
        ),
        "total_rows_per_part": (
            TOTAL_ROWS_PER_PART
        ),
        "total_audit_rows": len(
            audit_rows
        ),
        "sample_role_counts": dict(
            sorted(
                sample_role_counts.items()
            )
        ),
        "sample_group_counts": dict(
            sorted(
                sample_group_counts.items()
            )
        ),
        "part_sample_counts": {
            str(part): (
                part_sample_counts[part]
            )
            for part in parts
        },
        "batch_size": BATCH_SIZE,
        "batch_sizes": batch_sizes,
        "random_seed": RANDOM_SEED,
        "full_audit_sha256": sha256_file(
            FULL_AUDIT_PATH
        ),
    }

    SAMPLING_METRICS_PATH.write_text(
        json.dumps(
            sampling_metrics,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n--- corpus viability audit created ---"
    )

    print(
        "inventory_sentences_scanned: "
        f"{inventory_sentences_scanned:,}"
    )

    print(f"parts_sampled: {parts}")

    print(
        "total_audit_rows: "
        f"{len(audit_rows):,}"
    )

    print(
        "estimation_random_rows: "
        f"{sample_role_counts['ESTIMATION_RANDOM']:,}"
    )

    print(
        "diagnostic_targeted_rows: "
        f"{sample_role_counts['DIAGNOSTIC_TARGETED']:,}"
    )

    print(
        "sample_group_counts: "
        f"{dict(sorted(sample_group_counts.items()))}"
    )

    print(f"batch_sizes: {batch_sizes}")
    print(f"output: {OUTPUT_ROOT}")

    print("\n--- first audit rows ---")

    for row in audit_rows[:8]:
        print("\n" + "=" * 100)

        print(
            f"row={row['audit_row_number']} "
            f"role={row['sample_role']} "
            f"group={row['sample_group']} "
            f"part={row['part_number']}"
        )

        print(
            row["sentence_text"][:500]
        )

    print(
        "\nAllowed human_health_label values:"
    )

    print("HEALTHY")
    print("MINOR_DAMAGE_BUT_USABLE")
    print("HEAVY_CORRUPTION")
    print("FORMAT_OR_LIST")
    print("FRAGMENT")
    print("MIXED_OR_FOREIGN_TEXT")
    print("UNCERTAIN")

    print(
        "\nNo corpus, inventory, or segmenter "
        "was modified."
    )


if __name__ == "__main__":
    main()
