#!/usr/bin/env python3
"""
Prepare, inspect, merge, and summarize the 700 unbiased rows from the
Phase 3 corpus-viability audit.

The 300 DIAGNOSTIC_TARGETED rows are intentionally excluded from all
headline corpus-health percentages.

Modes
-----
prepare:
    Extract the 700 ESTIMATION_RANDOM rows and create deterministic
    human-review batches.

show:
    Print one review batch with previous/current/next sentence context.

status:
    Report labeling progress without changing files.

merge:
    Validate batch labels and merge them into a new labeled audit file.
    The original audit remains unchanged.

summarize:
    Require all 700 estimation rows to be labeled, calculate:
    - sampling-weighted overall percentages;
    - approximate stratified 95% confidence intervals;
    - projected usable sentence counts;
    - part-by-part quality;
    - corpus-retention decision support.

This script never changes:
- source corpus;
- Phase 2 normalized records;
- Phase 3 sentence inventories;
- sentence segmenter;
- original viability audit CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(
    "outputs/03_sentence_inventory/"
    "corpus_viability_audit_v1"
)

SOURCE_AUDIT_PATH = (
    ROOT / "corpus_viability_audit_v1.csv"
)

ESTIMATION_MASTER_PATH = (
    ROOT / "estimation_random_review_master_v1.csv"
)

BATCH_ROOT = (
    ROOT / "estimation_random_review_batches_v1"
)

MERGED_AUDIT_PATH = (
    ROOT
    / "corpus_viability_audit_v1."
    "with_estimation_labels.csv"
)

RESULT_ROOT = (
    ROOT / "estimation_weighted_results_v1"
)

UNLABELED_PATH = (
    ROOT / "estimation_random_unlabeled_rows_v1.csv"
)

BATCH_SIZE = 25
EXPECTED_ESTIMATION_ROWS = 700
EXPECTED_ROWS_PER_PART = 70

VALID_HEALTH_LABELS = {
    "HEALTHY",
    "MINOR_DAMAGE_BUT_USABLE",
    "HEAVY_CORRUPTION",
    "FORMAT_OR_LIST",
    "FRAGMENT",
    "MIXED_OR_FOREIGN_TEXT",
    "UNCERTAIN",
}

DIRECTLY_USABLE_LABELS = {
    "HEALTHY",
}

FILTERABLE_USABLE_LABELS = {
    "MINOR_DAMAGE_BUT_USABLE",
}

UNUSABLE_LABELS = {
    "HEAVY_CORRUPTION",
    "FORMAT_OR_LIST",
    "FRAGMENT",
    "MIXED_OR_FOREIGN_TEXT",
}


def read_csv(
    path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    """Read a CSV and preserve its exact field order."""
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise RuntimeError(
                f"CSV has no header: {path}"
            )

        return list(reader), list(reader.fieldnames)


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Write deterministic UTF-8 CSV output."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def clean_label(value: str | None) -> str:
    """Normalize only the manually entered label field."""
    return (value or "").strip().upper()


def estimation_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Return only unbiased estimation rows."""
    selected = [
        row
        for row in rows
        if row.get("sample_role")
        == "ESTIMATION_RANDOM"
    ]

    selected.sort(
        key=lambda row: int(
            row["audit_row_number"]
        )
    )

    return selected


def validate_estimation_design(
    rows: list[dict[str, str]],
) -> None:
    """Validate the intended 700-row stratified sample."""
    if len(rows) != EXPECTED_ESTIMATION_ROWS:
        raise RuntimeError(
            "Expected exactly "
            f"{EXPECTED_ESTIMATION_ROWS} "
            "ESTIMATION_RANDOM rows, found "
            f"{len(rows)}."
        )

    sample_ids = [
        row["sample_id"]
        for row in rows
    ]

    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError(
            "Duplicate sample_id values found."
        )

    part_counts = Counter(
        int(row["part_number"])
        for row in rows
    )

    unexpected = {
        part: count
        for part, count in part_counts.items()
        if count != EXPECTED_ROWS_PER_PART
    }

    if unexpected:
        raise RuntimeError(
            "Expected 70 estimation rows per part. "
            f"Unexpected counts: {unexpected}"
        )

    for row in rows:
        if row.get("sample_group") != "UNBIASED_RANDOM":
            raise RuntimeError(
                "An ESTIMATION_RANDOM row does not "
                "belong to UNBIASED_RANDOM: "
                f"{row['sample_id']}"
            )

        if not row.get("sampling_weight"):
            raise RuntimeError(
                "Missing sampling_weight for "
                f"{row['sample_id']}"
            )

        weight = float(row["sampling_weight"])

        if not math.isfinite(weight) or weight <= 0:
            raise RuntimeError(
                "Invalid sampling_weight for "
                f"{row['sample_id']}: {weight}"
            )


def review_guide_text() -> str:
    return """# Corpus Viability Estimation Review Guide

Only the 700 `ESTIMATION_RANDOM` rows contribute to the headline
corpus-health percentage.

## Labels

### HEALTHY

Use when the sentence is suitable as clean source text without a
meaningful textual repair.

Minor stylistic awkwardness is acceptable. A sentence does not need
to be elegant; it must be understandable, sentence-like, and
sufficiently clean for controlled typo generation.

### MINOR_DAMAGE_BUT_USABLE

Use when the intended sentence is clear and mostly reliable, but it
contains limited damage that could be detected or filtered.

Examples:

- one apostrophe gap;
- one merged token;
- one minor OCR character error;
- a small casing problem;
- one obvious typo that does not destroy sentence meaning.

This label means usable only after strict filtering or repair.

### HEAVY_CORRUPTION

Use when several words are damaged, Roman-numeral conversion is
widespread, OCR destruction is substantial, or the intended clean
sentence cannot be trusted.

### FORMAT_OR_LIST

Use for references, tables, glossaries, index entries, score lists,
bibliographies, headings joined together, metadata, formulas, or
other material that is not ordinary sentence prose.

### FRAGMENT

Use when the selected row is only a sentence fragment and cannot
stand as reliable clean sentence data.

### MIXED_OR_FOREIGN_TEXT

Use when a large or central portion is in another language, mixed
script, transliteration, or foreign material unsuitable for the
intended Uzbek training source.

### UNCERTAIN

Use only when the surrounding context is insufficient to decide.

## Important distinctions

- Do not mark every spelling error as HEAVY_CORRUPTION.
- One limited and interpretable error can be
  MINOR_DAMAGE_BUT_USABLE.
- Lists and metadata are FORMAT_OR_LIST even when their individual
  tokens are correctly spelled.
- The previous and next sentences are context only. Label the current
  `sentence_text`.
"""


def prepare_batches() -> None:
    """
    Extract the 700 unbiased rows and create 25-row review batches.
    """
    all_rows, fieldnames = read_csv(
        SOURCE_AUDIT_PATH
    )
    rows = estimation_rows(all_rows)
    validate_estimation_design(rows)

    if BATCH_ROOT.exists():
        raise FileExistsError(
            f"Review batches already exist: {BATCH_ROOT}\n"
            "They were not overwritten."
        )

    if ESTIMATION_MASTER_PATH.exists():
        raise FileExistsError(
            "Estimation review master already exists: "
            f"{ESTIMATION_MASTER_PATH}"
        )

    BATCH_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        ESTIMATION_MASTER_PATH,
        rows,
        fieldnames,
    )

    batch_sizes: list[int] = []

    for start in range(
        0,
        len(rows),
        BATCH_SIZE,
    ):
        batch_rows = rows[
            start:start + BATCH_SIZE
        ]
        batch_number = (
            start // BATCH_SIZE
        ) + 1

        batch_path = (
            BATCH_ROOT
            / f"batch-{batch_number:02d}.csv"
        )

        write_csv(
            batch_path,
            batch_rows,
            fieldnames,
        )

        batch_sizes.append(
            len(batch_rows)
        )

    guide_path = (
        ROOT
        / "estimation_random_label_guide_v1.md"
    )
    guide_path.write_text(
        review_guide_text(),
        encoding="utf-8",
    )

    metrics = {
        "source_audit": str(
            SOURCE_AUDIT_PATH
        ),
        "estimation_master": str(
            ESTIMATION_MASTER_PATH
        ),
        "batch_root": str(BATCH_ROOT),
        "estimation_rows": len(rows),
        "batch_size": BATCH_SIZE,
        "batch_count": len(batch_sizes),
        "batch_sizes": batch_sizes,
        "part_counts": dict(
            sorted(
                Counter(
                    int(row["part_number"])
                    for row in rows
                ).items()
            )
        ),
        "diagnostic_rows_excluded": (
            len(all_rows) - len(rows)
        ),
    }

    (
        ROOT
        / "estimation_random_review_metrics_v1.json"
    ).write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n--- estimation review prepared ---"
    )
    print(
        f"source_audit_rows: {len(all_rows):,}"
    )
    print(
        f"estimation_random_rows: {len(rows):,}"
    )
    print(
        "diagnostic_rows_excluded: "
        f"{len(all_rows) - len(rows):,}"
    )
    print(
        f"review_batches: {len(batch_sizes)}"
    )
    print(
        f"batch_sizes: {batch_sizes}"
    )
    print(
        f"batch_root: {BATCH_ROOT}"
    )
    print(
        f"label_guide: {guide_path}"
    )
    print(
        "\nPREPARATION STATUS: PASS"
    )


def clip(
    value: str | None,
    maximum: int,
) -> str:
    """Keep terminal review output readable."""
    text = (value or "").strip()

    if len(text) <= maximum:
        return text

    return text[:maximum].rstrip() + " ..."


def show_batch(
    batch_number: int,
) -> None:
    """Print one review batch with context."""
    batch_path = (
        BATCH_ROOT
        / f"batch-{batch_number:02d}.csv"
    )

    rows, _ = read_csv(batch_path)

    print(
        "\n--- CORPUS VIABILITY ESTIMATION "
        f"BATCH {batch_number:02d} ---"
    )
    print(f"path: {batch_path}")
    print(f"rows: {len(rows)}")

    for number, row in enumerate(
        rows,
        start=1,
    ):
        label = clean_label(
            row.get("human_health_label")
        )

        print("\n" + "=" * 115)
        print(f"REVIEW_ROW: {number}")
        print(
            f"audit_row_number: "
            f"{row['audit_row_number']}"
        )
        print(
            f"sample_id: {row['sample_id']}"
        )
        print(
            f"part: {row['part_number']} | "
            f"source: {row['source_record_id']} | "
            f"sentence_index: "
            f"{row['sentence_index_in_record']}"
        )
        print(
            f"characters: "
            f"{row['sentence_character_count']} | "
            f"tokens: {row['sentence_token_count']}"
        )
        print(
            "boundary_flags: "
            f"{row.get('boundary_flags') or 'NONE'}"
        )
        print(
            "parent_review_flags: "
            f"{row.get('parent_review_flags') or 'NONE'}"
        )
        print(
            "current_label: "
            f"{label or 'UNLABELED'}"
        )

        print("\nPREVIOUS SENTENCE")
        print(
            clip(
                row.get("previous_sentence"),
                500,
            )
            or "[NONE]"
        )

        print("\nCURRENT SENTENCE — LABEL THIS")
        print(
            clip(
                row.get("sentence_text"),
                1_200,
            )
        )

        print("\nNEXT SENTENCE")
        print(
            clip(
                row.get("next_sentence"),
                500,
            )
            or "[NONE]"
        )

    print(
        "\nAllowed human_health_label values:"
    )
    for label in sorted(
        VALID_HEALTH_LABELS
    ):
        print(label)

    print(
        "\nOnly CURRENT SENTENCE is labeled. "
        "Previous and next sentences provide context."
    )
    print(
        "No corpus or inventory was modified."
    )


def load_all_batch_rows(
) -> tuple[list[dict[str, str]], list[str]]:
    """Read every prepared batch in numeric order."""
    paths = sorted(
        BATCH_ROOT.glob("batch-*.csv")
    )

    if not paths:
        raise RuntimeError(
            f"No review batches found in {BATCH_ROOT}"
        )

    combined: list[dict[str, str]] = []
    fieldnames: list[str] | None = None

    for path in paths:
        rows, current_fields = read_csv(path)

        if fieldnames is None:
            fieldnames = current_fields
        elif current_fields != fieldnames:
            raise RuntimeError(
                "Batch CSV schemas differ: "
                f"{path}"
            )

        combined.extend(rows)

    if fieldnames is None:
        raise RuntimeError(
            "No CSV field names found."
        )

    return combined, fieldnames


def validate_batch_identity(
    rows: list[dict[str, str]],
) -> None:
    """Ensure batches still represent the exact 700-row sample."""
    master_rows, _ = read_csv(
        ESTIMATION_MASTER_PATH
    )
    validate_estimation_design(master_rows)

    expected_ids = {
        row["sample_id"]
        for row in master_rows
    }

    actual_ids = [
        row["sample_id"]
        for row in rows
    ]

    if len(actual_ids) != EXPECTED_ESTIMATION_ROWS:
        raise RuntimeError(
            "Review batches must contain exactly "
            f"{EXPECTED_ESTIMATION_ROWS} rows; "
            f"found {len(actual_ids)}."
        )

    if len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError(
            "Duplicate sample IDs found in batches."
        )

    actual_set = set(actual_ids)

    missing = expected_ids - actual_set
    unexpected = actual_set - expected_ids

    if missing or unexpected:
        raise RuntimeError(
            "Review-batch identity mismatch. "
            f"missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )


def labeling_status() -> None:
    """Report progress without changing anything."""
    rows, _ = load_all_batch_rows()
    validate_batch_identity(rows)

    total_counts = Counter()
    invalid_rows: list[dict[str, str]] = []

    print("\n--- labeling status by batch ---")

    for path in sorted(
        BATCH_ROOT.glob("batch-*.csv")
    ):
        batch_rows, _ = read_csv(path)
        batch_counts = Counter()

        for row in batch_rows:
            label = clean_label(
                row.get("human_health_label")
            )

            if not label:
                batch_counts["UNLABELED"] += 1
                total_counts["UNLABELED"] += 1
            elif label in VALID_HEALTH_LABELS:
                batch_counts[label] += 1
                total_counts[label] += 1
            else:
                batch_counts["INVALID"] += 1
                total_counts["INVALID"] += 1
                invalid_rows.append(row)

        print(
            f"{path.name}: "
            f"labeled="
            f"{len(batch_rows) - batch_counts['UNLABELED'] - batch_counts['INVALID']}"
            f"/{len(batch_rows)} "
            f"unlabeled={batch_counts['UNLABELED']} "
            f"invalid={batch_counts['INVALID']}"
        )

    print("\n--- cumulative labeling status ---")
    print(
        f"total_rows: {len(rows)}"
    )
    print(
        "counts: "
        f"{dict(sorted(total_counts.items()))}"
    )

    if invalid_rows:
        print("\nINVALID LABEL ROWS:")
        for row in invalid_rows[:20]:
            print(
                row["sample_id"],
                repr(
                    row.get(
                        "human_health_label",
                        "",
                    )
                ),
            )


def merge_labels() -> None:
    """
    Merge batch labels into a new audit CSV.

    The original viability audit is never overwritten.
    """
    batch_rows, _ = load_all_batch_rows()
    validate_batch_identity(batch_rows)

    labels_by_sample_id: dict[
        str,
        tuple[str, str],
    ] = {}

    invalid: list[
        tuple[str, str]
    ] = []
    unlabeled: list[dict[str, str]] = []

    for row in batch_rows:
        sample_id = row["sample_id"]
        label = clean_label(
            row.get("human_health_label")
        )
        note = (
            row.get("review_note") or ""
        ).strip()

        if not label:
            unlabeled.append(row)
            continue

        if label not in VALID_HEALTH_LABELS:
            invalid.append(
                (
                    sample_id,
                    row.get(
                        "human_health_label",
                        "",
                    ),
                )
            )
            continue

        labels_by_sample_id[sample_id] = (
            label,
            note,
        )

    if invalid:
        raise RuntimeError(
            "Invalid manual labels found: "
            f"{invalid[:20]}"
        )

    all_rows, fieldnames = read_csv(
        SOURCE_AUDIT_PATH
    )

    updated_rows = 0

    for row in all_rows:
        if (
            row.get("sample_role")
            != "ESTIMATION_RANDOM"
        ):
            continue

        sample_id = row["sample_id"]

        if sample_id not in labels_by_sample_id:
            row["human_health_label"] = ""
            row["review_note"] = ""
            continue

        label, note = labels_by_sample_id[
            sample_id
        ]

        row["human_health_label"] = label
        row["review_note"] = note
        updated_rows += 1

    write_csv(
        MERGED_AUDIT_PATH,
        all_rows,
        fieldnames,
    )

    write_csv(
        UNLABELED_PATH,
        unlabeled,
        fieldnames,
    )

    print(
        "\n--- estimation labels merged ---"
    )
    print(
        f"labels_merged: {updated_rows:,}"
    )
    print(
        f"unlabeled_rows: {len(unlabeled):,}"
    )
    print(
        f"merged_audit: {MERGED_AUDIT_PATH}"
    )
    print(
        f"unlabeled_export: {UNLABELED_PATH}"
    )

    if unlabeled:
        print(
            "\nMERGE STATUS: PARTIAL"
        )
        print(
            "Summary will remain blocked until "
            "all 700 rows are labeled."
        )
    else:
        print(
            "\nMERGE STATUS: COMPLETE"
        )


def stratified_binary_estimate(
    rows: list[dict[str, str]],
    positive_labels: set[str],
) -> dict[str, float]:
    """
    Calculate a stratified proportion and approximate 95% CI.

    Each corpus part is treated as a stratum. The finite population
    correction is included.
    """
    by_part: dict[
        int,
        list[dict[str, str]],
    ] = defaultdict(list)

    population_by_part: dict[
        int,
        int,
    ] = {}

    for row in rows:
        part = int(row["part_number"])
        population = int(
            row["part_population_sentences"]
        )

        by_part[part].append(row)

        previous_population = (
            population_by_part.get(part)
        )

        if (
            previous_population is not None
            and previous_population != population
        ):
            raise RuntimeError(
                "Inconsistent part population for "
                f"part {part}."
            )

        population_by_part[part] = population

    total_population = sum(
        population_by_part.values()
    )

    estimate = 0.0
    variance = 0.0

    for part, part_rows in by_part.items():
        population = population_by_part[part]
        sample_n = len(part_rows)

        positives = sum(
            clean_label(
                row["human_health_label"]
            )
            in positive_labels
            for row in part_rows
        )

        proportion = positives / sample_n
        stratum_weight = (
            population / total_population
        )

        estimate += (
            stratum_weight * proportion
        )

        if sample_n > 1:
            sample_variance = (
                sample_n
                / (sample_n - 1)
                * proportion
                * (1 - proportion)
            )

            sampling_fraction = (
                sample_n / population
            )

            variance += (
                stratum_weight**2
                * (1 - sampling_fraction)
                * sample_variance
                / sample_n
            )

    standard_error = math.sqrt(
        max(variance, 0.0)
    )

    lower = max(
        0.0,
        estimate - 1.96 * standard_error,
    )
    upper = min(
        1.0,
        estimate + 1.96 * standard_error,
    )

    return {
        "proportion": estimate,
        "percentage": estimate * 100,
        "standard_error": standard_error,
        "ci95_lower_percentage": lower * 100,
        "ci95_upper_percentage": upper * 100,
        "projected_sentence_count": round(
            estimate * total_population
        ),
        "population_sentences": total_population,
    }


def percentage(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator * 100


def corpus_decision(
    *,
    healthy_percentage: float,
    usable_percentage: float,
    heavy_percentage: float,
    uncertain_percentage: float,
    minimum_part_usable_percentage: float,
    part_usable_spread: float,
) -> tuple[str, list[str]]:
    """
    Produce transparent engineering decision support.

    These thresholds are project policy, not universal scientific law.
    """
    reasons: list[str] = []

    uneven_parts = (
        minimum_part_usable_percentage < 70.0
        or part_usable_spread > 20.0
    )

    if uncertain_percentage > 5.0:
        decision = (
            "INCONCLUSIVE_REVIEW_UNCERTAIN_ROWS"
        )
        reasons.append(
            "More than 5% of the weighted sample "
            "is still uncertain."
        )

    elif (
        usable_percentage >= 85.0
        and healthy_percentage >= 60.0
        and heavy_percentage <= 10.0
    ):
        if uneven_parts:
            decision = (
                "RETAIN_GOOD_PARTS_FILTER_BAD_PARTS_"
                "AND_SUPPLEMENT"
            )
            reasons.append(
                "Overall usability is strong, but "
                "quality differs materially by part."
            )
        else:
            decision = (
                "RETAIN_AS_PRINCIPAL_FILTERED_CORPUS"
            )
            reasons.append(
                "Weighted usability is at least 85%, "
                "healthy text is at least 60%, and "
                "heavy corruption is at most 10%."
            )

    elif usable_percentage >= 70.0:
        decision = (
            "RETAIN_AS_SECONDARY_AND_SUPPLEMENT"
        )
        reasons.append(
            "The usable share is meaningful but not "
            "strong enough for an independent clean "
            "principal corpus."
        )

    else:
        decision = (
            "REPLACE_AS_PRINCIPAL_CLEAN_TEXT_SOURCE"
        )
        reasons.append(
            "Less than 70% of the weighted sample is "
            "usable for controlled clean-text training."
        )

    if uneven_parts:
        reasons.append(
            "Part-level quality is uneven: either a "
            "part is below 70% usable or the usable-rate "
            "spread exceeds 20 percentage points."
        )

    return decision, reasons


def summarize() -> None:
    """Calculate weighted overall and part-level results."""
    all_rows, _ = read_csv(
        MERGED_AUDIT_PATH
    )

    rows = estimation_rows(all_rows)
    validate_estimation_design(rows)

    invalid: list[
        tuple[str, str]
    ] = []
    unlabeled: list[str] = []

    for row in rows:
        label = clean_label(
            row.get("human_health_label")
        )

        if not label:
            unlabeled.append(row["sample_id"])
        elif label not in VALID_HEALTH_LABELS:
            invalid.append(
                (
                    row["sample_id"],
                    label,
                )
            )

    if invalid:
        raise RuntimeError(
            "Invalid labels prevent summary: "
            f"{invalid[:20]}"
        )

    if unlabeled:
        raise RuntimeError(
            "All 700 estimation rows must be labeled "
            "before summarization. "
            f"Unlabeled rows: {len(unlabeled)}"
        )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_results = {
        label: stratified_binary_estimate(
            rows,
            {label},
        )
        for label in sorted(
            VALID_HEALTH_LABELS
        )
    }

    healthy_result = (
        stratified_binary_estimate(
            rows,
            DIRECTLY_USABLE_LABELS,
        )
    )

    minor_result = (
        stratified_binary_estimate(
            rows,
            FILTERABLE_USABLE_LABELS,
        )
    )

    usable_result = (
        stratified_binary_estimate(
            rows,
            DIRECTLY_USABLE_LABELS
            | FILTERABLE_USABLE_LABELS,
        )
    )

    unusable_result = (
        stratified_binary_estimate(
            rows,
            UNUSABLE_LABELS,
        )
    )

    heavy_result = (
        label_results["HEAVY_CORRUPTION"]
    )

    uncertain_result = (
        label_results["UNCERTAIN"]
    )

    by_part: dict[
        int,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        by_part[
            int(row["part_number"])
        ].append(row)

    part_rows: list[dict[str, Any]] = []
    part_usable_percentages: list[float] = []

    for part in sorted(by_part):
        current = by_part[part]
        counts = Counter(
            clean_label(
                row["human_health_label"]
            )
            for row in current
        )

        sample_n = len(current)
        population = int(
            current[0][
                "part_population_sentences"
            ]
        )

        healthy_count = counts["HEALTHY"]
        minor_count = counts[
            "MINOR_DAMAGE_BUT_USABLE"
        ]

        usable_count = (
            healthy_count + minor_count
        )

        unusable_count = sum(
            counts[label]
            for label in UNUSABLE_LABELS
        )

        usable_pct = percentage(
            usable_count,
            sample_n,
        )
        part_usable_percentages.append(
            usable_pct
        )

        part_rows.append(
            {
                "part_number": part,
                "part_population_sentences": population,
                "sample_rows": sample_n,
                "healthy_count": healthy_count,
                "healthy_percentage": percentage(
                    healthy_count,
                    sample_n,
                ),
                "minor_damage_usable_count": (
                    minor_count
                ),
                "minor_damage_usable_percentage": (
                    percentage(
                        minor_count,
                        sample_n,
                    )
                ),
                "usable_total_count": usable_count,
                "usable_total_percentage": (
                    usable_pct
                ),
                "unusable_total_count": (
                    unusable_count
                ),
                "unusable_total_percentage": (
                    percentage(
                        unusable_count,
                        sample_n,
                    )
                ),
                "heavy_corruption_count": (
                    counts[
                        "HEAVY_CORRUPTION"
                    ]
                ),
                "format_or_list_count": (
                    counts["FORMAT_OR_LIST"]
                ),
                "fragment_count": (
                    counts["FRAGMENT"]
                ),
                "mixed_or_foreign_count": (
                    counts[
                        "MIXED_OR_FOREIGN_TEXT"
                    ]
                ),
                "uncertain_count": (
                    counts["UNCERTAIN"]
                ),
                "uncertain_percentage": (
                    percentage(
                        counts["UNCERTAIN"],
                        sample_n,
                    )
                ),
            }
        )

    minimum_part_usable = min(
        part_usable_percentages
    )
    maximum_part_usable = max(
        part_usable_percentages
    )
    usable_spread = (
        maximum_part_usable
        - minimum_part_usable
    )

    decision, decision_reasons = (
        corpus_decision(
            healthy_percentage=(
                healthy_result["percentage"]
            ),
            usable_percentage=(
                usable_result["percentage"]
            ),
            heavy_percentage=(
                heavy_result["percentage"]
            ),
            uncertain_percentage=(
                uncertain_result["percentage"]
            ),
            minimum_part_usable_percentage=(
                minimum_part_usable
            ),
            part_usable_spread=(
                usable_spread
            ),
        )
    )

    weighted_rows = []

    for label in sorted(
        VALID_HEALTH_LABELS
    ):
        result = label_results[label]
        weighted_rows.append(
            {
                "health_label": label,
                **result,
            }
        )

    write_csv(
        RESULT_ROOT
        / "weighted_label_distribution.csv",
        weighted_rows,
        [
            "health_label",
            "proportion",
            "percentage",
            "standard_error",
            "ci95_lower_percentage",
            "ci95_upper_percentage",
            "projected_sentence_count",
            "population_sentences",
        ],
    )

    part_fieldnames = list(
        part_rows[0].keys()
    )

    write_csv(
        RESULT_ROOT
        / "part_health_summary.csv",
        part_rows,
        part_fieldnames,
    )

    summary = {
        "experiment_name": (
            "corpus_viability_estimation_"
            "weighted_v1"
        ),
        "source_audit": str(
            MERGED_AUDIT_PATH
        ),
        "estimation_rows": len(rows),
        "diagnostic_rows_used_in_headline": 0,
        "population_sentences": (
            usable_result[
                "population_sentences"
            ]
        ),
        "weighted_label_distribution": (
            label_results
        ),
        "key_metrics": {
            "healthy": healthy_result,
            "minor_damage_but_usable": (
                minor_result
            ),
            "usable_total": usable_result,
            "unusable_total": unusable_result,
        },
        "part_quality": {
            "minimum_usable_percentage": (
                minimum_part_usable
            ),
            "maximum_usable_percentage": (
                maximum_part_usable
            ),
            "usable_percentage_spread": (
                usable_spread
            ),
        },
        "decision": decision,
        "decision_reasons": decision_reasons,
        "decision_thresholds": {
            "principal_minimum_usable_pct": 85,
            "principal_minimum_healthy_pct": 60,
            "principal_maximum_heavy_pct": 10,
            "secondary_minimum_usable_pct": 70,
            "maximum_uncertain_pct": 5,
            "part_minimum_usable_pct": 70,
            "maximum_part_spread_pct_points": 20,
        },
    }

    (
        RESULT_ROOT / "weighted_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "CORPUS VIABILITY WEIGHTED RESULT",
        "=" * 72,
        "",
        f"Estimation sample rows: {len(rows):,}",
        (
            "Estimated inventory population: "
            f"{usable_result['population_sentences']:,} sentences"
        ),
        "",
        (
            "HEALTHY: "
            f"{healthy_result['percentage']:.2f}% "
            f"(95% CI "
            f"{healthy_result['ci95_lower_percentage']:.2f}%–"
            f"{healthy_result['ci95_upper_percentage']:.2f}%)"
        ),
        (
            "MINOR DAMAGE BUT USABLE: "
            f"{minor_result['percentage']:.2f}%"
        ),
        (
            "TOTAL USABLE: "
            f"{usable_result['percentage']:.2f}% "
            f"(95% CI "
            f"{usable_result['ci95_lower_percentage']:.2f}%–"
            f"{usable_result['ci95_upper_percentage']:.2f}%)"
        ),
        (
            "PROJECTED USABLE SENTENCES: "
            f"{usable_result['projected_sentence_count']:,}"
        ),
        (
            "TOTAL UNUSABLE: "
            f"{unusable_result['percentage']:.2f}%"
        ),
        (
            "HEAVY CORRUPTION: "
            f"{heavy_result['percentage']:.2f}%"
        ),
        (
            "UNCERTAIN: "
            f"{uncertain_result['percentage']:.2f}%"
        ),
        "",
        (
            "LOWEST PART USABLE RATE: "
            f"{minimum_part_usable:.2f}%"
        ),
        (
            "HIGHEST PART USABLE RATE: "
            f"{maximum_part_usable:.2f}%"
        ),
        (
            "PART USABLE-RATE SPREAD: "
            f"{usable_spread:.2f} percentage points"
        ),
        "",
        f"DECISION: {decision}",
        "",
        "REASONS:",
    ]

    report_lines.extend(
        f"- {reason}"
        for reason in decision_reasons
    )

    report_text = "\n".join(
        report_lines
    ) + "\n"

    (
        RESULT_ROOT / "decision_report.txt"
    ).write_text(
        report_text,
        encoding="utf-8",
    )

    print("\n" + report_text)
    print(
        f"weighted_summary: "
        f"{RESULT_ROOT / 'weighted_summary.json'}"
    )
    print(
        f"part_summary: "
        f"{RESULT_ROOT / 'part_health_summary.csv'}"
    )
    print(
        f"decision_report: "
        f"{RESULT_ROOT / 'decision_report.txt'}"
    )
    print(
        "\nWEIGHTED SUMMARY STATUS: PASS"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review and summarize the unbiased "
            "corpus-viability sample."
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices={
            "prepare",
            "show",
            "status",
            "merge",
            "summarize",
        },
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help=(
            "Batch number used by --mode show."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "prepare":
        prepare_batches()

    elif args.mode == "show":
        if args.batch < 1:
            raise ValueError(
                "--batch must be at least 1."
            )
        show_batch(args.batch)

    elif args.mode == "status":
        labeling_status()

    elif args.mode == "merge":
        merge_labels()

    elif args.mode == "summarize":
        summarize()

    else:
        raise RuntimeError(
            f"Unsupported mode: {args.mode}"
        )


if __name__ == "__main__":
    main()
