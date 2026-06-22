#!/usr/bin/env python3

import argparse
import csv
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ALLOWED_LABELS = [
    "HEALTHY",
    "MINOR_DAMAGE_BUT_USABLE",
    "FORMAT_OR_LIST",
    "FRAGMENT",
    "HEAVY_CORRUPTION",
    "MIXED_OR_FOREIGN_TEXT",
    "UNCERTAIN",
]

POTENTIALLY_USABLE_LABELS = {
    "HEALTHY",
    "MINOR_DAMAGE_BUT_USABLE",
}

EXPECTED_BATCH_COUNT = 28
EXPECTED_ROWS_PER_BATCH = 25
EXPECTED_TOTAL_ROWS = 700


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the completed corpus viability review batches and "
            "produce a reproducible summary report."
        )
    )

    parser.add_argument(
        "--batch-dir",
        default=(
            "outputs/03_sentence_inventory/"
            "corpus_viability_audit_v1/"
            "estimation_random_review_batches_v1"
        ),
        help="Directory containing batch-01.csv through batch-28.csv.",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/03_sentence_inventory/"
            "corpus_viability_audit_v1/"
            "estimation_review_summary_v1"
        ),
        help="Directory in which report files will be written.",
    )

    parser.add_argument(
        "--examples-per-label",
        type=int,
        default=5,
        help="Maximum representative examples exported for each label.",
    )

    return parser.parse_args()


def first_nonempty(
    row: Dict[str, str],
    candidate_columns: Iterable[str],
) -> str:
    for column in candidate_columns:
        value = str(row.get(column, "") or "").strip()
        if value:
            return value
    return ""


def split_flags(value: str) -> List[str]:
    value = str(value or "").strip()

    if not value or value == "NONE":
        return ["NONE"]

    return [
        flag.strip()
        for flag in value.split("|")
        if flag.strip()
    ] or ["NONE"]


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0

    proportion = successes / total
    denominator = 1 + (z * z / total)

    centre = (
        proportion
        + (z * z / (2 * total))
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )

    return centre - half_width, centre + half_width


def write_csv(
    path: Path,
    fieldnames: List[str],
    rows: Iterable[Dict[str, object]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def percentage(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total * 100


def make_group_rows(
    group_counts: Dict[str, Counter],
    group_column: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for group_name in sorted(group_counts):
        counts = group_counts[group_name]
        total = sum(counts.values())

        usable = sum(
            counts[label]
            for label in POTENTIALLY_USABLE_LABELS
        )

        row: Dict[str, object] = {
            group_column: group_name,
            "total": total,
            "potentially_usable": usable,
            "potentially_usable_pct": round(
                percentage(usable, total),
                2,
            ),
        }

        for label in ALLOWED_LABELS:
            row[label.lower()] = counts[label]

        rows.append(row)

    return rows


def main() -> None:
    args = parse_args()

    batch_dir = Path(args.batch_dir)
    output_dir = Path(args.output_dir)

    if not batch_dir.is_dir():
        raise FileNotFoundError(
            f"Batch directory does not exist: {batch_dir}"
        )

    batch_paths = sorted(
        batch_dir.glob("batch-[0-9][0-9].csv")
    )

    if len(batch_paths) != EXPECTED_BATCH_COUNT:
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_BATCH_COUNT} batch files, "
            f"but found {len(batch_paths)}."
        )

    required_columns = {
        "sample_id",
        "human_health_label",
        "human_health_notes",
    }

    total_rows = 0
    sample_ids: set[str] = set()
    duplicate_sample_ids: List[str] = []
    unlabeled_rows: List[Dict[str, object]] = []
    invalid_labels: List[Dict[str, object]] = []
    missing_columns: List[Dict[str, object]] = []
    wrong_row_counts: List[Dict[str, object]] = []

    label_counts: Counter = Counter()
    batch_label_counts: Dict[str, Counter] = defaultdict(Counter)
    part_label_counts: Dict[str, Counter] = defaultdict(Counter)
    boundary_label_counts: Dict[str, Counter] = defaultdict(Counter)
    parent_flag_label_counts: Dict[str, Counter] = defaultdict(Counter)

    examples: Dict[str, List[Dict[str, object]]] = {
        label: []
        for label in ALLOWED_LABELS
    }

    sentence_column_candidates = [
        "current_sentence",
        "sentence_text",
        "normalized_sentence",
        "sentence",
        "text",
        "raw_text",
    ]

    for batch_path in batch_paths:
        with batch_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)
            fieldnames = set(reader.fieldnames or [])

            missing = sorted(required_columns - fieldnames)

            if missing:
                missing_columns.append(
                    {
                        "batch": batch_path.name,
                        "missing_columns": missing,
                    }
                )
                continue

            batch_rows = list(reader)

        if len(batch_rows) != EXPECTED_ROWS_PER_BATCH:
            wrong_row_counts.append(
                {
                    "batch": batch_path.name,
                    "rows": len(batch_rows),
                }
            )

        for csv_row_number, row in enumerate(
            batch_rows,
            start=2,
        ):
            total_rows += 1

            sample_id = str(
                row.get("sample_id", "") or ""
            ).strip()

            label = str(
                row.get("human_health_label", "") or ""
            ).strip()

            notes = str(
                row.get("human_health_notes", "") or ""
            ).strip()

            if sample_id in sample_ids:
                duplicate_sample_ids.append(sample_id)
            else:
                sample_ids.add(sample_id)

            if not label:
                unlabeled_rows.append(
                    {
                        "batch": batch_path.name,
                        "csv_row_number": csv_row_number,
                        "sample_id": sample_id,
                    }
                )
                continue

            if label not in ALLOWED_LABELS:
                invalid_labels.append(
                    {
                        "batch": batch_path.name,
                        "csv_row_number": csv_row_number,
                        "sample_id": sample_id,
                        "label": label,
                    }
                )
                continue

            part = str(
                row.get("part_number", "")
                or row.get("part", "")
                or "UNKNOWN"
            ).strip() or "UNKNOWN"

            boundary_flags = split_flags(
                row.get("boundary_flags", "")
            )

            parent_flags = split_flags(
                row.get("parent_review_flags", "")
            )

            label_counts[label] += 1
            batch_label_counts[batch_path.stem][label] += 1
            part_label_counts[part][label] += 1

            for flag in boundary_flags:
                boundary_label_counts[flag][label] += 1

            for flag in parent_flags:
                parent_flag_label_counts[flag][label] += 1

            if len(examples[label]) < args.examples_per_label:
                sentence_text = first_nonempty(
                    row,
                    sentence_column_candidates,
                )

                examples[label].append(
                    {
                        "label": label,
                        "batch": batch_path.name,
                        "sample_id": sample_id,
                        "audit_row_number": str(
                            row.get("audit_row_number", "") or ""
                        ).strip(),
                        "part": part,
                        "boundary_flags": "|".join(boundary_flags),
                        "parent_review_flags": "|".join(parent_flags),
                        "sentence_text": sentence_text,
                        "human_health_notes": notes,
                    }
                )

    validation_errors: List[str] = []

    if missing_columns:
        validation_errors.append(
            f"{len(missing_columns)} files have missing columns"
        )

    if wrong_row_counts:
        validation_errors.append(
            f"{len(wrong_row_counts)} batches have incorrect row counts"
        )

    if total_rows != EXPECTED_TOTAL_ROWS:
        validation_errors.append(
            f"expected {EXPECTED_TOTAL_ROWS} rows, found {total_rows}"
        )

    if unlabeled_rows:
        validation_errors.append(
            f"{len(unlabeled_rows)} rows are unlabeled"
        )

    if invalid_labels:
        validation_errors.append(
            f"{len(invalid_labels)} rows have invalid labels"
        )

    if duplicate_sample_ids:
        validation_errors.append(
            f"{len(duplicate_sample_ids)} duplicate sample IDs found"
        )

    if validation_errors:
        print("\n--- VALIDATION FAILED ---")
        for error in validation_errors:
            print(f"- {error}")

        if missing_columns:
            print("\nMissing-column details:")
            for item in missing_columns:
                print(item)

        if wrong_row_counts:
            print("\nWrong-row-count details:")
            for item in wrong_row_counts:
                print(item)

        if unlabeled_rows:
            print("\nFirst unlabeled rows:")
            for item in unlabeled_rows[:20]:
                print(item)

        if invalid_labels:
            print("\nFirst invalid labels:")
            for item in invalid_labels[:20]:
                print(item)

        if duplicate_sample_ids:
            print("\nFirst duplicate sample IDs:")
            for sample_id in duplicate_sample_ids[:20]:
                print(sample_id)

        raise RuntimeError(
            "Input validation failed. No report files were written."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    usable_count = sum(
        label_counts[label]
        for label in POTENTIALLY_USABLE_LABELS
    )

    usable_lower, usable_upper = wilson_interval(
        usable_count,
        total_rows,
    )

    overall_rows: List[Dict[str, object]] = []

    for label in ALLOWED_LABELS:
        count = label_counts[label]

        if label == "HEALTHY":
            preliminary_action = "KEEP_AS_CLEAN_CANDIDATE"
        elif label == "MINOR_DAMAGE_BUT_USABLE":
            preliminary_action = "NORMALIZE_OR_REVIEW_BEFORE_USE"
        elif label == "UNCERTAIN":
            preliminary_action = "MANUAL_REVIEW"
        else:
            preliminary_action = "EXCLUDE_FROM_INITIAL_CLEAN_INVENTORY"

        overall_rows.append(
            {
                "label": label,
                "count": count,
                "percentage": round(
                    percentage(count, total_rows),
                    2,
                ),
                "preliminary_action": preliminary_action,
            }
        )

    write_csv(
        output_dir / "corpus_viability_label_counts_v1.csv",
        [
            "label",
            "count",
            "percentage",
            "preliminary_action",
        ],
        overall_rows,
    )

    batch_rows = make_group_rows(
        batch_label_counts,
        "batch",
    )

    part_rows = make_group_rows(
        part_label_counts,
        "part",
    )

    boundary_rows = make_group_rows(
        boundary_label_counts,
        "boundary_flag",
    )

    parent_flag_rows = make_group_rows(
        parent_flag_label_counts,
        "parent_review_flag",
    )

    group_fieldnames = [
        "total",
        "potentially_usable",
        "potentially_usable_pct",
    ] + [
        label.lower()
        for label in ALLOWED_LABELS
    ]

    write_csv(
        output_dir / "corpus_viability_by_batch_v1.csv",
        ["batch"] + group_fieldnames,
        batch_rows,
    )

    write_csv(
        output_dir / "corpus_viability_by_part_v1.csv",
        ["part"] + group_fieldnames,
        part_rows,
    )

    write_csv(
        output_dir / "corpus_viability_by_boundary_flag_v1.csv",
        ["boundary_flag"] + group_fieldnames,
        boundary_rows,
    )

    write_csv(
        output_dir / "corpus_viability_by_parent_flag_v1.csv",
        ["parent_review_flag"] + group_fieldnames,
        parent_flag_rows,
    )

    flattened_examples: List[Dict[str, object]] = []

    for label in ALLOWED_LABELS:
        flattened_examples.extend(examples[label])

    write_csv(
        output_dir / "corpus_viability_examples_v1.csv",
        [
            "label",
            "batch",
            "sample_id",
            "audit_row_number",
            "part",
            "boundary_flags",
            "parent_review_flags",
            "sentence_text",
            "human_health_notes",
        ],
        flattened_examples,
    )

    summary = {
        "audit_version": "corpus_viability_audit_v1",
        "review_summary_version": "estimation_review_summary_v1",
        "validation": {
            "batch_files": len(batch_paths),
            "total_rows": total_rows,
            "expected_total_rows": EXPECTED_TOTAL_ROWS,
            "unlabeled_rows": 0,
            "invalid_labels": 0,
            "duplicate_sample_ids": 0,
            "files_missing_required_columns": 0,
            "batches_with_wrong_row_count": 0,
            "status": "PASS",
        },
        "overall": {
            "label_counts": {
                label: label_counts[label]
                for label in ALLOWED_LABELS
            },
            "healthy_count": label_counts["HEALTHY"],
            "healthy_percentage": round(
                percentage(
                    label_counts["HEALTHY"],
                    total_rows,
                ),
                2,
            ),
            "minor_damage_but_usable_count": (
                label_counts["MINOR_DAMAGE_BUT_USABLE"]
            ),
            "minor_damage_but_usable_percentage": round(
                percentage(
                    label_counts["MINOR_DAMAGE_BUT_USABLE"],
                    total_rows,
                ),
                2,
            ),
            "potentially_usable_count": usable_count,
            "potentially_usable_percentage": round(
                percentage(
                    usable_count,
                    total_rows,
                ),
                2,
            ),
            "potentially_usable_wilson_95_ci_percentage": {
                "lower": round(usable_lower * 100, 2),
                "upper": round(usable_upper * 100, 2),
            },
        },
        "acceptance_policy": {
            "keep_as_clean_candidate": [
                "HEALTHY",
            ],
            "normalize_or_review_before_use": [
                "MINOR_DAMAGE_BUT_USABLE",
            ],
            "manual_review": [
                "UNCERTAIN",
            ],
            "exclude_from_initial_clean_inventory": [
                "FORMAT_OR_LIST",
                "FRAGMENT",
                "HEAVY_CORRUPTION",
                "MIXED_OR_FOREIGN_TEXT",
            ],
        },
        "decision": {
            "status": "GO_WITH_CURATION",
            "plain_conclusion": (
                "The corpus is viable for continued sentence curation, "
                "but it is not safe to use as clean training data without "
                "filtering, normalization, and provenance preservation."
            ),
        },
    }

    with (
        output_dir / "corpus_viability_summary_v1.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    markdown_lines: List[str] = [
        "# Corpus Viability Estimation Review — Version 1",
        "",
        "## Validation",
        "",
        f"- Batch files: {len(batch_paths)}",
        f"- Reviewed rows: {total_rows}",
        "- Unlabeled rows: 0",
        "- Invalid labels: 0",
        "- Duplicate sample IDs: 0",
        "- Missing required columns: 0",
        "- Batches with incorrect row counts: 0",
        "- Validation status: **PASS**",
        "",
        "## Overall label distribution",
        "",
        "| Label | Rows | Percentage | Initial action |",
        "|---|---:|---:|---|",
    ]

    for item in overall_rows:
        markdown_lines.append(
            f"| {item['label']} "
            f"| {item['count']} "
            f"| {item['percentage']:.2f}% "
            f"| {item['preliminary_action']} |"
        )

    markdown_lines.extend(
        [
            "",
            "## Main viability result",
            "",
            (
                f"- Healthy: **{label_counts['HEALTHY']} "
                f"({percentage(label_counts['HEALTHY'], total_rows):.2f}%)**"
            ),
            (
                "- Minor damage but usable: "
                f"**{label_counts['MINOR_DAMAGE_BUT_USABLE']} "
                f"({percentage(label_counts['MINOR_DAMAGE_BUT_USABLE'], total_rows):.2f}%)**"
            ),
            (
                f"- Potentially usable total: **{usable_count} "
                f"({percentage(usable_count, total_rows):.2f}%)**"
            ),
            (
                "- Approximate 95% confidence interval for the "
                f"potentially usable share: **{usable_lower * 100:.2f}% "
                f"to {usable_upper * 100:.2f}%**"
            ),
            "",
            "## Acceptance policy",
            "",
            "### Keep as clean candidates",
            "",
            "- `HEALTHY`",
            "",
            "### Keep only after safe normalization or additional review",
            "",
            "- `MINOR_DAMAGE_BUT_USABLE`",
            "",
            "### Send to manual review",
            "",
            "- `UNCERTAIN`",
            "",
            "### Exclude from the initial clean-sentence inventory",
            "",
            "- `FORMAT_OR_LIST`",
            "- `FRAGMENT`",
            "- `HEAVY_CORRUPTION`",
            "- `MIXED_OR_FOREIGN_TEXT`",
            "",
            "## Decision",
            "",
            "**GO — WITH CURATION REQUIRED**",
            "",
            (
                "The corpus contains enough useful Uzbek sentence material "
                "to continue the project. It must not be used directly as "
                "clean training data. Sentence filtering, controlled "
                "normalization, source traceability, and conservative "
                "acceptance rules remain necessary."
            ),
            "",
            "## Important interpretation",
            "",
            (
                "The percentages are estimates from a manually reviewed "
                "random sample of 700 extracted segments. They describe "
                "estimated corpus viability and are not yet exact counts "
                "for the complete corpus."
            ),
            "",
            "## Representative examples",
            "",
        ]
    )

    for label in ALLOWED_LABELS:
        markdown_lines.append(f"### {label}")
        markdown_lines.append("")

        if not examples[label]:
            markdown_lines.append("No examples exported.")
            markdown_lines.append("")
            continue

        for example in examples[label]:
            sentence = str(
                example.get("sentence_text", "") or ""
            ).strip()

            sentence_preview = textwrap.shorten(
                sentence,
                width=240,
                placeholder=" ...",
            ) if sentence else "[Sentence column not found]"

            notes = str(
                example.get("human_health_notes", "") or ""
            ).strip()

            markdown_lines.append(
                f"- `{example['sample_id']}` — {sentence_preview}"
            )

            if notes:
                markdown_lines.append(
                    f"  - Review note: {notes}"
                )

        markdown_lines.append("")

    report_path = (
        output_dir / "corpus_viability_report_v1.md"
    )

    report_path.write_text(
        "\n".join(markdown_lines).rstrip() + "\n",
        encoding="utf-8",
    )

    print("\n--- CORPUS VIABILITY REPORT BUILT ---")
    print(f"input_batch_dir: {batch_dir}")
    print(f"output_dir: {output_dir}")
    print(f"batch_files: {len(batch_paths)}")
    print(f"total_rows: {total_rows}")
    print(f"label_counts: {dict(label_counts)}")
    print(
        "potentially_usable: "
        f"{usable_count}/{total_rows} "
        f"({percentage(usable_count, total_rows):.2f}%)"
    )
    print(
        "potentially_usable_95pct_ci: "
        f"{usable_lower * 100:.2f}%–{usable_upper * 100:.2f}%"
    )
    print("decision: GO_WITH_CURATION")
    print("\nCreated files:")

    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"- {path}")

    print("\nCORPUS VIABILITY REPORT: PASS")


if __name__ == "__main__":
    main()
