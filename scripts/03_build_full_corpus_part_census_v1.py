#!/usr/bin/env python3

"""
Build an exact census of the full curated-corpus part files.

The script reads every part-*.jsonl file to count records and inspect
basic source-ID consistency.

It does not modify the curated corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import hashlib
import json
import re


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    ROOT
    / "outputs"
    / "02_curated_corpus"
    / "normalize_full_v1"
    / "normalized_records"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "03_sentence_inventory"
    / "full_corpus_part_census_v1"
)

CSV_PATH = (
    OUTPUT_DIR
    / "full_corpus_part_census_v1.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "full_corpus_part_census_summary_v1.json"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "full_corpus_part_census_report_v1.md"
)

PART_PATTERN = re.compile(
    r"^part-(\d+)\.jsonl$"
)

PILOT_PARTS = {
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
        6,
    )


def parse_source_id(
    raw_line: bytes | None,
) -> tuple[str, str]:
    if not raw_line:
        return "", "EMPTY_FILE"

    try:
        record = json.loads(
            raw_line.decode("utf-8")
        )
    except Exception as error:
        return "", (
            "JSON_PARSE_ERROR:"
            + type(error).__name__
        )

    source_id = record.get(
        "source_record_id"
    )

    if source_id is None:
        return "", "SOURCE_ID_MISSING"

    return str(source_id), "OK"


def inspect_part_file(
    path: Path,
    part_number: int,
) -> dict[str, Any]:
    file_size = path.stat().st_size

    nonempty_rows = 0
    empty_lines = 0

    first_nonempty_line: bytes | None = None
    last_nonempty_line: bytes | None = None

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)

            stripped = raw_line.strip()

            if not stripped:
                empty_lines += 1
                continue

            nonempty_rows += 1

            if first_nonempty_line is None:
                first_nonempty_line = stripped

            last_nonempty_line = stripped

    first_source_id, first_status = (
        parse_source_id(
            first_nonempty_line
        )
    )

    last_source_id, last_status = (
        parse_source_id(
            last_nonempty_line
        )
    )

    source_range_matches_part = ""

    if (
        first_status == "OK"
        and last_status == "OK"
    ):
        first_part = (
            int(first_source_id)
            // 100_000
        )

        last_part = (
            int(last_source_id)
            // 100_000
        )

        source_range_matches_part = (
            first_part == part_number
            and last_part == part_number
        )

    return {
        "part_number": part_number,
        "filename": path.name,
        "file_size_bytes": file_size,
        "file_size_human": human_size(
            file_size
        ),
        "nonempty_record_rows": (
            nonempty_rows
        ),
        "empty_lines": empty_lines,
        "first_source_record_id": (
            first_source_id
        ),
        "first_record_status": (
            first_status
        ),
        "last_source_record_id": (
            last_source_id
        ),
        "last_record_status": (
            last_status
        ),
        "source_range_matches_part": (
            source_range_matches_part
        ),
        "included_in_segment_10k_v2": (
            part_number in PILOT_PARTS
        ),
        "pilot_records_used": (
            min(nonempty_rows, 1_000)
            if part_number in PILOT_PARTS
            else 0
        ),
        "sha256": digest.hexdigest(),
    }


def write_csv(
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = list(rows[0])

    with CSV_PATH.open(
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
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: "
            f"{INPUT_DIR}"
        )

    part_files = []

    for path in INPUT_DIR.glob(
        "part-*.jsonl"
    ):
        match = PART_PATTERN.fullmatch(
            path.name
        )

        if not match:
            continue

        part_files.append(
            (
                int(match.group(1)),
                path,
            )
        )

    part_files.sort(
        key=lambda item: item[0]
    )

    if not part_files:
        raise RuntimeError(
            "No part-*.jsonl files found."
        )

    part_numbers = [
        part_number
        for part_number, _ in part_files
    ]

    if (
        len(part_numbers)
        != len(set(part_numbers))
    ):
        raise RuntimeError(
            "Duplicate part numbers found."
        )

    print(
        "--- SCANNING FULL CURATED CORPUS ---"
    )
    print(
        "part_files_discovered:",
        len(part_files),
    )
    print(
        "minimum_part_number:",
        min(part_numbers),
    )
    print(
        "maximum_part_number:",
        max(part_numbers),
    )
    print()

    rows = []

    for index, (
        part_number,
        path,
    ) in enumerate(
        part_files,
        start=1,
    ):
        row = inspect_part_file(
            path,
            part_number,
        )

        rows.append(row)

        print(
            f"[{index}/{len(part_files)}] "
            f"part={part_number} "
            f"records="
            f"{row['nonempty_record_rows']} "
            f"size="
            f"{row['file_size_human']} "
            f"id_match="
            f"{row['source_range_matches_part']}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(rows)

    total_records = sum(
        int(row["nonempty_record_rows"])
        for row in rows
    )

    total_bytes = sum(
        int(row["file_size_bytes"])
        for row in rows
    )

    total_empty_lines = sum(
        int(row["empty_lines"])
        for row in rows
    )

    pilot_parts_present = sorted(
        part
        for part in PILOT_PARTS
        if part in set(part_numbers)
    )

    pilot_parts_missing = sorted(
        PILOT_PARTS
        - set(part_numbers)
    )

    pilot_population_records = sum(
        int(row["nonempty_record_rows"])
        for row in rows
        if row[
            "included_in_segment_10k_v2"
        ]
    )

    pilot_records_actually_used = sum(
        int(row["pilot_records_used"])
        for row in rows
    )

    invalid_source_ranges = [
        int(row["part_number"])
        for row in rows
        if row[
            "source_range_matches_part"
        ] is not True
    ]

    missing_numeric_parts = [
        part
        for part in range(
            min(part_numbers),
            max(part_numbers) + 1,
        )
        if part not in set(part_numbers)
    ]

    summary = {
        "status": "FULL_CORPUS_PART_CENSUS_COMPLETE",
        "input_directory": str(
            INPUT_DIR.relative_to(ROOT)
        ),
        "part_file_count": len(rows),
        "minimum_part_number": (
            min(part_numbers)
        ),
        "maximum_part_number": (
            max(part_numbers)
        ),
        "part_numbers": part_numbers,
        "missing_numeric_parts": (
            missing_numeric_parts
        ),
        "total_nonempty_records": (
            total_records
        ),
        "total_empty_lines": (
            total_empty_lines
        ),
        "total_size_bytes": total_bytes,
        "total_size_human": human_size(
            total_bytes
        ),
        "pilot_parts_requested": sorted(
            PILOT_PARTS
        ),
        "pilot_parts_present": (
            pilot_parts_present
        ),
        "pilot_parts_missing": (
            pilot_parts_missing
        ),
        "pilot_part_population_records": (
            pilot_population_records
        ),
        "pilot_records_actually_used": (
            pilot_records_actually_used
        ),
        "pilot_used_pct_of_full_records": (
            percentage(
                pilot_records_actually_used,
                total_records,
            )
        ),
        "pilot_part_population_pct_of_full_records": (
            percentage(
                pilot_population_records,
                total_records,
            )
        ),
        "parts_with_invalid_source_id_range": (
            invalid_source_ranges
        ),
        "output_csv": str(
            CSV_PATH.relative_to(ROOT)
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
        "# Full Curated Corpus Part Census v1",
        "",
        "## Status",
        "",
        "`FULL_CORPUS_PART_CENSUS_COMPLETE`",
        "",
        "## Full curated corpus",
        "",
        f"- Part files: {len(rows)}",
        f"- Part range: "
        f"{min(part_numbers)}–"
        f"{max(part_numbers)}",
        f"- Total non-empty records: "
        f"{total_records:,}",
        f"- Total size: "
        f"{human_size(total_bytes)}",
        f"- Empty lines: "
        f"{total_empty_lines:,}",
        "",
        "## Previous pilot coverage",
        "",
        f"- Pilot parts: "
        f"{sorted(PILOT_PARTS)}",
        f"- Pilot records actually used: "
        f"{pilot_records_actually_used:,}",
        f"- Pilot used percentage of full records: "
        f"{summary['pilot_used_pct_of_full_records']}%",
        f"- Full population inside pilot parts: "
        f"{pilot_population_records:,}",
        f"- Pilot-part population percentage of "
        f"full records: "
        f"{summary['pilot_part_population_pct_of_full_records']}%",
        "",
        "## Validation",
        "",
        f"- Missing requested pilot parts: "
        f"{pilot_parts_missing}",
        f"- Parts with invalid first/last "
        f"source-ID range: "
        f"{invalid_source_ranges}",
        f"- Missing numeric part filenames: "
        f"{missing_numeric_parts}",
        "",
        "This census does not evaluate sentence quality. "
        "It establishes the population from which the next "
        "representative sample must be drawn.",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print(
        "--- FULL CORPUS PART CENSUS BUILT ---"
    )
    print(
        "part_file_count:",
        summary["part_file_count"],
    )
    print(
        "total_nonempty_records:",
        total_records,
    )
    print(
        "total_size_human:",
        summary["total_size_human"],
    )
    print(
        "pilot_records_actually_used:",
        pilot_records_actually_used,
    )
    print(
        "pilot_used_pct_of_full_records:",
        summary[
            "pilot_used_pct_of_full_records"
        ],
    )
    print(
        "pilot_part_population_pct_of_full_records:",
        summary[
            "pilot_part_population_pct_of_full_records"
        ],
    )
    print(
        "parts_with_invalid_source_id_range:",
        invalid_source_ranges,
    )
    print()
    print("Created files:")
    print(
        "-",
        CSV_PATH.relative_to(ROOT),
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
        "FULL CORPUS PART CENSUS: PASS"
    )


if __name__ == "__main__":
    main()
