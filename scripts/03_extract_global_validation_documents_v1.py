#!/usr/bin/env python3

"""
Extract the documents selected by the global validation manifest.

The script streams the relevant curated-corpus part files and copies
only the requested records into a versioned derived JSONL file.

It does not modify the source corpus or the sampling manifest.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import csv
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    ROOT
    / "outputs"
    / "02_curated_corpus"
    / "normalize_full_v1"
    / "normalized_records"
)

SAMPLE_ROOT = (
    ROOT
    / "outputs"
    / "03_sentence_inventory"
    / "global_validation_document_sample_v1"
)

MANIFEST_PATH = (
    SAMPLE_ROOT
    / "global_validation_document_manifest_v1.csv"
)

EXTRACTED_PATH = (
    SAMPLE_ROOT
    / "extracted_documents_v1.jsonl"
)

RESULTS_PATH = (
    SAMPLE_ROOT
    / "extraction_results_v1.csv"
)

INVALID_PATH = (
    SAMPLE_ROOT
    / "invalid_extractions_v1.csv"
)

SUMMARY_PATH = (
    SAMPLE_ROOT
    / "extraction_summary_v1.json"
)

REPORT_PATH = (
    SAMPLE_ROOT
    / "extraction_report_v1.md"
)


EXPECTED_DOCUMENTS = 6_950
EXPECTED_PARTS = 139


RESULT_FIELDNAMES = [
    "global_sample_number",
    "document_sample_id",
    "part_number",
    "filename",
    "local_row_number",
    "expected_source_record_id",
    "observed_source_record_id",
    "source_id_matches",
    "observed_source_part",
    "source_part_matches",
    "json_parse_success",
    "normalized_text_present",
    "normalized_text_nonempty",
    "normalized_text_character_count",
    "normalized_text_token_count",
    "normalized_text_sha256",
    "source_line_sha256",
    "original_record_field_count",
    "raw_text_field_present",
    "extraction_status",
    "failure_reasons",
]


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


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


def read_manifest() -> list[dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        raw_rows = list(csv.DictReader(handle))

    if len(raw_rows) != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"Expected {EXPECTED_DOCUMENTS} manifest rows, "
            f"found {len(raw_rows)}."
        )

    required_columns = {
        "global_sample_number",
        "document_sample_id",
        "part_number",
        "filename",
        "local_row_number",
        "source_record_id",
        "part_record_count",
        "part_eligible_record_count",
        "part_allocated_documents",
        "inclusion_probability",
        "sampling_weight",
        "old_pilot_part",
        "old_pilot_document_excluded",
        "random_seed",
    }

    missing = required_columns - set(
        raw_rows[0]
    )

    if missing:
        raise ValueError(
            "Manifest is missing columns: "
            f"{sorted(missing)}"
        )

    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        rows.append(
            {
                "global_sample_number": int(
                    raw["global_sample_number"]
                ),
                "document_sample_id": (
                    raw["document_sample_id"].strip()
                ),
                "part_number": int(
                    raw["part_number"]
                ),
                "filename": (
                    raw["filename"].strip()
                ),
                "local_row_number": int(
                    raw["local_row_number"]
                ),
                "source_record_id": int(
                    raw["source_record_id"]
                ),
                "part_record_count": int(
                    raw["part_record_count"]
                ),
                "part_eligible_record_count": int(
                    raw[
                        "part_eligible_record_count"
                    ]
                ),
                "part_allocated_documents": int(
                    raw[
                        "part_allocated_documents"
                    ]
                ),
                "inclusion_probability": float(
                    raw["inclusion_probability"]
                ),
                "sampling_weight": float(
                    raw["sampling_weight"]
                ),
                "old_pilot_part": truth(
                    raw["old_pilot_part"]
                ),
                "old_pilot_document_excluded": (
                    truth(
                        raw[
                            "old_pilot_document_excluded"
                        ]
                    )
                ),
                "random_seed": int(
                    raw["random_seed"]
                ),
            }
        )

    global_numbers = [
        row["global_sample_number"]
        for row in rows
    ]

    if sorted(global_numbers) != list(
        range(1, EXPECTED_DOCUMENTS + 1)
    ):
        raise ValueError(
            "Manifest global sample numbers "
            "are not exactly 1–6950."
        )

    sample_ids = [
        row["document_sample_id"]
        for row in rows
    ]

    source_ids = [
        row["source_record_id"]
        for row in rows
    ]

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            "Duplicate document sample IDs "
            "found in manifest."
        )

    if len(source_ids) != len(set(source_ids)):
        raise ValueError(
            "Duplicate source record IDs "
            "found in manifest."
        )

    observed_parts = {
        row["part_number"]
        for row in rows
    }

    if observed_parts != set(
        range(EXPECTED_PARTS)
    ):
        missing_parts = sorted(
            set(range(EXPECTED_PARTS))
            - observed_parts
        )

        raise ValueError(
            f"Manifest is missing parts: "
            f"{missing_parts}"
        )

    pilot_overlap = [
        row
        for row in rows
        if row[
            "old_pilot_document_excluded"
        ]
    ]

    if pilot_overlap:
        raise ValueError(
            "Manifest contains old pilot overlap."
        )

    source_formula_errors = [
        row
        for row in rows
        if row["source_record_id"]
        != (
            row["part_number"] * 100_000
            + row["local_row_number"]
        )
    ]

    if source_formula_errors:
        raise ValueError(
            "Manifest contains source-ID formula "
            "mismatches."
        )

    return rows


def build_failure_result(
    manifest_row: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "global_sample_number": (
            manifest_row["global_sample_number"]
        ),
        "document_sample_id": (
            manifest_row["document_sample_id"]
        ),
        "part_number": (
            manifest_row["part_number"]
        ),
        "filename": (
            manifest_row["filename"]
        ),
        "local_row_number": (
            manifest_row["local_row_number"]
        ),
        "expected_source_record_id": (
            manifest_row["source_record_id"]
        ),
        "observed_source_record_id": "",
        "source_id_matches": False,
        "observed_source_part": "",
        "source_part_matches": False,
        "json_parse_success": False,
        "normalized_text_present": False,
        "normalized_text_nonempty": False,
        "normalized_text_character_count": 0,
        "normalized_text_token_count": 0,
        "normalized_text_sha256": "",
        "source_line_sha256": "",
        "original_record_field_count": 0,
        "raw_text_field_present": False,
        "extraction_status": "FAIL",
        "failure_reasons": reason,
    }


def inspect_selected_line(
    raw_line: bytes,
    manifest_row: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any] | None,
]:
    stripped = raw_line.strip()
    failure_reasons: list[str] = []

    source_line_hash = (
        hashlib.sha256(stripped).hexdigest()
        if stripped
        else ""
    )

    record: dict[str, Any] | None = None
    json_parse_success = False

    if not stripped:
        failure_reasons.append(
            "EMPTY_SOURCE_LINE"
        )
    else:
        try:
            parsed = json.loads(
                stripped.decode("utf-8")
            )

            if not isinstance(parsed, dict):
                failure_reasons.append(
                    "JSON_RECORD_NOT_OBJECT"
                )
            else:
                record = parsed
                json_parse_success = True

        except UnicodeDecodeError:
            failure_reasons.append(
                "UTF8_DECODE_ERROR"
            )

        except json.JSONDecodeError:
            failure_reasons.append(
                "JSON_PARSE_ERROR"
            )

    observed_source_id: int | None = None
    observed_source_part: int | None = None

    if record is not None:
        raw_source_id = record.get(
            "source_record_id"
        )

        if raw_source_id is None:
            failure_reasons.append(
                "SOURCE_RECORD_ID_MISSING"
            )
        else:
            try:
                observed_source_id = int(
                    raw_source_id
                )

                observed_source_part = (
                    observed_source_id
                    // 100_000
                )

            except (
                TypeError,
                ValueError,
            ):
                failure_reasons.append(
                    "SOURCE_RECORD_ID_INVALID"
                )

    expected_source_id = (
        manifest_row["source_record_id"]
    )

    source_id_matches = (
        observed_source_id
        == expected_source_id
    )

    source_part_matches = (
        observed_source_part
        == manifest_row["part_number"]
    )

    if (
        observed_source_id is not None
        and not source_id_matches
    ):
        failure_reasons.append(
            "SOURCE_RECORD_ID_MISMATCH"
        )

    if (
        observed_source_part is not None
        and not source_part_matches
    ):
        failure_reasons.append(
            "SOURCE_PART_MISMATCH"
        )

    normalized_text_present = False
    normalized_text_nonempty = False
    normalized_text = ""
    normalized_text_hash = ""

    if record is not None:
        normalized_value = record.get(
            "normalized_text"
        )

        normalized_text_present = (
            isinstance(normalized_value, str)
        )

        if not normalized_text_present:
            failure_reasons.append(
                "NORMALIZED_TEXT_MISSING_OR_NOT_STRING"
            )
        else:
            normalized_text = normalized_value

            normalized_text_nonempty = bool(
                normalized_text.strip()
            )

            if not normalized_text_nonempty:
                failure_reasons.append(
                    "NORMALIZED_TEXT_EMPTY"
                )
            else:
                normalized_text_hash = (
                    hashlib.sha256(
                        normalized_text.encode(
                            "utf-8"
                        )
                    ).hexdigest()
                )

    extraction_status = (
        "PASS"
        if not failure_reasons
        else "FAIL"
    )

    result = {
        "global_sample_number": (
            manifest_row["global_sample_number"]
        ),
        "document_sample_id": (
            manifest_row["document_sample_id"]
        ),
        "part_number": (
            manifest_row["part_number"]
        ),
        "filename": (
            manifest_row["filename"]
        ),
        "local_row_number": (
            manifest_row["local_row_number"]
        ),
        "expected_source_record_id": (
            expected_source_id
        ),
        "observed_source_record_id": (
            observed_source_id
            if observed_source_id is not None
            else ""
        ),
        "source_id_matches": (
            source_id_matches
        ),
        "observed_source_part": (
            observed_source_part
            if observed_source_part is not None
            else ""
        ),
        "source_part_matches": (
            source_part_matches
        ),
        "json_parse_success": (
            json_parse_success
        ),
        "normalized_text_present": (
            normalized_text_present
        ),
        "normalized_text_nonempty": (
            normalized_text_nonempty
        ),
        "normalized_text_character_count": (
            len(normalized_text)
        ),
        "normalized_text_token_count": (
            len(normalized_text.split())
        ),
        "normalized_text_sha256": (
            normalized_text_hash
        ),
        "source_line_sha256": (
            source_line_hash
        ),
        "original_record_field_count": (
            len(record)
            if record is not None
            else 0
        ),
        "raw_text_field_present": (
            record is not None
            and isinstance(
                record.get("raw_text"),
                str,
            )
        ),
        "extraction_status": (
            extraction_status
        ),
        "failure_reasons": "|".join(
            failure_reasons
        ),
    }

    if extraction_status != "PASS":
        return result, None

    extracted_wrapper = {
        "global_sample_number": (
            manifest_row["global_sample_number"]
        ),
        "document_sample_id": (
            manifest_row["document_sample_id"]
        ),
        "sample_metadata": {
            "part_number": (
                manifest_row["part_number"]
            ),
            "filename": (
                manifest_row["filename"]
            ),
            "local_row_number": (
                manifest_row[
                    "local_row_number"
                ]
            ),
            "source_record_id": (
                expected_source_id
            ),
            "part_record_count": (
                manifest_row[
                    "part_record_count"
                ]
            ),
            "part_eligible_record_count": (
                manifest_row[
                    "part_eligible_record_count"
                ]
            ),
            "part_allocated_documents": (
                manifest_row[
                    "part_allocated_documents"
                ]
            ),
            "inclusion_probability": (
                manifest_row[
                    "inclusion_probability"
                ]
            ),
            "sampling_weight": (
                manifest_row[
                    "sampling_weight"
                ]
            ),
            "random_seed": (
                manifest_row["random_seed"]
            ),
            "old_pilot_part": (
                manifest_row["old_pilot_part"]
            ),
            "old_pilot_overlap": False,
        },
        "source_line_sha256": (
            source_line_hash
        ),
        "normalized_text_sha256": (
            normalized_text_hash
        ),
        "source_record": record,
    }

    return result, extracted_wrapper


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULT_FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    manifest_rows = read_manifest()

    rows_by_part: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in manifest_rows:
        rows_by_part[
            row["part_number"]
        ].append(row)

    results: list[dict[str, Any]] = []
    extracted_wrappers: list[
        dict[str, Any]
    ] = []

    lines_scanned_by_part: dict[
        int,
        int,
    ] = {}

    print(
        "--- EXTRACTING GLOBAL VALIDATION DOCUMENTS ---"
    )
    print(
        "manifest_documents:",
        len(manifest_rows),
    )
    print(
        "manifest_parts:",
        len(rows_by_part),
    )
    print()

    for progress_number, part_number in enumerate(
        sorted(rows_by_part),
        start=1,
    ):
        requested_rows = rows_by_part[
            part_number
        ]

        filenames = {
            row["filename"]
            for row in requested_rows
        }

        if len(filenames) != 1:
            raise ValueError(
                f"Part {part_number} has multiple "
                f"manifest filenames: {filenames}"
            )

        filename = next(iter(filenames))
        input_path = INPUT_DIR / filename

        if not input_path.exists():
            for manifest_row in requested_rows:
                results.append(
                    build_failure_result(
                        manifest_row,
                        "SOURCE_PART_FILE_MISSING",
                    )
                )

            print(
                f"[{progress_number}/{EXPECTED_PARTS}] "
                f"part={part_number} "
                f"requested={len(requested_rows)} "
                f"status=SOURCE_FILE_MISSING"
            )
            continue

        requested_by_local_row = {
            row["local_row_number"]: row
            for row in requested_rows
        }

        if (
            len(requested_by_local_row)
            != len(requested_rows)
        ):
            raise ValueError(
                f"Part {part_number} has duplicate "
                "local-row requests."
            )

        selected_local_rows = set(
            requested_by_local_row
        )

        maximum_selected_row = max(
            selected_local_rows
        )

        found_local_rows: set[int] = set()
        lines_scanned = 0

        with input_path.open("rb") as handle:
            for local_row_number, raw_line in enumerate(
                handle
            ):
                lines_scanned = (
                    local_row_number + 1
                )

                if (
                    local_row_number
                    in selected_local_rows
                ):
                    manifest_row = (
                        requested_by_local_row[
                            local_row_number
                        ]
                    )

                    result, wrapper = (
                        inspect_selected_line(
                            raw_line,
                            manifest_row,
                        )
                    )

                    results.append(result)

                    if wrapper is not None:
                        extracted_wrappers.append(
                            wrapper
                        )

                    found_local_rows.add(
                        local_row_number
                    )

                if (
                    local_row_number
                    >= maximum_selected_row
                ):
                    break

        lines_scanned_by_part[
            part_number
        ] = lines_scanned

        missing_local_rows = sorted(
            selected_local_rows
            - found_local_rows
        )

        for missing_local_row in missing_local_rows:
            manifest_row = (
                requested_by_local_row[
                    missing_local_row
                ]
            )

            results.append(
                build_failure_result(
                    manifest_row,
                    "REQUESTED_LOCAL_ROW_NOT_FOUND",
                )
            )

        part_passes = sum(
            result["extraction_status"]
            == "PASS"
            for result in results
            if result["part_number"]
            == part_number
        )

        part_failures = (
            len(requested_rows)
            - part_passes
        )

        print(
            f"[{progress_number}/{EXPECTED_PARTS}] "
            f"part={part_number:<3} "
            f"requested={len(requested_rows):<2} "
            f"pass={part_passes:<2} "
            f"fail={part_failures:<2} "
            f"lines_scanned={lines_scanned}"
        )

    results.sort(
        key=lambda row: int(
            row["global_sample_number"]
        )
    )

    extracted_wrappers.sort(
        key=lambda row: int(
            row["global_sample_number"]
        )
    )

    SAMPLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        RESULTS_PATH,
        results,
    )

    invalid_results = [
        row
        for row in results
        if row["extraction_status"]
        != "PASS"
    ]

    write_csv(
        INVALID_PATH,
        invalid_results,
    )

    with EXTRACTED_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for wrapper in extracted_wrappers:
            handle.write(
                json.dumps(
                    wrapper,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")

    result_sample_ids = [
        row["document_sample_id"]
        for row in results
    ]

    extracted_sample_ids = [
        row["document_sample_id"]
        for row in extracted_wrappers
    ]

    observed_source_ids = [
        int(row["observed_source_record_id"])
        for row in results
        if row["extraction_status"]
        == "PASS"
    ]

    result_part_counts = Counter(
        int(row["part_number"])
        for row in results
    )

    pass_part_counts = Counter(
        int(row["sample_metadata"][
            "part_number"
        ])
        for row in extracted_wrappers
    )

    text_hash_counts = Counter(
        row["normalized_text_sha256"]
        for row in results
        if (
            row["extraction_status"]
            == "PASS"
            and row[
                "normalized_text_sha256"
            ]
        )
    )

    duplicate_text_groups = sum(
        count > 1
        for count in text_hash_counts.values()
    )

    duplicate_text_rows = sum(
        count
        for count in text_hash_counts.values()
        if count > 1
    )

    failure_reason_counts = Counter()

    for row in invalid_results:
        for reason in str(
            row["failure_reasons"]
        ).split("|"):
            if reason:
                failure_reason_counts[
                    reason
                ] += 1

    validation_errors: list[str] = []

    if len(results) != EXPECTED_DOCUMENTS:
        validation_errors.append(
            "RESULT_ROW_COUNT_MISMATCH"
        )

    if (
        len(result_sample_ids)
        != len(set(result_sample_ids))
    ):
        validation_errors.append(
            "DUPLICATE_RESULT_SAMPLE_IDS"
        )

    if (
        len(extracted_sample_ids)
        != len(set(extracted_sample_ids))
    ):
        validation_errors.append(
            "DUPLICATE_EXTRACTED_SAMPLE_IDS"
        )

    if (
        len(observed_source_ids)
        != len(set(observed_source_ids))
    ):
        validation_errors.append(
            "DUPLICATE_OBSERVED_SOURCE_IDS"
        )

    if invalid_results:
        validation_errors.append(
            "INVALID_EXTRACTIONS_PRESENT"
        )

    if (
        len(extracted_wrappers)
        != EXPECTED_DOCUMENTS
    ):
        validation_errors.append(
            "EXTRACTED_DOCUMENT_COUNT_MISMATCH"
        )

    if len(result_part_counts) != EXPECTED_PARTS:
        validation_errors.append(
            "RESULT_PART_COVERAGE_MISMATCH"
        )

    if len(pass_part_counts) != EXPECTED_PARTS:
        validation_errors.append(
            "PASS_PART_COVERAGE_MISMATCH"
        )

    output_sha256 = sha256_file(
        EXTRACTED_PATH
    )

    output_size = EXTRACTED_PATH.stat().st_size

    summary = {
        "status": (
            "PASS"
            if not validation_errors
            else "FAIL"
        ),
        "manifest_path": str(
            MANIFEST_PATH.relative_to(ROOT)
        ),
        "source_directory": str(
            INPUT_DIR.relative_to(ROOT)
        ),
        "requested_documents": (
            EXPECTED_DOCUMENTS
        ),
        "result_rows": len(results),
        "extracted_documents": (
            len(extracted_wrappers)
        ),
        "invalid_extractions": (
            len(invalid_results)
        ),
        "parts_requested": (
            len(rows_by_part)
        ),
        "parts_with_passed_documents": (
            len(pass_part_counts)
        ),
        "duplicate_result_sample_ids": (
            len(result_sample_ids)
            - len(set(result_sample_ids))
        ),
        "duplicate_extracted_sample_ids": (
            len(extracted_sample_ids)
            - len(set(extracted_sample_ids))
        ),
        "duplicate_observed_source_ids": (
            len(observed_source_ids)
            - len(set(observed_source_ids))
        ),
        "source_id_mismatches": sum(
            not truth(
                row["source_id_matches"]
            )
            for row in results
        ),
        "source_part_mismatches": sum(
            not truth(
                row["source_part_matches"]
            )
            for row in results
        ),
        "json_parse_failures": sum(
            not truth(
                row["json_parse_success"]
            )
            for row in results
        ),
        "missing_or_empty_normalized_text": sum(
            not truth(
                row[
                    "normalized_text_nonempty"
                ]
            )
            for row in results
        ),
        "failure_reason_counts": dict(
            sorted(
                failure_reason_counts.items()
            )
        ),
        "duplicate_normalized_text_groups": (
            duplicate_text_groups
        ),
        "duplicate_normalized_text_rows": (
            duplicate_text_rows
        ),
        "minimum_lines_scanned_in_part": min(
            lines_scanned_by_part.values(),
            default=0,
        ),
        "maximum_lines_scanned_in_part": max(
            lines_scanned_by_part.values(),
            default=0,
        ),
        "output_jsonl": str(
            EXTRACTED_PATH.relative_to(ROOT)
        ),
        "output_jsonl_size_bytes": (
            output_size
        ),
        "output_jsonl_size_human": (
            human_size(output_size)
        ),
        "output_jsonl_sha256": (
            output_sha256
        ),
        "validation_errors": (
            validation_errors
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
        "# Global Validation Document Extraction v1",
        "",
        "## Status",
        "",
        f"`{summary['status']}`",
        "",
        "## Extraction validation",
        "",
        f"- Requested documents: "
        f"{EXPECTED_DOCUMENTS:,}",
        f"- Extracted documents: "
        f"{len(extracted_wrappers):,}",
        f"- Invalid extractions: "
        f"{len(invalid_results):,}",
        f"- Parts represented: "
        f"{len(pass_part_counts)} / "
        f"{EXPECTED_PARTS}",
        f"- Source-ID mismatches: "
        f"{summary['source_id_mismatches']}",
        f"- Source-part mismatches: "
        f"{summary['source_part_mismatches']}",
        f"- JSON parse failures: "
        f"{summary['json_parse_failures']}",
        f"- Missing or empty normalized text: "
        f"{summary['missing_or_empty_normalized_text']}",
        f"- Duplicate observed source IDs: "
        f"{summary['duplicate_observed_source_ids']}",
        "",
        "## Preliminary duplicate-text observation",
        "",
        f"- Duplicate normalized-text groups: "
        f"{duplicate_text_groups}",
        f"- Rows belonging to duplicate-text groups: "
        f"{duplicate_text_rows}",
        "",
        "Duplicate text is reported for later analysis. "
        "It is not treated as an extraction failure.",
        "",
        "## Reproducibility",
        "",
        f"- Extracted JSONL SHA-256: "
        f"`{output_sha256}`",
        f"- Extracted JSONL size: "
        f"{human_size(output_size)}",
        "",
        "This step validates extraction and lineage only. "
        "It does not evaluate linguistic or sentence quality.",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print(
        "--- GLOBAL VALIDATION DOCUMENT EXTRACTION BUILT ---"
    )
    print(
        "requested_documents:",
        EXPECTED_DOCUMENTS,
    )
    print(
        "result_rows:",
        len(results),
    )
    print(
        "extracted_documents:",
        len(extracted_wrappers),
    )
    print(
        "invalid_extractions:",
        len(invalid_results),
    )
    print(
        "parts_with_passed_documents:",
        len(pass_part_counts),
    )
    print(
        "source_id_mismatches:",
        summary["source_id_mismatches"],
    )
    print(
        "source_part_mismatches:",
        summary["source_part_mismatches"],
    )
    print(
        "json_parse_failures:",
        summary["json_parse_failures"],
    )
    print(
        "missing_or_empty_normalized_text:",
        summary[
            "missing_or_empty_normalized_text"
        ],
    )
    print(
        "duplicate_observed_source_ids:",
        summary[
            "duplicate_observed_source_ids"
        ],
    )
    print(
        "duplicate_normalized_text_groups:",
        duplicate_text_groups,
    )
    print(
        "duplicate_normalized_text_rows:",
        duplicate_text_rows,
    )
    print(
        "output_jsonl_size:",
        human_size(output_size),
    )
    print(
        "output_jsonl_sha256:",
        output_sha256,
    )
    print()
    print("Created files:")
    print(
        "-",
        EXTRACTED_PATH.relative_to(ROOT),
    )
    print(
        "-",
        RESULTS_PATH.relative_to(ROOT),
    )
    print(
        "-",
        INVALID_PATH.relative_to(ROOT),
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

    if validation_errors:
        print(
            "GLOBAL VALIDATION DOCUMENT EXTRACTION: FAIL"
        )
        print(
            "validation_errors:",
            validation_errors,
        )

        raise SystemExit(1)

    print(
        "GLOBAL VALIDATION DOCUMENT EXTRACTION: PASS"
    )
    print(
        "STATUS: READY_FOR_SEGMENTATION"
    )


if __name__ == "__main__":
    main()
