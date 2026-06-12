#!/usr/bin/env python3
"""
Independently verify a Phase 2 safe-normalization output directory.

The verifier does not reuse the producer's normalization functions. It streams the
raw top-level JSON array and the derived JSONL shards side by side and checks:

- sequential source_record_id coverage
- one derived row per raw record
- raw and normalized SHA-256 values
- raw/normalized character counts
- operation_count consistency
- exact semantics of TRIM_ONE_TRAILING_ASCII_SPACE
- status/review Boolean consistency
- no unapproved operation names
- idempotence of the approved trim rule
- completion marker and manifest accounting
- raw-corpus immutability
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

try:
    from ijson.backends import python as ijson_backend
except ImportError as exc:
    raise SystemExit(
        "ERROR: ijson is required. Install project requirements first."
    ) from exc


APPROVED_OPERATION = "TRIM_ONE_TRAILING_ASCII_SPACE"
APPROVED_OPERATIONS = {APPROVED_OPERATION}


@dataclass(frozen=True)
class FileFingerprint:
    size_bytes: int
    mtime_ns: int
    first_chunk_sha256: str
    last_chunk_sha256: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_file(path: Path, chunk_size: int = 1024 * 1024) -> FileFingerprint:
    stat = path.stat()
    with path.open("rb") as handle:
        first = handle.read(chunk_size)
        if stat.st_size > chunk_size:
            handle.seek(max(0, stat.st_size - chunk_size))
        last = handle.read(chunk_size)
    return FileFingerprint(
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        first_chunk_sha256=hashlib.sha256(first).hexdigest(),
        last_chunk_sha256=hashlib.sha256(last).hexdigest(),
    )


def derived_rows(audit_dir: Path) -> Iterator[Dict[str, Any]]:
    records_dir = audit_dir / "normalized_records"
    parts = sorted(records_dir.glob("part-*.jsonl"))
    if not parts:
        raise SystemExit(f"ERROR: no JSONL parts found under {records_dir}")

    for part in parts:
        with part.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    raise RuntimeError(
                        f"Blank JSONL line: {part}:{line_number}"
                    )
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"Non-object JSONL row: {part}:{line_number}"
                    )
                yield row


def validate_row(
    source_record_id: int,
    raw_record: Any,
    row: Mapping[str, Any],
    problems: Dict[str, int],
) -> None:
    def add(name: str) -> None:
        problems[name] = problems.get(name, 0) + 1

    if row.get("source_record_id") != source_record_id:
        add("source_record_id_mismatch")

    if not isinstance(raw_record, dict):
        add("raw_non_dictionary_record")
        return
    if "text" not in raw_record:
        add("raw_missing_text_field")
        return
    raw_text = raw_record["text"]
    if not isinstance(raw_text, str):
        add("raw_non_string_text")
        return

    if row.get("input_valid") is not True:
        add("derived_input_valid_false")
        return

    normalized_text = row.get("normalized_text")
    if not isinstance(normalized_text, str):
        add("normalized_text_not_string")
        return

    if row.get("raw_text_sha256") != sha256_text(raw_text):
        add("raw_hash_mismatch")
    if row.get("normalized_text_sha256") != sha256_text(normalized_text):
        add("normalized_hash_mismatch")
    if row.get("raw_character_count") != len(raw_text):
        add("raw_character_count_mismatch")
    if row.get("normalized_character_count") != len(normalized_text):
        add("normalized_character_count_mismatch")

    expected_delta = len(normalized_text) - len(raw_text)
    if row.get("character_delta") != expected_delta:
        add("character_delta_mismatch")

    operations = row.get("operations_applied")
    if not isinstance(operations, list):
        add("operations_not_list")
        operations = []

    if row.get("operation_count") != len(operations):
        add("operation_count_mismatch")

    unknown = set(operations) - APPROVED_OPERATIONS
    if unknown:
        add("unapproved_operation")

    changed = normalized_text != raw_text

    if operations == [APPROVED_OPERATION]:
        valid_trim = (
            len(raw_text) >= 2
            and raw_text.endswith(" ")
            and not raw_text[-2].isspace()
            and normalized_text == raw_text[:-1]
            and expected_delta == -1
        )
        if not valid_trim:
            add("invalid_trim_semantics")
    elif operations:
        add("unexpected_operation_sequence")
    else:
        if changed:
            add("unexplained_text_change")
        if expected_delta != 0:
            add("unchanged_delta_nonzero")

    # Independent idempotence check for the only approved rule.
    second_operations: List[str] = []
    second_text = normalized_text
    if (
        len(second_text) >= 2
        and second_text.endswith(" ")
        and not second_text[-2].isspace()
    ):
        second_text = second_text[:-1]
        second_operations.append(APPROVED_OPERATION)

    if second_text != normalized_text or second_operations:
        add("idempotence_failure")

    review_flags = row.get("review_flags")
    if not isinstance(review_flags, list):
        add("review_flags_not_list")
        review_flags = []

    review_required = row.get("review_required")
    if review_required is not bool(review_flags):
        add("review_boolean_mismatch")

    status = row.get("normalization_status")
    expected_status = (
        "CHANGED_SAFE_REVIEW_REQUIRED"
        if changed and review_flags
        else "CHANGED_SAFE"
        if changed
        else "UNCHANGED_REVIEW_REQUIRED"
        if review_flags
        else "UNCHANGED"
    )
    if status != expected_status:
        add("normalization_status_mismatch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently verify Phase 2 normalized JSONL against raw corpus."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-records",
        type=int,
        required=True,
        help="Expected number of raw/derived records to verify.",
    )
    parser.add_argument("--progress-every", type=int, default=10_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_records < 1:
        raise SystemExit("ERROR: --max-records must be positive.")

    raw_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not (output_dir / "_NORMALIZATION_COMPLETE").exists():
        raise SystemExit("ERROR: completion marker is missing.")
    if not (output_dir / "summary.json").exists():
        raise SystemExit("ERROR: summary.json is missing.")
    if not (output_dir / "output_manifest.json").exists():
        raise SystemExit("ERROR: output_manifest.json is missing.")

    before = fingerprint_file(raw_path)
    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output_dir / "output_manifest.json").read_text(encoding="utf-8")
    )

    problems: Dict[str, int] = {}
    verified = 0

    with raw_path.open("rb") as raw_handle:
        raw_items: Iterable[Any] = itertools.islice(
            ijson_backend.items(raw_handle, "item"),
            args.max_records,
        )
        rows = derived_rows(output_dir)

        for source_record_id, pair in enumerate(
            itertools.zip_longest(raw_items, rows, fillvalue=None)
        ):
            raw_record, row = pair
            if raw_record is None:
                problems["extra_derived_row"] = (
                    problems.get("extra_derived_row", 0) + 1
                )
                continue
            if row is None:
                problems["missing_derived_row"] = (
                    problems.get("missing_derived_row", 0) + 1
                )
                continue

            validate_row(source_record_id, raw_record, row, problems)
            verified += 1

            if args.progress_every and verified % args.progress_every == 0:
                print(
                    f"[verify phase2] records={verified:,} "
                    f"problems={sum(problems.values()):,}",
                    file=sys.stderr,
                    flush=True,
                )

    after = fingerprint_file(raw_path)
    immutability_pass = before == after

    checks = {
        "verified_record_count": verified == args.max_records,
        "summary_record_count": summary.get("records_scanned") == args.max_records,
        "summary_written_count": (
            summary.get("normalization_counts", {}).get("records_written")
            == args.max_records
        ),
        "manifest_record_count": (
            manifest.get("total_records") == args.max_records
        ),
        "problem_count_zero": sum(problems.values()) == 0,
        "immutability": immutability_pass,
    }

    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "records_verified": verified,
        "problem_counts": dict(sorted(problems.items())),
        "checks": {
            key: "PASS" if value else "FAIL"
            for key, value in checks.items()
        },
        "immutability_check": "PASS" if immutability_pass else "FAIL",
        "input_path": str(raw_path),
        "output_directory": str(output_dir),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
