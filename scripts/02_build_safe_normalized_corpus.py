#!/usr/bin/env python3
"""
Phase 2 safe derived-corpus normalizer for Uzbek Contextual Spelling Correction Lab v2.

Version 1.0 intentionally allows only narrow, mechanically safe transformations.
The initial approved operation is:

    TRIM_ONE_TRAILING_ASCII_SPACE

It removes one final U+0020 ASCII space only when the preceding character is not
whitespace. It does not repair spelling, apostrophe gaps, punctuation, merged
words, split words, source credits, or grammar.

Safety properties
-----------------
- Reads the immutable top-level JSON array only.
- Uses the pure-Python ijson backend explicitly; no silent backend fallback.
- Writes a new sharded JSONL dataset; never edits the raw corpus.
- Preserves source_record_id and raw-text SHA-256 provenance.
- Logs every applied operation and every remaining review signal.
- Validates character deltas, allowlist compliance, record accounting, and
  idempotence for every string-valued text record.
- Leaves _NORMALIZATION_INCOMPLETE unless all validations pass.

The validated Phase 1 helper script 01_inspect_raw_corpus_v2_1.py must be stored
beside this script. Its apostrophe-gap rules and file fingerprint logic are
reused so Phase 2 flags remain consistent with Phase 1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


SCRIPT_VERSION = "1.0"
SCHEMA_VERSION = "phase_02_normalized_record_v1"
SOURCE_RECORD_ID_BASE = 0

DEFAULT_INPUT = Path("/mnt/uzbekvoice_storage/text_data/normalized.json")
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/uzbekvoice_storage/projects/"
    "uzbek-contextual-spelling-correction-lab-v2/outputs/"
    "02_curated_corpus/normalize_50k_v1"
)
LEGACY_SCRIPT_NAME = "01_inspect_raw_corpus_v2_1.py"

TRIM_ONE_TRAILING_ASCII_SPACE = "TRIM_ONE_TRAILING_ASCII_SPACE"
SUPPORTED_OPERATIONS = (TRIM_ONE_TRAILING_ASCII_SPACE,)
DEFAULT_ALLOWED_OPERATIONS = (TRIM_ONE_TRAILING_ASCII_SPACE,)


@dataclass(frozen=True)
class NormalizationResult:
    normalized_text: str
    operations_applied: Tuple[str, ...]
    review_flags: Tuple[str, ...]
    high_confidence_gap_count: int
    ambiguous_gap_count: int
    high_confidence_rule_ids: Tuple[str, ...]
    ambiguous_rule_ids: Tuple[str, ...]
    normalization_status: str
    review_required: bool


class JsonlShardWriter:
    """Write deterministic JSONL parts with a fixed maximum row count."""

    def __init__(self, directory: Path, records_per_part: int) -> None:
        if records_per_part < 1:
            raise ValueError("records_per_part must be positive")
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=False)
        self.records_per_part = records_per_part
        self.total_records = 0
        self._part_index = -1
        self._part_rows = 0
        self._handle: Optional[Any] = None
        self._current_path: Optional[Path] = None
        self._parts: List[Dict[str, Any]] = []
        self.closed = False

    def _open_next_part(self) -> None:
        self._close_current_part()
        self._part_index += 1
        self._part_rows = 0
        self._current_path = self.directory / f"part-{self._part_index:05d}.jsonl"
        self._handle = self._current_path.open("w", encoding="utf-8", newline="\n")

    def _close_current_part(self) -> None:
        if self._handle is None or self._current_path is None:
            return
        self._handle.flush()
        self._handle.close()
        digest = hashlib.sha256()
        with self._current_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self._parts.append(
            {
                "part_index": self._part_index,
                "filename": self._current_path.name,
                "record_count": self._part_rows,
                "size_bytes": self._current_path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
        self._handle = None
        self._current_path = None

    def write(self, row: Mapping[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("JSONL shard writer is closed")
        if self._handle is None or self._part_rows >= self.records_per_part:
            self._open_next_part()
        assert self._handle is not None
        self._handle.write(
            json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self._part_rows += 1
        self.total_records += 1

    def close(self) -> None:
        if self.closed:
            return
        self._close_current_part()
        self.closed = True

    def manifest(self) -> Dict[str, Any]:
        if not self.closed:
            raise RuntimeError("close writer before requesting manifest")
        return {
            "format": "JSONL",
            "encoding": "UTF-8",
            "schema_version": SCHEMA_VERSION,
            "records_per_part": self.records_per_part,
            "total_records": self.total_records,
            "part_count": len(self._parts),
            "parts": self._parts,
        }


def load_phase_1_helpers() -> Any:
    path = Path(__file__).resolve().with_name(LEGACY_SCRIPT_NAME)
    if not path.is_file():
        raise SystemExit(f"ERROR: required Phase 1 helper script is missing: {path}")
    module_name = "uzbek_contextual_phase_1_v2_1_helpers"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"ERROR: could not load Phase 1 helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    required = (
        "fingerprint_file",
        "find_apostrophe_gap_matches",
        "visible_preview",
        "DeterministicSampler",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise SystemExit(
            "ERROR: Phase 1 helper script is incompatible; missing: "
            + ", ".join(missing)
        )
    return module


V21 = load_phase_1_helpers()

try:
    from ijson.backends import python as IJSON_BACKEND
except ImportError:
    IJSON_BACKEND = None  # type: ignore[assignment]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a provenance-preserving derived corpus using only explicitly "
            "allowlisted mechanical normalization operations."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-records",
        type=int,
        default=50_000,
        help="Maximum array items to process. Use 0 for the complete corpus.",
    )
    parser.add_argument(
        "--records-per-part",
        type=int,
        default=100_000,
        help="Maximum JSONL rows per output part.",
    )
    parser.add_argument("--example-limit-per-category", type=int, default=12)
    parser.add_argument("--preview-chars", type=int, default=320)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument(
        "--sample-seed",
        default="uzbek-contextual-spelling-correction-lab-v2-phase2-v1",
    )
    parser.add_argument(
        "--allow-operation",
        action="append",
        choices=SUPPORTED_OPERATIONS,
        dest="allowed_operations",
        help=(
            "Explicitly allow an operation. Repeat for multiple operations. "
            f"Default: {', '.join(DEFAULT_ALLOWED_OPERATIONS)}"
        ),
    )
    parser.add_argument("--overwrite-output-dir", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> Tuple[Path, Path, Tuple[str, ...]]:
    if args.max_records < 0:
        raise SystemExit("ERROR: --max-records must be 0 or positive.")
    for name in (
        "records_per_part",
        "example_limit_per_category",
        "preview_chars",
    ):
        if getattr(args, name) < 1:
            raise SystemExit(f"ERROR: --{name.replace('_', '-')} must be positive.")
    if args.progress_every < 0:
        raise SystemExit("ERROR: --progress-every must be 0 or positive.")

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file() or not os.access(input_path, os.R_OK):
        raise SystemExit(f"ERROR: input corpus is missing or unreadable: {input_path}")
    if output_dir == Path("/"):
        raise SystemExit("ERROR: refusing to use filesystem root as output directory.")

    requested = args.allowed_operations or list(DEFAULT_ALLOWED_OPERATIONS)
    allowed_operations = tuple(dict.fromkeys(requested))
    unknown = sorted(set(allowed_operations) - set(SUPPORTED_OPERATIONS))
    if unknown:
        raise SystemExit("ERROR: unsupported operations requested: " + ", ".join(unknown))
    return input_path, output_dir, allowed_operations


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise SystemExit(f"ERROR: output path is not a directory: {path}")
        if any(path.iterdir()):
            if not overwrite:
                raise SystemExit(
                    f"ERROR: output directory is not empty: {path}\n"
                    "Use a new directory or pass --overwrite-output-dir explicitly."
                )
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "examples").mkdir()
    (path / "_NORMALIZATION_INCOMPLETE").write_text(
        "The normalization run has not completed successfully yet.\n",
        encoding="utf-8",
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_exactly_one_safe_trailing_ascii_space(text: str) -> bool:
    """Return True only for a single final U+0020 preceded by non-whitespace."""
    return len(text) >= 2 and text[-1] == " " and not text[-2].isspace()


def apply_allowed_operations(
    text: str,
    allowed_operations: Sequence[str],
) -> Tuple[str, Tuple[str, ...]]:
    """Apply operations in a fixed deterministic order."""
    normalized = text
    applied: List[str] = []

    for operation in SUPPORTED_OPERATIONS:
        if operation not in allowed_operations:
            continue
        if operation == TRIM_ONE_TRAILING_ASCII_SPACE:
            if is_exactly_one_safe_trailing_ascii_space(normalized):
                normalized = normalized[:-1]
                applied.append(operation)
        else:  # Defensive: supported constants and implementation must stay aligned.
            raise RuntimeError(f"No implementation exists for operation: {operation}")

    return normalized, tuple(applied)


def detect_boundary_review_flags(text: str) -> List[str]:
    flags: List[str] = []
    if not text.strip():
        flags.append("EMPTY_OR_WHITESPACE_TEXT")
    if text and text[0].isspace():
        flags.append("LEADING_WHITESPACE")

    if text.endswith("  "):
        flags.append("MULTIPLE_TRAILING_ASCII_SPACES")
    elif text.endswith("\t"):
        flags.append("TRAILING_TAB")
    elif text.endswith("\n") or text.endswith("\r"):
        flags.append("TRAILING_NEWLINE")
    elif text.endswith(" ") and len(text) >= 2 and text[-2].isspace():
        flags.append("WHITESPACE_BEFORE_TRAILING_ASCII_SPACE")
    elif text and text[-1].isspace() and text[-1] != " ":
        flags.append("OTHER_TRAILING_WHITESPACE")
    return flags


def determine_status(changed: bool, review_required: bool) -> str:
    if changed and review_required:
        return "CHANGED_SAFE_REVIEW_REQUIRED"
    if changed:
        return "CHANGED_SAFE"
    if review_required:
        return "UNCHANGED_REVIEW_REQUIRED"
    return "UNCHANGED"


def normalize_text(
    text: str,
    allowed_operations: Sequence[str],
) -> NormalizationResult:
    normalized_text, operations = apply_allowed_operations(text, allowed_operations)

    review_flags = detect_boundary_review_flags(text)
    high_matches, ambiguous_matches = V21.find_apostrophe_gap_matches(normalized_text)
    high_rule_ids = tuple(sorted({item["rule_id"] for item in high_matches}))
    ambiguous_rule_ids = tuple(
        sorted({item["rule_id"] for item in ambiguous_matches})
    )
    if high_matches:
        review_flags.append("HIGH_CONFIDENCE_APOSTROPHE_GAP")
    if ambiguous_matches:
        review_flags.append("AMBIGUOUS_APOSTROPHE_GAP")

    unique_flags = tuple(sorted(set(review_flags)))
    review_required = bool(unique_flags)
    changed = normalized_text != text
    return NormalizationResult(
        normalized_text=normalized_text,
        operations_applied=operations,
        review_flags=unique_flags,
        high_confidence_gap_count=len(high_matches),
        ambiguous_gap_count=len(ambiguous_matches),
        high_confidence_rule_ids=high_rule_ids,
        ambiguous_rule_ids=ambiguous_rule_ids,
        normalization_status=determine_status(changed, review_required),
        review_required=review_required,
    )


def validate_normalization_result(
    raw_text: str,
    result: NormalizationResult,
    allowed_operations: Sequence[str],
) -> Counter[str]:
    """Return validation-problem counters; an approved result returns all zeros."""
    problems: Counter[str] = Counter()
    changed = raw_text != result.normalized_text
    delta = len(result.normalized_text) - len(raw_text)

    unapproved = set(result.operations_applied) - set(allowed_operations)
    if unapproved:
        problems["unapproved_operation"] += len(unapproved)

    if changed and not result.operations_applied:
        problems["unexplained_text_change"] += 1
    if not changed and result.operations_applied:
        problems["operation_without_text_change"] += 1

    if result.operations_applied == (TRIM_ONE_TRAILING_ASCII_SPACE,):
        if not (
            is_exactly_one_safe_trailing_ascii_space(raw_text)
            and result.normalized_text == raw_text[:-1]
            and delta == -1
        ):
            problems["invalid_trim_transformation"] += 1
    elif result.operations_applied:
        problems["unexpected_operation_sequence"] += 1
    elif delta != 0:
        problems["invalid_unchanged_delta"] += 1

    second_text, second_operations = apply_allowed_operations(
        result.normalized_text, allowed_operations
    )
    if second_text != result.normalized_text or second_operations:
        problems["idempotence_failure"] += 1

    expected_status = determine_status(changed, result.review_required)
    if result.normalization_status != expected_status:
        problems["status_mismatch"] += 1
    if result.review_required != bool(result.review_flags):
        problems["review_boolean_mismatch"] += 1

    return problems


def output_row_for_valid_text(
    source_record_id: int,
    raw_text: str,
    result: NormalizationResult,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_record_id": source_record_id,
        "input_valid": True,
        "raw_text_sha256": sha256_text(raw_text),
        "normalized_text_sha256": sha256_text(result.normalized_text),
        "raw_character_count": len(raw_text),
        "normalized_character_count": len(result.normalized_text),
        "character_delta": len(result.normalized_text) - len(raw_text),
        "normalized_text": result.normalized_text,
        "operations_applied": list(result.operations_applied),
        "operation_count": len(result.operations_applied),
        "normalization_status": result.normalization_status,
        "review_flags": list(result.review_flags),
        "review_required": result.review_required,
        "high_confidence_gap_count": result.high_confidence_gap_count,
        "ambiguous_gap_count": result.ambiguous_gap_count,
        "high_confidence_rule_ids": list(result.high_confidence_rule_ids),
        "ambiguous_rule_ids": list(result.ambiguous_rule_ids),
    }


def output_row_for_invalid_input(
    source_record_id: int,
    record: Any,
    reason: str,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_record_id": source_record_id,
        "input_valid": False,
        "raw_text_sha256": None,
        "normalized_text_sha256": None,
        "raw_character_count": None,
        "normalized_character_count": None,
        "character_delta": None,
        "normalized_text": None,
        "operations_applied": [],
        "operation_count": 0,
        "normalization_status": "INVALID_INPUT_REVIEW_REQUIRED",
        "review_flags": [reason],
        "review_required": True,
        "high_confidence_gap_count": 0,
        "ambiguous_gap_count": 0,
        "high_confidence_rule_ids": [],
        "ambiguous_rule_ids": [],
        "record_type": type(record).__name__,
    }


def write_counter_csv(path: Path, label_field: str, counter: Mapping[str, int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[label_field, "record_count"])
        writer.writeheader()
        for label, count in sorted(counter.items()):
            writer.writerow({label_field: label, "record_count": count})


def write_sampler_categories(
    sampler: Any,
    output_dir: Path,
    prefix: str,
) -> None:
    for category in sampler.categories():
        rows = sampler.rows(category)
        if not rows:
            continue
        fieldnames: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in category)
        with (output_dir / f"{prefix}{safe}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def markdown_table(rows: Iterable[Tuple[str, Any]]) -> str:
    lines = ["| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in rows)
    return "\n".join(lines)


def build_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 2 Safe Derived-Corpus Normalization — v1",
            "",
            "This run applied only explicitly allowlisted mechanical operations. "
            "It did not repair spelling, apostrophe gaps, punctuation, grammar, "
            "merged words, split words, or source credits.",
            "",
            "## Run identity",
            "",
            markdown_table(
                [
                    ("script_version", summary["script_version"]),
                    ("schema_version", summary["schema_version"]),
                    ("records_scanned", summary["records_scanned"]),
                    ("max_records_requested", summary["max_records_requested"]),
                    ("elapsed_seconds", round(summary["elapsed_seconds"], 6)),
                    ("immutability_check", summary["immutability_check"]),
                ]
            ),
            "",
            "## Normalization results",
            "",
            markdown_table(
                [(key, value) for key, value in summary["normalization_counts"].items()]
            ),
            "",
            "## Operation counts",
            "",
            markdown_table(
                [(key, value) for key, value in summary["operation_counts"].items()]
            ),
            "",
            "## Review-flag counts",
            "",
            markdown_table(
                [(key, value) for key, value in summary["review_flag_counts"].items()]
            ),
            "",
            "## Validation",
            "",
            markdown_table(
                [
                    (name, result["status"])
                    for name, result in summary["validation_checks"].items()
                ]
            ),
            "",
            "## Interpretation boundary",
            "",
            "- `CHANGED_SAFE` means only an allowlisted mechanical operation ran.",
            "- Review flags describe remaining risk; they are not gold error labels.",
            "- Apostrophe-gap signals were retained but never corrected.",
            "- The immutable raw corpus was not modified.",
            "",
        ]
    )


def run_normalization(args: argparse.Namespace) -> Dict[str, Any]:
    if IJSON_BACKEND is None:
        raise SystemExit(
            "ERROR: the pure-Python ijson backend is required. Install requirements.txt."
        )

    input_path, output_dir, allowed_operations = validate_args(args)
    prepare_output_dir(output_dir, args.overwrite_output_dir)
    run_configuration = {
        "script_version": SCRIPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "effective_allowed_operations": list(allowed_operations),
        "parser_backend": "ijson.backends.python",
        "phase_boundary": "mechanical normalization only; no linguistic correction",
    }
    atomic_write_json(output_dir / "run_configuration.json", run_configuration)

    started_at = time.time()
    pre_fingerprint = V21.fingerprint_file(input_path)
    counters: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    review_flag_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    validation_problem_counts: Counter[str] = Counter()

    status_sampler = V21.DeterministicSampler(
        args.example_limit_per_category, args.sample_seed + "|status"
    )
    review_sampler = V21.DeterministicSampler(
        args.example_limit_per_category, args.sample_seed + "|review"
    )

    writer = JsonlShardWriter(
        output_dir / "normalized_records", args.records_per_part
    )

    try:
        with input_path.open("rb") as corpus_handle:
            items: Iterator[Any] = IJSON_BACKEND.items(corpus_handle, "item")
            selected_items: Iterable[Any] = (
                itertools.islice(items, args.max_records)
                if args.max_records
                else items
            )

            for source_record_id, record in enumerate(selected_items):
                counters["records_scanned"] += 1

                if not isinstance(record, dict):
                    counters["non_dictionary_records"] += 1
                    row = output_row_for_invalid_input(
                        source_record_id, record, "NON_DICTIONARY_RECORD"
                    )
                elif "text" not in record:
                    counters["missing_text_field"] += 1
                    row = output_row_for_invalid_input(
                        source_record_id, record, "MISSING_TEXT_FIELD"
                    )
                elif not isinstance(record["text"], str):
                    counters["non_string_text_records"] += 1
                    row = output_row_for_invalid_input(
                        source_record_id, record, "NON_STRING_TEXT"
                    )
                else:
                    counters["valid_string_text_records"] += 1
                    raw_text = record["text"]
                    result = normalize_text(raw_text, allowed_operations)
                    problems = validate_normalization_result(
                        raw_text, result, allowed_operations
                    )
                    validation_problem_counts.update(problems)
                    row = output_row_for_valid_text(source_record_id, raw_text, result)

                    changed = raw_text != result.normalized_text
                    counters["records_changed" if changed else "records_unchanged"] += 1
                    counters["characters_removed"] += max(
                        0, len(raw_text) - len(result.normalized_text)
                    )
                    counters["records_review_required"] += int(result.review_required)
                    counters["records_without_review_flags"] += int(
                        not result.review_required
                    )
                    status_counts[result.normalization_status] += 1
                    operation_counts.update(result.operations_applied)
                    review_flag_counts.update(result.review_flags)

                    example = {
                        "source_record_id": source_record_id,
                        "normalization_status": result.normalization_status,
                        "operations_applied": ";".join(result.operations_applied),
                        "review_flags": ";".join(result.review_flags),
                        "raw_character_count": len(raw_text),
                        "normalized_character_count": len(result.normalized_text),
                        "raw_preview": V21.visible_preview(raw_text, args.preview_chars),
                        "normalized_preview": V21.visible_preview(
                            result.normalized_text, args.preview_chars
                        ),
                    }
                    status_sampler.add(
                        result.normalization_status, source_record_id, example
                    )
                    for flag in result.review_flags:
                        review_sampler.add(flag, source_record_id, example)

                writer.write(row)
                counters["records_written"] += 1

                if (
                    args.progress_every
                    and counters["records_scanned"] % args.progress_every == 0
                ):
                    elapsed = time.time() - started_at
                    rate = counters["records_scanned"] / max(elapsed, 1e-9)
                    atomic_write_json(
                        output_dir / "checkpoint.json",
                        {
                            "script_version": SCRIPT_VERSION,
                            "status": "RUNNING",
                            "records_scanned": counters["records_scanned"],
                            "records_written": counters["records_written"],
                            "records_changed": counters["records_changed"],
                            "records_review_required": counters[
                                "records_review_required"
                            ],
                            "rate_records_per_second": rate,
                        },
                    )
                    print(
                        f"[normalize v{SCRIPT_VERSION}] "
                        f"records={counters['records_scanned']:,} "
                        f"rate={rate:,.1f} records/s "
                        f"changed={counters['records_changed']:,} "
                        f"review_required={counters['records_review_required']:,}",
                        file=sys.stderr,
                        flush=True,
                    )

        writer.close()
        output_manifest = writer.manifest()
        atomic_write_json(output_dir / "output_manifest.json", output_manifest)

        post_fingerprint = V21.fingerprint_file(input_path)
        immutability_passed = pre_fingerprint == post_fingerprint
        if not immutability_passed:
            raise RuntimeError("Raw corpus fingerprint changed during normalization.")

        records_scanned = counters["records_scanned"]
        valid_records = counters["valid_string_text_records"]
        invalid_records = (
            counters["non_dictionary_records"]
            + counters["missing_text_field"]
            + counters["non_string_text_records"]
        )

        checks: Dict[str, Dict[str, Any]] = {}

        def add_check(name: str, observed: Any, expected: Any) -> None:
            passed = observed == expected
            checks[name] = {
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "expected": expected,
            }
            if not passed:
                raise RuntimeError(
                    f"Validation failed for {name}: observed={observed!r}, "
                    f"expected={expected!r}"
                )

        add_check("input_output_record_count", counters["records_written"], records_scanned)
        add_check("manifest_record_count", output_manifest["total_records"], records_scanned)
        add_check("input_partition", valid_records + invalid_records, records_scanned)
        add_check(
            "changed_unchanged_partition",
            counters["records_changed"] + counters["records_unchanged"],
            valid_records,
        )
        add_check(
            "operation_count_matches_changed_records",
            sum(operation_counts.values()),
            counters["records_changed"],
        )
        add_check(
            "character_removal_matches_changed_records",
            counters["characters_removed"],
            counters["records_changed"],
        )
        add_check(
            "review_partition",
            counters["records_review_required"]
            + counters["records_without_review_flags"],
            valid_records,
        )
        add_check(
            "validation_problem_count",
            sum(validation_problem_counts.values()),
            0,
        )
        add_check("immutability", immutability_passed, True)

        write_counter_csv(
            output_dir / "operation_counts.csv", "operation", operation_counts
        )
        write_counter_csv(
            output_dir / "review_flag_counts.csv", "review_flag", review_flag_counts
        )
        write_counter_csv(
            output_dir / "status_counts.csv", "normalization_status", status_counts
        )
        write_sampler_categories(
            status_sampler, output_dir / "examples", "status_"
        )
        write_sampler_categories(
            review_sampler, output_dir / "examples", "review_"
        )

        elapsed_seconds = time.time() - started_at
        summary: Dict[str, Any] = {
            "script_version": SCRIPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "input_path": str(input_path),
            "output_directory": str(output_dir),
            "max_records_requested": args.max_records,
            "records_per_part": args.records_per_part,
            "records_scanned": records_scanned,
            "allowed_operations": list(allowed_operations),
            "normalization_counts": {
                "records_written": counters["records_written"],
                "valid_string_text_records": valid_records,
                "invalid_input_records": invalid_records,
                "records_changed": counters["records_changed"],
                "records_unchanged": counters["records_unchanged"],
                "records_review_required": counters["records_review_required"],
                "records_without_review_flags": counters[
                    "records_without_review_flags"
                ],
                "characters_removed": counters["characters_removed"],
            },
            "input_problem_counts": {
                "non_dictionary_records": counters["non_dictionary_records"],
                "missing_text_field": counters["missing_text_field"],
                "non_string_text_records": counters["non_string_text_records"],
            },
            "operation_counts": {
                name: int(operation_counts.get(name, 0))
                for name in SUPPORTED_OPERATIONS
            },
            "review_flag_counts": {
                name: int(count)
                for name, count in sorted(review_flag_counts.items())
            },
            "status_counts": {
                name: int(count) for name, count in sorted(status_counts.items())
            },
            "validation_problem_counts": {
                name: int(count)
                for name, count in sorted(validation_problem_counts.items())
            },
            "validation_checks": checks,
            "output_manifest": output_manifest,
            "parser": {
                "library": "ijson",
                "backend": "python",
                "module": "ijson.backends.python",
            },
            "raw_input_fingerprint_before": asdict(pre_fingerprint),
            "raw_input_fingerprint_after": asdict(post_fingerprint),
            "immutability_check": "PASS",
            "elapsed_seconds": elapsed_seconds,
            "safety": {
                "raw_corpus_modified": False,
                "linguistic_correction_performed": False,
                "automatic_record_exclusion_performed": False,
                "review_flags_are_gold": False,
                "idempotence_checked_per_valid_record": True,
            },
        }

        atomic_write_json(output_dir / "validation_checks.json", checks)
        atomic_write_json(output_dir / "summary.json", summary)
        (output_dir / "report.md").write_text(
            build_report(summary), encoding="utf-8"
        )
        atomic_write_json(
            output_dir / "checkpoint.json",
            {
                "script_version": SCRIPT_VERSION,
                "status": "COMPLETE",
                "records_scanned": records_scanned,
                "records_written": counters["records_written"],
                "records_changed": counters["records_changed"],
                "records_review_required": counters["records_review_required"],
                "elapsed_seconds": elapsed_seconds,
            },
        )

        incomplete = output_dir / "_NORMALIZATION_INCOMPLETE"
        if incomplete.exists():
            incomplete.unlink()
        (output_dir / "_NORMALIZATION_COMPLETE").write_text(
            "Normalization completed successfully. Raw corpus immutability: PASS.\n",
            encoding="utf-8",
        )
        return summary

    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        (output_dir / "_NORMALIZATION_FAILED.txt").write_text(
            "Normalization failed. Do not approve partial outputs. "
            "See the terminal traceback.\n",
            encoding="utf-8",
        )
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_normalization(args)
    print(
        json.dumps(
            {
                "status": "PASS",
                "script_version": summary["script_version"],
                "records_scanned": summary["records_scanned"],
                "records_written": summary["normalization_counts"]["records_written"],
                "records_changed": summary["normalization_counts"]["records_changed"],
                "records_review_required": summary["normalization_counts"][
                    "records_review_required"
                ],
                "characters_removed": summary["normalization_counts"][
                    "characters_removed"
                ],
                "immutability_check": summary["immutability_check"],
                "output_directory": summary["output_directory"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
