#!/usr/bin/env python3
"""
Full-scale raw-corpus audit for Uzbek Contextual Spelling Correction Lab v2.

Version 2.2 keeps the validated v2.1 linguistic and structural heuristics but
removes the live SQLite duplicate database from the streaming quality audit.
Instead, it writes a compact, sharded SHA-256 inventory for a separate exact-
duplicate analysis pass.

Safety properties
-----------------
- Reads the immutable top-level JSON array only.
- Uses the pure-Python ijson backend explicitly; no silent backend fallback.
- Never edits, normalizes, corrects, deletes, or approves corpus text.
- Writes checkpoints and run configuration before completion.
- Leaves _AUDIT_INCOMPLETE in place unless every validation passes.
- source_record_id is the zero-based position in the JSON array.

This script must be stored beside 01_inspect_raw_corpus_v2_1.py because it
reuses the already validated v2.1 regexes, samplers, counters, and utility
classes. The duplicate analyser is 01_analyze_duplicate_hashes_v2_2.py.
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
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence


SCRIPT_VERSION = "2.2"
SOURCE_RECORD_ID_BASE = 0
DEFAULT_INPUT = Path("/mnt/uzbekvoice_storage/text_data/normalized.json")
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/uzbekvoice_storage/projects/"
    "uzbek-contextual-spelling-correction-lab-v2/outputs/"
    "01_corpus_audit/scan_200k_v2_2"
)
LEGACY_SCRIPT_NAME = "01_inspect_raw_corpus_v2_1.py"
HASH_SHARD_COUNT = 256
HASH_RECORD_STRUCT = struct.Struct(">32sQI")  # sha256, source_record_id, char_count
HASH_RECORD_SIZE = HASH_RECORD_STRUCT.size


def load_v2_1_module() -> Any:
    path = Path(__file__).resolve().with_name(LEGACY_SCRIPT_NAME)
    if not path.is_file():
        raise SystemExit(
            f"ERROR: required validated helper script is missing: {path}"
        )
    module_name = "uzbek_contextual_audit_v2_1_helpers"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"ERROR: could not load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


V21 = load_v2_1_module()

try:
    from ijson.backends import python as IJSON_BACKEND
except ImportError:
    IJSON_BACKEND = None  # type: ignore[assignment]


class HashShardWriter:
    """Write fixed-width binary identity records partitioned by first digest byte."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=False)
        self._handles: Dict[int, Any] = {}
        self.counts = [0] * HASH_SHARD_COUNT
        self.total_records = 0
        self.closed = False

    def _handle(self, shard_id: int) -> Any:
        handle = self._handles.get(shard_id)
        if handle is None:
            path = self.directory / f"hash_{shard_id:02x}.bin"
            handle = path.open("wb", buffering=1024 * 1024)
            self._handles[shard_id] = handle
        return handle

    def add(self, source_record_id: int, text: str) -> None:
        if self.closed:
            raise RuntimeError("hash shard writer is already closed")
        if source_record_id < 0 or source_record_id > 0xFFFFFFFFFFFFFFFF:
            raise OverflowError(f"source_record_id out of range: {source_record_id}")
        character_count = len(text)
        if character_count > 0xFFFFFFFF:
            raise OverflowError(
                f"character_count out of range at record {source_record_id}: "
                f"{character_count}"
            )
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        shard_id = digest[0]
        self._handle(shard_id).write(
            HASH_RECORD_STRUCT.pack(digest, source_record_id, character_count)
        )
        self.counts[shard_id] += 1
        self.total_records += 1

    def flush(self) -> None:
        for handle in self._handles.values():
            handle.flush()

    def close(self) -> None:
        if self.closed:
            return
        for handle in self._handles.values():
            handle.flush()
            handle.close()
        self.closed = True

    def manifest(self) -> Dict[str, Any]:
        if not self.closed:
            raise RuntimeError("close hash shard writer before creating manifest")
        files = []
        for shard_id, record_count in enumerate(self.counts):
            if not record_count:
                continue
            path = self.directory / f"hash_{shard_id:02x}.bin"
            size_bytes = path.stat().st_size
            expected_bytes = record_count * HASH_RECORD_SIZE
            if size_bytes != expected_bytes:
                raise RuntimeError(
                    f"hash shard size mismatch for {path}: "
                    f"observed={size_bytes}, expected={expected_bytes}"
                )
            files.append(
                {
                    "shard_id": shard_id,
                    "filename": path.name,
                    "record_count": record_count,
                    "size_bytes": size_bytes,
                }
            )
        return {
            "format": "fixed_width_binary",
            "endianness": "big",
            "record_layout": [
                {"name": "sha256", "type": "32 raw bytes"},
                {"name": "source_record_id", "type": "unsigned 64-bit integer"},
                {"name": "character_count", "type": "unsigned 32-bit integer"},
            ],
            "record_size_bytes": HASH_RECORD_SIZE,
            "shard_rule": "first byte of SHA-256 digest",
            "shard_count": HASH_SHARD_COUNT,
            "total_records": self.total_records,
            "files": files,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream-audit the Uzbek corpus using the pure-Python ijson backend. "
            "Writes compact hash shards; exact duplicates are analysed separately."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-records",
        type=int,
        default=200_000,
        help="Maximum array items to scan. Use 0 for the complete corpus.",
    )
    parser.add_argument("--block-size", type=int, default=10_000)
    parser.add_argument("--short-record-max-chars", type=int, default=120)
    parser.add_argument("--standalone-credit-max-chars", type=int, default=220)
    parser.add_argument("--example-limit-per-category", type=int, default=12)
    parser.add_argument("--extreme-record-limit", type=int, default=25)
    parser.add_argument("--preview-chars", type=int, default=320)
    parser.add_argument("--context-chars", type=int, default=90)
    parser.add_argument(
        "--sample-seed",
        default="uzbek-contextual-spelling-correction-lab-v2-audit-v2.2",
    )
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_records < 0:
        raise SystemExit("ERROR: --max-records must be 0 or positive.")
    for name in (
        "block_size",
        "short_record_max_chars",
        "standalone_credit_max_chars",
        "example_limit_per_category",
        "extreme_record_limit",
        "preview_chars",
        "context_chars",
    ):
        if getattr(args, name) < 1:
            raise SystemExit(f"ERROR: --{name.replace('_', '-')} must be positive.")
    if args.progress_every < 0:
        raise SystemExit("ERROR: --progress-every must be 0 or positive.")
    input_path = args.input.expanduser().resolve()
    output_path = args.output_dir.expanduser().resolve()
    if not input_path.is_file() or not os.access(input_path, os.R_OK):
        raise SystemExit(f"ERROR: input corpus is missing or unreadable: {input_path}")
    if output_path == Path("/"):
        raise SystemExit("ERROR: refusing to use filesystem root as output directory.")


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
    (path / "quality_flag_examples").mkdir()
    (path / "category_examples").mkdir()
    (path / "_AUDIT_INCOMPLETE").write_text(
        "The audit has not completed successfully yet.\n", encoding="utf-8"
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def write_checkpoint(
    output_dir: Path,
    counters: Counter[str],
    hash_writer: HashShardWriter,
    started_at_epoch: float,
) -> None:
    elapsed = time.time() - started_at_epoch
    records = counters["records_scanned"]
    atomic_write_json(
        output_dir / "checkpoint.json",
        {
            "script_version": SCRIPT_VERSION,
            "status": "RUNNING",
            "records_scanned": records,
            "string_text_records": counters["string_text_records"],
            "hash_records_written": hash_writer.total_records,
            "records_with_high_confidence_gap": counters[
                "records_with_high_confidence_gap"
            ],
            "records_with_ambiguous_gap": counters[
                "records_with_ambiguous_gap"
            ],
            "elapsed_seconds": elapsed,
            "average_records_per_second": records / max(elapsed, 1e-9),
            "updated_at_epoch": time.time(),
        },
    )


def markdown_table(rows: Sequence[tuple[str, Any]]) -> str:
    lines = ["| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in rows)
    return "\n".join(lines)


def build_report(summary: Mapping[str, Any]) -> str:
    structure = summary["record_structure"]
    length = summary["text_length_characters"]
    apostrophe = summary["apostrophe_gap"]
    hashes = summary["hash_inventory"]
    lines = [
        "# Phase 1 Raw Corpus Audit — v2.2",
        "",
        "This report is descriptive only. No corpus text was corrected, normalised, "
        "deleted, excluded, or automatically approved.",
        "",
        "## Run identity",
        "",
        markdown_table(
            [
                ("script_version", summary["script_version"]),
                ("records_scanned", summary["records_scanned"]),
                ("max_records_requested", summary["max_records_requested"]),
                ("source_record_id_base", summary["source_record_id_base"]),
                ("parser_backend", summary["parser"]["backend"]),
                ("elapsed_seconds", round(summary["elapsed_seconds"], 6)),
                ("immutability_check", summary["immutability_check"]),
            ]
        ),
        "",
        "## Structural findings",
        "",
        markdown_table([(key, value) for key, value in structure.items()]),
        "",
        "## Text length in characters",
        "",
        markdown_table([(key, value) for key, value in length.items()]),
        "",
        "## Boundary and quality flags",
        "",
        markdown_table(
            [(key, value) for key, value in summary["quality_flags"].items()]
        ),
        "",
        "## Apostrophe-gap review signals",
        "",
        markdown_table(
            [
                ("records_with_high_confidence_gap", apostrophe["records_with_high_confidence_gap"]),
                ("records_with_ambiguous_gap", apostrophe["records_with_ambiguous_gap"]),
                ("records_with_any_gap_signal", apostrophe["records_with_any_gap_signal"]),
                ("high_confidence_match_count", apostrophe["high_confidence_match_count"]),
                ("ambiguous_match_count", apostrophe["ambiguous_match_count"]),
            ]
        ),
        "",
        "## Hash inventory for separate duplicate analysis",
        "",
        markdown_table(
            [
                ("hash_records_written", hashes["total_records"]),
                ("binary_record_size_bytes", hashes["record_size_bytes"]),
                ("nonempty_shards", len(hashes["files"])),
                ("exact_duplicates_calculated_here", False),
            ]
        ),
        "",
        "Exact duplicate statistics are intentionally calculated by "
        "`01_analyze_duplicate_hashes_v2_2.py`, not by this streaming audit.",
        "",
        "## Source/credit and short-record heuristics",
        "",
        "These remain review-only. Manual review found that the standalone-credit "
        "heuristic can produce false positives when substantive text ends with a "
        "publication marker; it must not be used alone to delete data.",
        "",
        markdown_table(
            [(key, value) for key, value in summary["source_credit"].items()]
        ),
        "",
        markdown_table(
            [(key, value) for key, value in summary["short_record_roles"].items()]
        ),
        "",
        "## Interpretation boundary",
        "",
        "- Flagged text is not automatically proven wrong.",
        "- Unflagged text is not automatically proven correct.",
        "- Hash equality is only a duplicate candidate until exact verification.",
        "- The immutable raw corpus was not modified.",
        "",
    ]
    return "\n".join(lines)


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    if IJSON_BACKEND is None:
        raise SystemExit(
            "ERROR: the pure-Python ijson backend is required. Install requirements.txt."
        )
    validate_args(args)
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prepare_output_dir(output_dir, args.overwrite_output_dir)

    run_configuration = {
        "script_version": SCRIPT_VERSION,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "parser_backend": "ijson.backends.python",
        "duplicate_strategy": "compact hash shards; exact analysis separate",
        "legacy_helper_script": str(
            Path(__file__).resolve().with_name(LEGACY_SCRIPT_NAME)
        ),
    }
    atomic_write_json(output_dir / "run_configuration.json", run_configuration)

    started_at_epoch = time.time()
    pre_fingerprint = V21.fingerprint_file(input_path)
    boundary_info = V21.inspect_json_boundaries(input_path)

    counters: Counter[str] = Counter()
    quality_flag_counts: Counter[str] = Counter()
    script_counts: Counter[str] = Counter()
    apostrophe_variant_counts: Counter[str] = Counter()
    suspicious_unicode_total: Counter[str] = Counter()
    apostrophe_rule_counts: Counter[str] = Counter()
    source_marker_counts: Counter[str] = Counter()
    short_role_counts: Counter[str] = Counter()

    length_histogram = V21.LengthHistogram()
    extremes = V21.ExtremeRecords(args.extreme_record_limit)
    block_accumulator = V21.BlockAccumulator(args.block_size)
    quality_sampler = V21.DeterministicSampler(
        args.example_limit_per_category, args.sample_seed + "|quality"
    )
    category_sampler = V21.DeterministicSampler(
        args.example_limit_per_category, args.sample_seed + "|category"
    )
    apostrophe_sampler = V21.DeterministicSampler(
        args.example_limit_per_category, args.sample_seed + "|apostrophe"
    )

    hash_writer = HashShardWriter(output_dir / "hash_shards")
    apostrophe_records_path = output_dir / "apostrophe_gap_records.csv"
    block_distribution_path = output_dir / "apostrophe_gap_block_distribution.csv"
    source_credit_records_path = output_dir / "source_credit_records.csv"
    short_record_roles_path = output_dir / "short_record_roles.csv"

    apostrophe_record_fields = [
        "source_record_id", "character_count", "high_confidence_match_count",
        "ambiguous_match_count", "high_confidence_rule_ids", "ambiguous_rule_ids",
        "matches_per_1000_characters", "review_status", "text_preview",
    ]
    source_credit_fields = [
        "source_record_id", "character_count", "contains_source_or_credit_phrase",
        "standalone_source_or_credit", "matched_marker_ids", "person_name_like",
        "review_status", "text_preview",
    ]
    short_role_fields = [
        "source_record_id", "character_count", "short_record_role",
        "standalone_source_or_credit", "person_name_like", "review_status",
        "text_preview",
    ]

    block_rows_written = 0
    block_validation_totals: Counter[str] = Counter()

    def account_block(row: Mapping[str, Any]) -> None:
        nonlocal block_rows_written
        block_rows_written += 1
        for key in (
            "records_scanned_in_block",
            "records_with_high_confidence_gap",
            "records_with_ambiguous_gap",
            "high_confidence_match_count",
            "ambiguous_match_count",
        ):
            block_validation_totals[key] += int(row[key])

    try:
        with (
            apostrophe_records_path.open("w", encoding="utf-8", newline="") as apostrophe_handle,
            block_distribution_path.open("w", encoding="utf-8", newline="") as block_handle,
            source_credit_records_path.open("w", encoding="utf-8", newline="") as source_handle,
            short_record_roles_path.open("w", encoding="utf-8", newline="") as short_handle,
            input_path.open("rb") as corpus_handle,
        ):
            apostrophe_writer = csv.DictWriter(apostrophe_handle, fieldnames=apostrophe_record_fields)
            block_writer = csv.DictWriter(block_handle, fieldnames=V21.BlockAccumulator.FIELDNAMES)
            source_writer = csv.DictWriter(source_handle, fieldnames=source_credit_fields)
            short_writer = csv.DictWriter(short_handle, fieldnames=short_role_fields)
            for writer in (apostrophe_writer, block_writer, source_writer, short_writer):
                writer.writeheader()

            items: Iterator[Any] = IJSON_BACKEND.items(corpus_handle, "item")
            selected_items: Iterable[Any] = (
                itertools.islice(items, args.max_records) if args.max_records else items
            )

            for source_record_id, record in enumerate(selected_items):
                counters["records_scanned"] += 1
                high_matches: List[Dict[str, Any]] = []
                ambiguous_matches: List[Dict[str, Any]] = []

                if isinstance(record, dict):
                    counters["dictionary_records"] += 1
                    if "text" not in record:
                        counters["missing_text_field"] += 1
                        quality_flag_counts["missing_text_field"] += 1
                        quality_sampler.add(
                            "missing_text_field", source_record_id,
                            {
                                "source_record_id": source_record_id,
                                "record_type": type(record).__name__,
                                "record_preview": V21.visible_preview(
                                    json.dumps(record, ensure_ascii=False), args.preview_chars
                                ),
                            },
                        )
                    else:
                        counters["records_with_text_field"] += 1
                        text = record["text"]
                        if isinstance(text, str):
                            counters["string_text_records"] += 1
                            character_count = len(text)
                            text_preview = V21.visible_preview(text, args.preview_chars)
                            length_histogram.add(character_count)
                            extremes.add(source_record_id, character_count, text_preview)
                            hash_writer.add(source_record_id, text)

                            if not text.strip():
                                counters["empty_or_whitespace_texts"] += 1
                                quality_flag_counts["empty_or_whitespace_text"] += 1

                            has_leading = bool(text) and text[0].isspace()
                            has_trailing = bool(text) and text[-1].isspace()
                            repeated_horizontal = (
                                V21.REPEATED_HORIZONTAL_WHITESPACE_RE.search(text) is not None
                            )
                            if has_leading:
                                quality_flag_counts["leading_whitespace"] += 1
                                quality_sampler.add(
                                    "leading_whitespace", source_record_id,
                                    {"source_record_id": source_record_id, "character_count": character_count, "text_preview": text_preview},
                                )
                            if has_trailing:
                                quality_flag_counts["trailing_whitespace"] += 1
                                quality_sampler.add(
                                    "trailing_whitespace", source_record_id,
                                    {"source_record_id": source_record_id, "character_count": character_count, "text_preview": text_preview},
                                )
                            if repeated_horizontal:
                                quality_flag_counts["repeated_horizontal_whitespace"] += 1
                                quality_sampler.add(
                                    "repeated_horizontal_whitespace", source_record_id,
                                    {"source_record_id": source_record_id, "character_count": character_count, "text_preview": text_preview},
                                )

                            if text.startswith(" "):
                                quality_flag_counts["starts_with_ascii_space"] += 1
                            if text.startswith("\t"):
                                quality_flag_counts["starts_with_tab"] += 1
                            if text.startswith("\n") or text.startswith("\r"):
                                quality_flag_counts["starts_with_newline"] += 1
                            if text.endswith(" ") and not text.endswith("  "):
                                quality_flag_counts["ends_with_exactly_one_ascii_space"] += 1
                            elif text.endswith("  "):
                                quality_flag_counts["ends_with_multiple_ascii_spaces"] += 1
                            if text.endswith("\t"):
                                quality_flag_counts["ends_with_tab"] += 1
                            if text.endswith("\n") or text.endswith("\r"):
                                quality_flag_counts["ends_with_newline"] += 1
                            if not has_trailing:
                                quality_flag_counts["no_trailing_whitespace"] += 1

                            script_counts[V21.classify_script(text)] += 1
                            for variant_id, char in V21.APOSTROPHE_VARIANTS.items():
                                count = text.count(char)
                                if count:
                                    apostrophe_variant_counts[variant_id] += count
                            unicode_counts = V21.suspicious_unicode_counts(text)
                            if unicode_counts:
                                quality_flag_counts["suspicious_unicode_record"] += 1
                                suspicious_unicode_total.update(unicode_counts)
                                quality_sampler.add(
                                    "suspicious_unicode_record", source_record_id,
                                    {
                                        "source_record_id": source_record_id,
                                        "character_count": character_count,
                                        "suspicious_codepoints": ";".join(sorted(unicode_counts)),
                                        "suspicious_occurrence_count": sum(unicode_counts.values()),
                                        "text_preview": text_preview,
                                    },
                                )

                            high_matches, ambiguous_matches = V21.find_apostrophe_gap_matches(text)
                            high_count = len(high_matches)
                            ambiguous_count = len(ambiguous_matches)
                            if high_count:
                                counters["records_with_high_confidence_gap"] += 1
                            if ambiguous_count:
                                counters["records_with_ambiguous_gap"] += 1
                            if high_count or ambiguous_count:
                                counters["records_with_any_gap_signal"] += 1
                                high_rule_ids = sorted({m["rule_id"] for m in high_matches})
                                ambiguous_rule_ids = sorted({m["rule_id"] for m in ambiguous_matches})
                                total_matches = high_count + ambiguous_count
                                apostrophe_writer.writerow(
                                    {
                                        "source_record_id": source_record_id,
                                        "character_count": character_count,
                                        "high_confidence_match_count": high_count,
                                        "ambiguous_match_count": ambiguous_count,
                                        "high_confidence_rule_ids": ";".join(high_rule_ids),
                                        "ambiguous_rule_ids": ";".join(ambiguous_rule_ids),
                                        "matches_per_1000_characters": round(1000.0 * total_matches / max(1, character_count), 6),
                                        "review_status": V21.review_status_for_gap(high_count, ambiguous_count),
                                        "text_preview": text_preview,
                                    }
                                )
                            for confidence_group, matches in (
                                ("HIGH_CONFIDENCE_APOSTROPHE_GAP", high_matches),
                                ("AMBIGUOUS_APOSTROPHE_GAP", ambiguous_matches),
                            ):
                                for match in matches:
                                    apostrophe_rule_counts[match["rule_id"]] += 1
                                    apostrophe_sampler.add(
                                        f"{confidence_group}__{match['rule_id']}",
                                        source_record_id,
                                        {
                                            "source_record_id": source_record_id,
                                            "confidence_group": confidence_group,
                                            "rule_id": match["rule_id"],
                                            "matched_fragment": match["matched_fragment"],
                                            "match_start": match["match_start"],
                                            "match_end": match["match_end"],
                                            "context_preview": V21.match_context(
                                                text, match["match_start"], match["match_end"], args.context_chars
                                            ),
                                        },
                                    )

                            name_like = V21.is_name_like(text)
                            if name_like:
                                quality_flag_counts["person_name_like"] += 1
                                quality_sampler.add(
                                    "person_name_like", source_record_id,
                                    {"source_record_id": source_record_id, "character_count": character_count, "text_preview": text_preview},
                                )
                            marker_ids = V21.matched_source_credit_markers(text)
                            source_marker_counts.update(marker_ids)
                            contains_source_credit = bool(marker_ids)
                            standalone_source_credit = V21.is_standalone_source_or_credit(
                                text, marker_ids, args.standalone_credit_max_chars, name_like
                            )
                            if contains_source_credit:
                                counters["contains_source_or_credit_phrase"] += 1
                            if standalone_source_credit:
                                counters["standalone_source_or_credit"] += 1
                            if contains_source_credit or standalone_source_credit:
                                source_row = {
                                    "source_record_id": source_record_id,
                                    "character_count": character_count,
                                    "contains_source_or_credit_phrase": str(contains_source_credit).lower(),
                                    "standalone_source_or_credit": str(standalone_source_credit).lower(),
                                    "matched_marker_ids": ";".join(marker_ids),
                                    "person_name_like": str(name_like).lower(),
                                    "review_status": "REVIEW_REQUIRED",
                                    "text_preview": text_preview,
                                }
                                source_writer.writerow(source_row)
                                if contains_source_credit:
                                    category_sampler.add("CONTAINS_SOURCE_OR_CREDIT_PHRASE", source_record_id, source_row)
                                if standalone_source_credit:
                                    category_sampler.add("STANDALONE_SOURCE_OR_CREDIT", source_record_id, source_row)

                            role = V21.short_record_role(
                                text, args.short_record_max_chars, standalone_source_credit, name_like
                            )
                            if role is not None:
                                short_role_counts[role] += 1
                                short_row = {
                                    "source_record_id": source_record_id,
                                    "character_count": character_count,
                                    "short_record_role": role,
                                    "standalone_source_or_credit": str(standalone_source_credit).lower(),
                                    "person_name_like": str(name_like).lower(),
                                    "review_status": "REVIEW_REQUIRED",
                                    "text_preview": text_preview,
                                }
                                short_writer.writerow(short_row)
                                category_sampler.add(role, source_record_id, short_row)
                        else:
                            counters["non_string_text_records"] += 1
                            quality_flag_counts["non_string_text"] += 1
                else:
                    counters["non_dictionary_records"] += 1
                    quality_flag_counts["non_dictionary_record"] += 1

                completed_block = block_accumulator.add(
                    source_record_id, len(high_matches), len(ambiguous_matches)
                )
                if completed_block is not None:
                    block_writer.writerow(completed_block)
                    account_block(completed_block)

                if args.progress_every and counters["records_scanned"] % args.progress_every == 0:
                    apostrophe_handle.flush()
                    block_handle.flush()
                    source_handle.flush()
                    short_handle.flush()
                    hash_writer.flush()
                    write_checkpoint(output_dir, counters, hash_writer, started_at_epoch)
                    elapsed = time.time() - started_at_epoch
                    rate = counters["records_scanned"] / max(elapsed, 1e-9)
                    print(
                        f"[audit v{SCRIPT_VERSION}] records={counters['records_scanned']:,} "
                        f"rate={rate:,.1f} records/s "
                        f"high_gap_records={counters['records_with_high_confidence_gap']:,} "
                        f"ambiguous_gap_records={counters['records_with_ambiguous_gap']:,} "
                        f"hash_records={hash_writer.total_records:,}",
                        file=sys.stderr,
                        flush=True,
                    )

            final_block = block_accumulator.finalize()
            if final_block is not None:
                block_writer.writerow(final_block)
                account_block(final_block)

        hash_writer.close()
        hash_manifest = hash_writer.manifest()
        atomic_write_json(output_dir / "hash_manifest.json", hash_manifest)

        for key in ("trailing_whitespace", "leading_whitespace", "repeated_horizontal_whitespace"):
            quality_flag_counts[key] = quality_flag_counts.get(key, 0)

        V21.write_sampler_categories(quality_sampler, output_dir / "quality_flag_examples", prefix="")
        V21.write_sampler_categories(category_sampler, output_dir / "category_examples", prefix="")
        V21.write_sampler_categories(apostrophe_sampler, output_dir / "category_examples", prefix="apostrophe_")
        V21.write_csv(
            output_dir / "shortest_record_examples.csv",
            ["source_record_id", "character_count", "text_preview"],
            extremes.shortest_rows(),
        )
        V21.write_csv(
            output_dir / "longest_record_examples.csv",
            ["source_record_id", "character_count", "text_preview"],
            extremes.longest_rows(),
        )
        V21.write_csv(
            output_dir / "boundary_whitespace_summary.csv",
            ["metric", "record_count"],
            [
                {"metric": key, "record_count": quality_flag_counts.get(key, 0)}
                for key in (
                    "leading_whitespace", "trailing_whitespace", "starts_with_ascii_space",
                    "starts_with_tab", "starts_with_newline", "ends_with_exactly_one_ascii_space",
                    "ends_with_multiple_ascii_spaces", "ends_with_tab", "ends_with_newline",
                    "no_trailing_whitespace", "repeated_horizontal_whitespace",
                )
            ],
        )

        post_fingerprint = V21.fingerprint_file(input_path)
        if pre_fingerprint != post_fingerprint:
            raise RuntimeError("Input corpus fingerprint changed during the audit.")

        records_scanned = counters["records_scanned"]
        high_match_total = sum(
            apostrophe_rule_counts[rule_id] for rule_id in V21.HIGH_CONFIDENCE_RULE_IDS
        )
        ambiguous_match_total = sum(
            apostrophe_rule_counts[rule_id] for rule_id in V21.AMBIGUOUS_RULE_IDS
        )
        validation_checks: Dict[str, Dict[str, Any]] = {}

        def add_check(name: str, observed: Any, expected: Any) -> None:
            passed = observed == expected
            validation_checks[name] = {
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "expected": expected,
            }
            if not passed:
                raise RuntimeError(
                    f"Internal validation failed for {name}: "
                    f"observed={observed!r}, expected={expected!r}"
                )

        add_check("record_type_partition", counters["dictionary_records"] + counters["non_dictionary_records"], records_scanned)
        add_check("dictionary_text_field_partition", counters["records_with_text_field"] + counters["missing_text_field"], counters["dictionary_records"])
        add_check("text_type_partition", counters["string_text_records"] + counters["non_string_text_records"], counters["records_with_text_field"])
        add_check("trailing_boundary_partition", quality_flag_counts["trailing_whitespace"] + quality_flag_counts["no_trailing_whitespace"], counters["string_text_records"])
        add_check("block_record_total", block_validation_totals["records_scanned_in_block"], records_scanned)
        add_check("block_high_confidence_record_total", block_validation_totals["records_with_high_confidence_gap"], counters["records_with_high_confidence_gap"])
        add_check("block_ambiguous_record_total", block_validation_totals["records_with_ambiguous_gap"], counters["records_with_ambiguous_gap"])
        add_check("block_high_confidence_match_total", block_validation_totals["high_confidence_match_count"], high_match_total)
        add_check("block_ambiguous_match_total", block_validation_totals["ambiguous_match_count"], ambiguous_match_total)
        add_check("hash_record_total", hash_manifest["total_records"], counters["string_text_records"])
        add_check("hash_shard_record_total", sum(item["record_count"] for item in hash_manifest["files"]), counters["string_text_records"])

        elapsed_seconds = time.time() - started_at_epoch
        summary: Dict[str, Any] = {
            "script_version": SCRIPT_VERSION,
            "source_record_id_base": SOURCE_RECORD_ID_BASE,
            "input": {
                "path": str(input_path),
                "size_bytes": pre_fingerprint.size_bytes,
                "mtime_ns": pre_fingerprint.mtime_ns,
                "inode": pre_fingerprint.inode,
                "device": pre_fingerprint.device,
                "first_chunk_sha256": pre_fingerprint.first_chunk_sha256,
                "last_chunk_sha256": pre_fingerprint.last_chunk_sha256,
                **boundary_info,
            },
            "output_directory": str(output_dir),
            "max_records_requested": args.max_records,
            "block_size": args.block_size,
            "records_scanned": records_scanned,
            "record_structure": {
                "dictionary_records": counters["dictionary_records"],
                "non_dictionary_records": counters["non_dictionary_records"],
                "records_with_text_field": counters["records_with_text_field"],
                "missing_text_field": counters["missing_text_field"],
                "string_text_records": counters["string_text_records"],
                "non_string_text_records": counters["non_string_text_records"],
                "empty_or_whitespace_texts": counters["empty_or_whitespace_texts"],
            },
            "text_length_characters": length_histogram.summary(),
            "quality_flags": {key: int(value) for key, value in sorted(quality_flag_counts.items())},
            "script_distribution": {key: int(value) for key, value in sorted(script_counts.items())},
            "apostrophe_variants": {key: int(apostrophe_variant_counts.get(key, 0)) for key in V21.APOSTROPHE_VARIANTS},
            "suspicious_unicode_occurrences": {key: int(value) for key, value in sorted(suspicious_unicode_total.items())},
            "apostrophe_gap": {
                "records_with_high_confidence_gap": counters["records_with_high_confidence_gap"],
                "records_with_ambiguous_gap": counters["records_with_ambiguous_gap"],
                "records_with_any_gap_signal": counters["records_with_any_gap_signal"],
                "high_confidence_match_count": high_match_total,
                "ambiguous_match_count": ambiguous_match_total,
                "rule_match_counts": {rule_id: int(apostrophe_rule_counts.get(rule_id, 0)) for rule_id in V21.APOSTROPHE_GAP_RULES},
                "high_confidence_rule_ids": sorted(V21.HIGH_CONFIDENCE_RULE_IDS),
                "ambiguous_rule_ids": sorted(V21.AMBIGUOUS_RULE_IDS),
                "review_only": True,
                "text_corrected": False,
            },
            "source_credit": {
                "CONTAINS_SOURCE_OR_CREDIT_PHRASE": counters["contains_source_or_credit_phrase"],
                "STANDALONE_SOURCE_OR_CREDIT": counters["standalone_source_or_credit"],
                "marker_record_counts": {key: int(value) for key, value in sorted(source_marker_counts.items())},
                "review_only": True,
            },
            "short_record_roles": {
                "SHORT_SENTENCE_LIKE": short_role_counts["SHORT_SENTENCE_LIKE"],
                "SHORT_CREDIT_LIKE": short_role_counts["SHORT_CREDIT_LIKE"],
                "SHORT_FRAGMENT_OR_UNKNOWN": short_role_counts["SHORT_FRAGMENT_OR_UNKNOWN"],
                "total_short_records": sum(short_role_counts.values()),
                "review_only": True,
            },
            "block_distribution": {
                "block_rows_written": block_rows_written,
                "records_accounted_for": block_validation_totals["records_scanned_in_block"],
                "validation_totals": {key: int(value) for key, value in sorted(block_validation_totals.items())},
            },
            "hash_inventory": hash_manifest,
            "duplicates": {
                "status": "PENDING_SEPARATE_EXACT_ANALYSIS",
                "analyser": "01_analyze_duplicate_hashes_v2_2.py",
            },
            "validation_checks": validation_checks,
            "deterministic_sampling": {
                "seed": args.sample_seed,
                "limit_per_category": args.example_limit_per_category,
            },
            "elapsed_seconds": elapsed_seconds,
            "immutability_check": "PASS",
            "parser": {
                "library": "ijson",
                "version": getattr(IJSON_BACKEND, "__version__", "provided by parent package"),
                "backend": "python",
                "module": "ijson.backends.python",
                "item_prefix": "item",
            },
            "safety": {
                "raw_corpus_modified": False,
                "automatic_correction_performed": False,
                "automatic_exclusion_performed": False,
                "heuristic_labels_are_gold": False,
                "review_required": True,
            },
        }

        atomic_write_json(output_dir / "summary.json", summary)
        atomic_write_json(output_dir / "validation_checks.json", validation_checks)
        (output_dir / "report.md").write_text(build_report(summary), encoding="utf-8")
        write_checkpoint(output_dir, counters, hash_writer, started_at_epoch)
        checkpoint = json.loads((output_dir / "checkpoint.json").read_text(encoding="utf-8"))
        checkpoint["status"] = "COMPLETE"
        atomic_write_json(output_dir / "checkpoint.json", checkpoint)

        incomplete = output_dir / "_AUDIT_INCOMPLETE"
        if incomplete.exists():
            incomplete.unlink()
        (output_dir / "_AUDIT_COMPLETE").write_text(
            "Audit completed successfully. Immutability check: PASS. "
            "Exact duplicate analysis remains separate.\n",
            encoding="utf-8",
        )
        return summary

    except Exception:
        try:
            hash_writer.close()
        except Exception:
            pass
        (output_dir / "_AUDIT_FAILED.txt").write_text(
            "The audit failed. Do not approve partial outputs. See the terminal traceback.\n",
            encoding="utf-8",
        )
        raise


def main() -> None:
    args = parse_args()
    summary = run_audit(args)
    print(
        json.dumps(
            {
                "status": "PASS",
                "script_version": summary["script_version"],
                "records_scanned": summary["records_scanned"],
                "dictionary_records": summary["record_structure"]["dictionary_records"],
                "string_text_records": summary["record_structure"]["string_text_records"],
                "hash_records_written": summary["hash_inventory"]["total_records"],
                "records_with_high_confidence_gap": summary["apostrophe_gap"]["records_with_high_confidence_gap"],
                "records_with_ambiguous_gap": summary["apostrophe_gap"]["records_with_ambiguous_gap"],
                "block_rows_written": summary["block_distribution"]["block_rows_written"],
                "immutability_check": summary["immutability_check"],
                "duplicate_analysis": "PENDING_SEPARATE_EXACT_ANALYSIS",
                "output_directory": summary["output_directory"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
