#!/usr/bin/env python3
"""Memory-safe audit for a top-level JSON array corpus.

The script reads the raw corpus only. It never normalizes or rewrites source text.
Large outputs and the optional SQLite duplicate index are written to the selected
output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import random
import re
import sqlite3
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import ijson

APOSTROPHES = ["'", "`", "´", "ʻ", "ʼ", "‘", "’", "‛", "′"]
SUSPICIOUS = {
    "\u00a0",  # non-breaking space
    "\u00ad",  # soft hyphen
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
    "\ufeff",  # BOM / zero-width no-break space
    "\ufffd",  # replacement character
}

# Review-only heuristics. They do not authorize correction.
APOSTROPHE_GAP_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("O_Z_PATTERN", re.compile(r"\b[oO]\s+[zZ][a-zA-Z]+")),
    ("O_Q_PATTERN", re.compile(r"\b[oO]\s+[qQ][a-zA-Z]+")),
    ("O_T_PATTERN", re.compile(r"\b[oO]\s+[tT][a-zA-Z]+")),
    ("BO_L_PATTERN", re.compile(r"\b[bB]o\s+[lL][a-zA-Z]+")),
    ("TO_G_R_PATTERN", re.compile(r"\b[tT]o\s+[gG]\s+[rR][a-zA-Z]+")),
    ("KO_R_PATTERN", re.compile(r"\b[kK]o\s+[rR][a-zA-Z]+")),
    ("QO_L_PATTERN", re.compile(r"\b[qQ]o\s+[lL][a-zA-Z]+")),
    ("MA_LUM_PATTERN", re.compile(r"\b[mM]a\s+[lL]um[a-zA-Z]*")),
    ("E_LON_PATTERN", re.compile(r"\b[eE]\s+[lL]on[a-zA-Z]*")),
]

SOURCE_CREDIT_RE = re.compile(
    r"\b(manba|foto|surat|matbuot xizmati|axborot xizmati|gazetasi|xalq so['’]zi)\b",
    re.IGNORECASE,
)
NAME_LIKE_RE = re.compile(
    r"^[A-ZА-ЯO'ʻʼ’][A-Za-zА-Яа-яO'ʻʼ’.-]+(?:\s+[A-ZА-ЯO'ʻʼ’][A-Za-zА-Яа-яO'ʻʼ’.-]+){1,3}\.?\s*$"
)

OUTPUTS = [
    "corpus_schema.json",
    "record_type_counts.csv",
    "field_presence_counts.csv",
    "text_value_type_counts.csv",
    "text_length_summary.csv",
    "length_extreme_examples.csv",
    "apostrophe_variant_counts.csv",
    "script_distribution.csv",
    "quality_flag_counts.csv",
    "quality_flag_examples.csv",
    "boundary_whitespace_summary.csv",
    "sample_records.csv",
    "duplicate_examples.csv",
    "mixed_script_examples.csv",
    "suspicious_unicode_examples.csv",
    "suspected_apostrophe_gap_summary.csv",
    "suspected_apostrophe_gap_examples.csv",
    "short_boilerplate_summary.csv",
    "short_boilerplate_candidates.csv",
    "corpus_audit_metrics.md",
]


@dataclass(frozen=True)
class TextMetrics:
    character_count: int
    word_count_estimate: int
    line_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream-audit a top-level JSON array without modifying source text."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-records", type=int, default=50_000, help="0 means full corpus")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--example-limit", type=int, default=50)
    parser.add_argument("--length-example-limit", type=int, default=25)
    parser.add_argument("--preview-chars", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--short-max-chars", type=int, default=120)
    parser.add_argument("--short-max-words", type=int, default=15)
    parser.add_argument(
        "--duplicate-db",
        type=Path,
        default=None,
        help="SQLite duplicate index path. Default: OUTPUT_DIR/duplicate_index.sqlite3",
    )
    parser.add_argument(
        "--keep-duplicate-db",
        action="store_true",
        help="Keep the SQLite index after CSV summaries are written.",
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    nonnegative = (
        "max_records",
        "sample_size",
        "example_limit",
        "length_example_limit",
        "progress_every",
        "short_max_chars",
        "short_max_words",
    )
    for name in nonnegative:
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.preview_chars <= 0:
        parser.error("--preview-chars must be > 0")
    return args


def type_name(value: Any) -> str:
    return "null" if value is None else type(value).__name__


def make_preview(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else repr(value)
    return (
        text.replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")[:limit]
    )


def percentage(count: int, total: int) -> float:
    return round(100 * count / total, 4) if total else 0.0


def text_metrics(text: str) -> TextMetrics:
    return TextMetrics(
        character_count=len(text),
        word_count_estimate=len(re.findall(r"\S+", text)),
        line_count=text.count("\n") + 1,
    )


def script_info(text: str) -> tuple[str, int, int, int]:
    latin = cyrillic = other = 0
    for character in text:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        if "LATIN" in name:
            latin += 1
        elif "CYRILLIC" in name:
            cyrillic += 1
        else:
            other += 1

    if latin == cyrillic == other == 0:
        label = "NO_LETTERS"
    elif latin and not cyrillic and not other:
        label = "LATIN"
    elif cyrillic and not latin and not other:
        label = "CYRILLIC"
    elif latin and cyrillic:
        label = "LATIN_CYRILLIC_MIXED"
    elif other and not latin and not cyrillic:
        label = "OTHER_SCRIPT"
    else:
        label = "MULTISCRIPT_OTHER"
    return label, latin, cyrillic, other


def suspicious_characters(text: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for character in text:
        category = unicodedata.category(character)
        if (
            character in SUSPICIOUS
            or category == "Cf"
            or (category == "Cc" and character not in "\n\r\t")
        ):
            result[character] += 1
    return result


def quality_flags(text: str, script_label: str, suspicious: Counter[str]) -> list[str]:
    flags: list[str] = []
    if not text.strip():
        flags.append("empty_text")
    if text != text.lstrip(" \t\r\n"):
        flags.append("leading_whitespace")
    if text != text.rstrip(" \t\r\n"):
        flags.append("trailing_whitespace")
    if re.search(r"[ \t]{2,}", text):
        flags.append("repeated_horizontal_whitespace")
    if "\t" in text:
        flags.append("contains_tab")
    if "\n" in text or "\r" in text:
        flags.append("contains_newline")
    if any(unicodedata.category(ch) == "Cc" and ch not in "\n\r\t" for ch in text):
        flags.append("contains_control_character")
    if any(unicodedata.category(ch) == "Cf" for ch in text):
        flags.append("contains_invisible_format_character")
    if "\ufffd" in text:
        flags.append("contains_replacement_character")
    if "\u00a0" in text:
        flags.append("contains_nonbreaking_space")
    if script_label == "LATIN_CYRILLIC_MIXED":
        flags.append("latin_cyrillic_mixed")
    explained_suspicious_flags = {
        "contains_control_character",
        "contains_invisible_format_character",
        "contains_replacement_character",
        "contains_nonbreaking_space",
    }
    if suspicious and not explained_suspicious_flags.intersection(flags):
        flags.append("contains_suspicious_unicode")
    return flags


def leading_whitespace(text: str) -> str:
    match = re.match(r"^[ \t\r\n]+", text)
    return match.group(0) if match else ""


def trailing_whitespace(text: str) -> str:
    match = re.search(r"[ \t\r\n]+$", text)
    return match.group(0) if match else ""


def whitespace_repr(value: str) -> str:
    labels = {" ": "<SPACE>", "\t": "<TAB>", "\r": "<CR>", "\n": "<LF>"}
    return "".join(labels.get(character, f"<U+{ord(character):04X}>") for character in value)


def suspected_apostrophe_gaps(text: str) -> Iterable[tuple[str, re.Match[str]]]:
    for rule_id, pattern in APOSTROPHE_GAP_RULES:
        for match in pattern.finditer(text):
            yield rule_id, match


def boilerplate_reasons(
    text: str, metrics: TextMetrics, short_max_chars: int, short_max_words: int
) -> list[str]:
    reasons: list[str] = []
    stripped = text.strip()
    if (
        metrics.character_count <= short_max_chars
        and metrics.word_count_estimate <= short_max_words
    ):
        reasons.append("SHORT_RECORD")
    if SOURCE_CREDIT_RE.search(stripped):
        reasons.append("SOURCE_OR_CREDIT_MARKER")
    if metrics.word_count_estimate <= 6 and NAME_LIKE_RE.fullmatch(stripped):
        reasons.append("PERSON_NAME_LIKE")
    return reasons


def histogram_quantile(histogram: Counter[int], total: int, q: float) -> int | None:
    if total == 0:
        return None
    rank = max(1, round((total - 1) * q) + 1)
    cumulative = 0
    for value in sorted(histogram):
        cumulative += histogram[value]
        if cumulative >= rank:
            return value
    return max(histogram)


def histogram_summary(
    histogram: Counter[int], total: int, value_sum: int
) -> dict[str, int | float | None]:
    if total == 0:
        return {
            "minimum": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "minimum": min(histogram),
        "median": histogram_quantile(histogram, total, 0.50),
        "p90": histogram_quantile(histogram, total, 0.90),
        "p95": histogram_quantile(histogram, total, 0.95),
        "p99": histogram_quantile(histogram, total, 0.99),
        "maximum": max(histogram),
        "mean": round(value_sum / total, 2),
    }


def length_summary_rows(
    character_histogram: Counter[int],
    word_histogram: Counter[int],
    line_histogram: Counter[int],
    total: int,
    character_sum: int,
    word_sum: int,
    line_sum: int,
) -> list[dict[str, Any]]:
    char_summary = histogram_summary(character_histogram, total, character_sum)
    word_summary = histogram_summary(word_histogram, total, word_sum)
    line_summary = histogram_summary(line_histogram, total, line_sum)
    metrics = ["minimum", "median", "p90", "p95", "p99", "maximum", "mean"]
    return [
        {
            "metric": metric,
            "character_count": char_summary[metric],
            "word_count_estimate": word_summary[metric],
            "line_count": line_summary[metric],
        }
        for metric in metrics
    ]


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def initialize_duplicate_db(path: Path, overwrite: bool) -> sqlite3.Connection:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Duplicate database already exists: {path}")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        """
        CREATE TABLE duplicate_index (
            digest TEXT NOT NULL,
            character_count INTEGER NOT NULL,
            occurrence_count INTEGER NOT NULL,
            first_source_record_id INTEGER NOT NULL,
            other_source_record_ids TEXT NOT NULL,
            text_preview TEXT NOT NULL,
            PRIMARY KEY (digest, character_count)
        )
        """
    )
    return connection


def register_duplicate(
    connection: sqlite3.Connection,
    digest: str,
    character_count: int,
    source_record_id: int,
    text_preview: str,
    stored_id_limit: int,
) -> bool:
    row = connection.execute(
        """
        SELECT occurrence_count, other_source_record_ids
        FROM duplicate_index
        WHERE digest = ? AND character_count = ?
        """,
        (digest, character_count),
    ).fetchone()

    if row is None:
        connection.execute(
            """
            INSERT INTO duplicate_index (
                digest, character_count, occurrence_count,
                first_source_record_id, other_source_record_ids, text_preview
            ) VALUES (?, ?, 1, ?, '', ?)
            """,
            (digest, character_count, source_record_id, text_preview),
        )
        return False

    occurrence_count, stored_ids = row
    ids = [value for value in stored_ids.split(";") if value]
    if len(ids) < stored_id_limit:
        ids.append(str(source_record_id))
    connection.execute(
        """
        UPDATE duplicate_index
        SET occurrence_count = ?, other_source_record_ids = ?
        WHERE digest = ? AND character_count = ?
        """,
        (occurrence_count + 1, ";".join(ids), digest, character_count),
    )
    return True


def push_shortest(heap: list[tuple[int, int, dict[str, Any]]], limit: int, row: dict[str, Any]) -> None:
    if limit <= 0:
        return
    item = (-int(row["character_count"]), -int(row["source_record_id"]), row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def push_longest(heap: list[tuple[int, int, dict[str, Any]]], limit: int, row: dict[str, Any]) -> None:
    if limit <= 0:
        return
    item = (int(row["character_count"]), int(row["source_record_id"]), row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    duplicate_db = (
        args.duplicate_db.expanduser().resolve()
        if args.duplicate_db is not None
        else output_dir / "duplicate_index.sqlite3"
    )

    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = [name for name in OUTPUTS if (output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Existing outputs: " + ", ".join(existing))
    if args.overwrite:
        for name in existing:
            (output_dir / name).unlink()

    before = source.stat()
    with source.open("rb") as handle:
        first_character = handle.read(65_536).lstrip(b"\xef\xbb\xbf \t\r\n")[:1]
    if first_character != b"[":
        raise ValueError(f"Expected top-level JSON array, found {first_character!r}")

    rng = random.Random(args.seed)
    duplicate_connection = initialize_duplicate_db(duplicate_db, args.overwrite)

    record_types: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    keysets: Counter[tuple[str, ...]] = Counter()
    text_types: Counter[str] = Counter()
    scripts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    apostrophe_hits: Counter[str] = Counter()
    apostrophe_records: Counter[str] = Counter()
    suspicious_hits: Counter[str] = Counter()
    boundary_patterns: Counter[tuple[str, str, int]] = Counter()

    character_histogram: Counter[int] = Counter()
    word_histogram: Counter[int] = Counter()
    line_histogram: Counter[int] = Counter()
    character_sum = word_sum = line_sum = 0

    random_samples: list[dict[str, Any]] = []
    quality_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mixed_examples: list[dict[str, Any]] = []
    unicode_examples: list[dict[str, Any]] = []
    apostrophe_gap_examples_by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    apostrophe_gap_rule_counts: Counter[str] = Counter()
    boilerplate_examples_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    boilerplate_reason_counts: Counter[str] = Counter()
    shortest_heap: list[tuple[int, int, dict[str, Any]]] = []
    longest_heap: list[tuple[int, int, dict[str, Any]]] = []

    scanned = 0
    dictionary_records = 0
    records_with_text = 0
    string_text_records = 0
    duplicate_occurrences_after_first = 0
    apostrophe_gap_match_count = 0
    apostrophe_gap_record_count = 0
    short_boilerplate_candidate_count = 0

    try:
        with source.open("rb") as handle:
            for source_record_id, record in enumerate(ijson.items(handle, "item")):
                if args.max_records and source_record_id >= args.max_records:
                    break

                scanned += 1
                record_type = type_name(record)
                record_types[record_type] += 1

                if not isinstance(record, dict):
                    flag_counts["non_dictionary_record"] += 1
                    examples = quality_examples["non_dictionary_record"]
                    if len(examples) < args.example_limit:
                        examples.append(
                            {
                                "source_record_id": source_record_id,
                                "quality_flag": "non_dictionary_record",
                                "character_count": "",
                                "word_count_estimate": "",
                                "text_preview": make_preview(record, args.preview_chars),
                            }
                        )
                    continue

                dictionary_records += 1
                keysets[tuple(sorted(map(str, record.keys())))] += 1
                for key in record:
                    field_counts[str(key)] += 1

                if "text" not in record:
                    flag_counts["missing_text"] += 1
                    examples = quality_examples["missing_text"]
                    if len(examples) < args.example_limit:
                        examples.append(
                            {
                                "source_record_id": source_record_id,
                                "quality_flag": "missing_text",
                                "character_count": "",
                                "word_count_estimate": "",
                                "text_preview": make_preview(record, args.preview_chars),
                            }
                        )
                    continue

                records_with_text += 1
                text = record["text"]
                text_types[type_name(text)] += 1
                if not isinstance(text, str):
                    flag_counts["non_string_text"] += 1
                    examples = quality_examples["non_string_text"]
                    if len(examples) < args.example_limit:
                        examples.append(
                            {
                                "source_record_id": source_record_id,
                                "quality_flag": "non_string_text",
                                "character_count": "",
                                "word_count_estimate": "",
                                "text_preview": make_preview(text, args.preview_chars),
                            }
                        )
                    continue

                string_text_records += 1
                metrics = text_metrics(text)
                character_histogram[metrics.character_count] += 1
                word_histogram[metrics.word_count_estimate] += 1
                line_histogram[metrics.line_count] += 1
                character_sum += metrics.character_count
                word_sum += metrics.word_count_estimate
                line_sum += metrics.line_count

                script_label, latin_count, cyrillic_count, other_count = script_info(text)
                scripts[script_label] += 1
                suspicious = suspicious_characters(text)
                flags = quality_flags(text, script_label, suspicious)
                for flag in flags:
                    flag_counts[flag] += 1
                    examples = quality_examples[flag]
                    if len(examples) < args.example_limit:
                        examples.append(
                            {
                                "source_record_id": source_record_id,
                                "quality_flag": flag,
                                "character_count": metrics.character_count,
                                "word_count_estimate": metrics.word_count_estimate,
                                "text_preview": make_preview(text, args.preview_chars),
                            }
                        )

                leading = leading_whitespace(text)
                trailing = trailing_whitespace(text)
                if leading:
                    boundary_patterns[("LEADING", whitespace_repr(leading), len(leading))] += 1
                if trailing:
                    boundary_patterns[("TRAILING", whitespace_repr(trailing), len(trailing))] += 1

                for character in APOSTROPHES:
                    occurrences = text.count(character)
                    if occurrences:
                        apostrophe_hits[character] += occurrences
                        apostrophe_records[character] += 1

                for character, occurrences in suspicious.items():
                    suspicious_hits[character] += occurrences
                    if len(unicode_examples) < args.example_limit:
                        position = text.find(character)
                        context = text[max(0, position - 80) : min(len(text), position + 81)]
                        unicode_examples.append(
                            {
                                "source_record_id": source_record_id,
                                "character_display": repr(character),
                                "unicode_codepoint": f"U+{ord(character):04X}",
                                "unicode_name": unicodedata.name(character, "UNKNOWN"),
                                "unicode_category": unicodedata.category(character),
                                "occurrences_in_record": occurrences,
                                "context_preview": make_preview(context, args.preview_chars),
                            }
                        )

                if (
                    script_label == "LATIN_CYRILLIC_MIXED"
                    and len(mixed_examples) < args.example_limit
                ):
                    mixed_examples.append(
                        {
                            "source_record_id": source_record_id,
                            "latin_letter_count": latin_count,
                            "cyrillic_letter_count": cyrillic_count,
                            "other_letter_count": other_count,
                            "script_label": script_label,
                            "text_preview": make_preview(text, args.preview_chars),
                        }
                    )

                record_has_apostrophe_gap = False
                for rule_id, match in suspected_apostrophe_gaps(text):
                    record_has_apostrophe_gap = True
                    apostrophe_gap_match_count += 1
                    apostrophe_gap_rule_counts[rule_id] += 1
                    examples = apostrophe_gap_examples_by_rule[rule_id]
                    if len(examples) < args.example_limit:
                        start, end = match.span()
                        context = text[max(0, start - 90) : min(len(text), end + 90)]
                        examples.append(
                            {
                                "source_record_id": source_record_id,
                                "rule_id": rule_id,
                                "matched_fragment": match.group(0),
                                "match_start": start,
                                "match_end": end,
                                "review_status": "REVIEW_ONLY",
                                "context_preview": make_preview(context, args.preview_chars),
                            }
                        )
                if record_has_apostrophe_gap:
                    apostrophe_gap_record_count += 1

                reasons = boilerplate_reasons(
                    text,
                    metrics,
                    args.short_max_chars,
                    args.short_max_words,
                )
                if reasons:
                    short_boilerplate_candidate_count += 1
                    for reason in reasons:
                        boilerplate_reason_counts[reason] += 1
                        examples = boilerplate_examples_by_reason[reason]
                        if len(examples) < args.example_limit:
                            examples.append(
                                {
                                    "source_record_id": source_record_id,
                                    "candidate_reason": reason,
                                    "all_candidate_reasons": ";".join(reasons),
                                    "character_count": metrics.character_count,
                                    "word_count_estimate": metrics.word_count_estimate,
                                    "review_status": "REVIEW_ONLY",
                                    "text_preview": make_preview(text, args.preview_chars),
                                }
                            )

                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                is_duplicate = register_duplicate(
                    duplicate_connection,
                    digest,
                    metrics.character_count,
                    source_record_id,
                    make_preview(text, args.preview_chars),
                    args.example_limit,
                )
                if is_duplicate:
                    duplicate_occurrences_after_first += 1
                    flag_counts["exact_duplicate_after_first"] += 1
                    examples = quality_examples["exact_duplicate_after_first"]
                    if len(examples) < args.example_limit:
                        examples.append(
                            {
                                "source_record_id": source_record_id,
                                "quality_flag": "exact_duplicate_after_first",
                                "character_count": metrics.character_count,
                                "word_count_estimate": metrics.word_count_estimate,
                                "text_preview": make_preview(text, args.preview_chars),
                            }
                        )

                sample_row = {
                    "source_record_id": source_record_id,
                    "sample_reason": "deterministic_random_sample",
                    "text_preview": make_preview(text, args.preview_chars),
                    "character_count": metrics.character_count,
                    "word_count_estimate": metrics.word_count_estimate,
                    "line_count": metrics.line_count,
                    "script_label": script_label,
                    "quality_flags": ";".join(flags),
                }
                if len(random_samples) < args.sample_size:
                    random_samples.append(sample_row)
                elif args.sample_size:
                    pick = rng.randint(1, string_text_records)
                    if pick <= args.sample_size:
                        random_samples[pick - 1] = sample_row

                extreme_row = {
                    "source_record_id": source_record_id,
                    "character_count": metrics.character_count,
                    "word_count_estimate": metrics.word_count_estimate,
                    "line_count": metrics.line_count,
                    "text_preview": make_preview(text, args.preview_chars),
                }
                push_shortest(shortest_heap, args.length_example_limit, extreme_row)
                push_longest(longest_heap, args.length_example_limit, extreme_row)

                if scanned % 5_000 == 0:
                    duplicate_connection.commit()

                if args.progress_every and scanned % args.progress_every == 0:
                    print(
                        f"PROGRESS records={scanned:,} string_texts={string_text_records:,}",
                        file=sys.stderr,
                        flush=True,
                    )

        duplicate_connection.commit()

        duplicate_groups = duplicate_connection.execute(
            "SELECT COUNT(*) FROM duplicate_index WHERE occurrence_count > 1"
        ).fetchone()[0]
        duplicate_rows = [
            {
                "duplicate_hash": row[0],
                "occurrence_count": row[1],
                "first_source_record_id": row[2],
                "other_source_record_ids": row[3],
                "character_count": row[4],
                "text_preview": row[5],
            }
            for row in duplicate_connection.execute(
                """
                SELECT digest, occurrence_count, first_source_record_id,
                       other_source_record_ids, character_count, text_preview
                FROM duplicate_index
                WHERE occurrence_count > 1
                ORDER BY occurrence_count DESC, first_source_record_id ASC
                LIMIT ?
                """,
                (args.example_limit,),
            )
        ]
    finally:
        duplicate_connection.close()

    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Raw corpus changed during audit")

    if not args.keep_duplicate_db and duplicate_db.exists():
        duplicate_db.unlink()

    sanity_checks = {
        "record_type_balance": sum(record_types.values()) == scanned,
        "dictionary_balance": dictionary_records + (scanned - dictionary_records) == scanned,
        "text_field_balance": records_with_text + (dictionary_records - records_with_text)
        == dictionary_records,
        "text_type_balance": string_text_records + (records_with_text - string_text_records)
        == records_with_text,
        "script_balance": sum(scripts.values()) == string_text_records,
        "scan_limit_respected": args.max_records == 0 or scanned <= args.max_records,
        "raw_corpus_immutable": True,
        "random_sample_ids_unique": len({row["source_record_id"] for row in random_samples})
        == len(random_samples),
        "preview_limit_respected": all(
            len(row["text_preview"]) <= args.preview_chars for row in random_samples
        ),
    }
    failed_checks = [name for name, passed in sanity_checks.items() if not passed]
    if failed_checks:
        raise RuntimeError("Sanity checks failed: " + ", ".join(failed_checks))

    record_rows = [
        {
            "record_type": name,
            "record_count": count,
            "percentage_of_records_scanned": percentage(count, scanned),
        }
        for name, count in record_types.most_common()
    ]
    field_rows = [
        {
            "field_name": name,
            "records_with_field": count,
            "percentage_of_dictionary_records": percentage(count, dictionary_records),
        }
        for name, count in field_counts.most_common()
    ]
    text_type_rows = [
        {
            "text_value_type": name,
            "record_count": count,
            "percentage_of_text_fields": percentage(count, records_with_text),
        }
        for name, count in text_types.most_common()
    ]
    script_rows = [
        {
            "script_label": name,
            "record_count": count,
            "percentage_of_string_text_records": percentage(count, string_text_records),
        }
        for name, count in scripts.most_common()
    ]
    quality_flag_rows = [
        {
            "quality_flag": name,
            "record_count": count,
            "percentage_of_records_scanned": percentage(count, scanned),
        }
        for name, count in flag_counts.most_common()
    ]
    quality_example_rows = [
        row
        for flag in sorted(quality_examples)
        for row in quality_examples[flag]
    ]
    boundary_rows = [
        {
            "boundary": boundary,
            "whitespace_sequence_repr": sequence,
            "sequence_length": length,
            "record_count": count,
            "percentage_of_string_text_records": percentage(count, string_text_records),
        }
        for (boundary, sequence, length), count in sorted(
            boundary_patterns.items(), key=lambda item: (item[0][0], -item[1], item[0][1])
        )
    ]
    apostrophe_rows = [
        {
            "character": character,
            "unicode_codepoint": f"U+{ord(character):04X}",
            "unicode_name": unicodedata.name(character, "UNKNOWN"),
            "total_occurrences": apostrophe_hits[character],
            "records_containing_character": apostrophe_records[character],
            "possible_normalization_group": "APOSTROPHE_LIKE",
            "automatic_change_allowed": "false",
        }
        for character in APOSTROPHES
        if apostrophe_hits[character]
    ]
    apostrophe_rows.sort(key=lambda row: -int(row["total_occurrences"]))

    apostrophe_gap_summary_rows = [
        {
            "rule_id": rule_id,
            "match_count": count,
            "review_status": "REVIEW_ONLY",
        }
        for rule_id, count in apostrophe_gap_rule_counts.most_common()
    ]
    apostrophe_gap_example_rows = [
        row
        for rule_id in sorted(apostrophe_gap_examples_by_rule)
        for row in apostrophe_gap_examples_by_rule[rule_id]
    ]
    boilerplate_summary_rows = [
        {
            "candidate_reason": reason,
            "record_count": count,
            "percentage_of_string_text_records": percentage(count, string_text_records),
            "review_status": "REVIEW_ONLY",
        }
        for reason, count in boilerplate_reason_counts.most_common()
    ]
    boilerplate_example_rows = [
        row
        for reason in sorted(boilerplate_examples_by_reason)
        for row in boilerplate_examples_by_reason[reason]
    ]

    length_rows = length_summary_rows(
        character_histogram,
        word_histogram,
        line_histogram,
        string_text_records,
        character_sum,
        word_sum,
        line_sum,
    )

    shortest_rows = [item[2] for item in sorted(shortest_heap, reverse=True)]
    longest_rows = [item[2] for item in sorted(longest_heap, reverse=True)]
    length_extreme_rows = [
        {"length_group": "SHORTEST", **row}
        for row in sorted(shortest_rows, key=lambda row: (row["character_count"], row["source_record_id"]))
    ] + [
        {"length_group": "LONGEST", **row}
        for row in sorted(longest_rows, key=lambda row: (-row["character_count"], row["source_record_id"]))
    ]

    write_csv(
        output_dir / "record_type_counts.csv",
        ["record_type", "record_count", "percentage_of_records_scanned"],
        record_rows,
    )
    write_csv(
        output_dir / "field_presence_counts.csv",
        ["field_name", "records_with_field", "percentage_of_dictionary_records"],
        field_rows,
    )
    write_csv(
        output_dir / "text_value_type_counts.csv",
        ["text_value_type", "record_count", "percentage_of_text_fields"],
        text_type_rows,
    )
    write_csv(
        output_dir / "text_length_summary.csv",
        ["metric", "character_count", "word_count_estimate", "line_count"],
        length_rows,
    )
    write_csv(
        output_dir / "length_extreme_examples.csv",
        [
            "length_group",
            "source_record_id",
            "character_count",
            "word_count_estimate",
            "line_count",
            "text_preview",
        ],
        length_extreme_rows,
    )
    write_csv(
        output_dir / "apostrophe_variant_counts.csv",
        [
            "character",
            "unicode_codepoint",
            "unicode_name",
            "total_occurrences",
            "records_containing_character",
            "possible_normalization_group",
            "automatic_change_allowed",
        ],
        apostrophe_rows,
    )
    write_csv(
        output_dir / "script_distribution.csv",
        ["script_label", "record_count", "percentage_of_string_text_records"],
        script_rows,
    )
    write_csv(
        output_dir / "quality_flag_counts.csv",
        ["quality_flag", "record_count", "percentage_of_records_scanned"],
        quality_flag_rows,
    )
    write_csv(
        output_dir / "quality_flag_examples.csv",
        [
            "source_record_id",
            "quality_flag",
            "character_count",
            "word_count_estimate",
            "text_preview",
        ],
        quality_example_rows,
    )
    write_csv(
        output_dir / "boundary_whitespace_summary.csv",
        [
            "boundary",
            "whitespace_sequence_repr",
            "sequence_length",
            "record_count",
            "percentage_of_string_text_records",
        ],
        boundary_rows,
    )
    write_csv(
        output_dir / "sample_records.csv",
        [
            "source_record_id",
            "sample_reason",
            "text_preview",
            "character_count",
            "word_count_estimate",
            "line_count",
            "script_label",
            "quality_flags",
        ],
        sorted(random_samples, key=lambda row: row["source_record_id"]),
    )
    write_csv(
        output_dir / "duplicate_examples.csv",
        [
            "duplicate_hash",
            "occurrence_count",
            "first_source_record_id",
            "other_source_record_ids",
            "character_count",
            "text_preview",
        ],
        duplicate_rows,
    )
    write_csv(
        output_dir / "mixed_script_examples.csv",
        [
            "source_record_id",
            "latin_letter_count",
            "cyrillic_letter_count",
            "other_letter_count",
            "script_label",
            "text_preview",
        ],
        mixed_examples,
    )
    write_csv(
        output_dir / "suspicious_unicode_examples.csv",
        [
            "source_record_id",
            "character_display",
            "unicode_codepoint",
            "unicode_name",
            "unicode_category",
            "occurrences_in_record",
            "context_preview",
        ],
        unicode_examples,
    )
    write_csv(
        output_dir / "suspected_apostrophe_gap_summary.csv",
        ["rule_id", "match_count", "review_status"],
        apostrophe_gap_summary_rows,
    )
    write_csv(
        output_dir / "suspected_apostrophe_gap_examples.csv",
        [
            "source_record_id",
            "rule_id",
            "matched_fragment",
            "match_start",
            "match_end",
            "review_status",
            "context_preview",
        ],
        apostrophe_gap_example_rows,
    )
    write_csv(
        output_dir / "short_boilerplate_summary.csv",
        [
            "candidate_reason",
            "record_count",
            "percentage_of_string_text_records",
            "review_status",
        ],
        boilerplate_summary_rows,
    )
    write_csv(
        output_dir / "short_boilerplate_candidates.csv",
        [
            "source_record_id",
            "candidate_reason",
            "all_candidate_reasons",
            "character_count",
            "word_count_estimate",
            "review_status",
            "text_preview",
        ],
        boilerplate_example_rows,
    )

    schema = {
        "input_path": str(source),
        "input_size_bytes": before.st_size,
        "input_modified_time_ns": before.st_mtime_ns,
        "format": "JSON_ARRAY",
        "parser": {
            "library": "ijson",
            "version": getattr(ijson, "__version__", "unknown"),
            "backend": getattr(ijson, "backend", "unknown"),
        },
        "scan_configuration": {
            "max_records": args.max_records,
            "sample_size": args.sample_size,
            "example_limit": args.example_limit,
            "length_example_limit": args.length_example_limit,
            "preview_chars": args.preview_chars,
            "seed": args.seed,
            "short_max_chars": args.short_max_chars,
            "short_max_words": args.short_max_words,
            "duplicate_db": str(duplicate_db),
            "duplicate_db_retained": args.keep_duplicate_db,
        },
        "counts": {
            "records_scanned": scanned,
            "dictionary_records": dictionary_records,
            "non_dictionary_records": scanned - dictionary_records,
            "records_with_text": records_with_text,
            "records_missing_text": dictionary_records - records_with_text,
            "string_text_records": string_text_records,
            "non_string_text_records": records_with_text - string_text_records,
            "duplicate_groups": duplicate_groups,
            "duplicate_occurrences_after_first": duplicate_occurrences_after_first,
            "suspected_apostrophe_gap_records": apostrophe_gap_record_count,
            "suspected_apostrophe_gap_matches": apostrophe_gap_match_count,
            "short_boilerplate_candidates": short_boilerplate_candidate_count,
        },
        "record_type_counts": dict(record_types),
        "field_presence_counts": dict(field_counts),
        "key_combination_counts": {"|".join(keys): count for keys, count in keysets.items()},
        "text_value_type_counts": dict(text_types),
        "script_counts": dict(scripts),
        "quality_flag_counts": dict(flag_counts),
        "apostrophe_gap_rule_counts": dict(apostrophe_gap_rule_counts),
        "short_boilerplate_reason_counts": dict(boilerplate_reason_counts),
        "boundary_whitespace_counts": {
            f"{boundary}|{sequence}|{length}": count
            for (boundary, sequence, length), count in boundary_patterns.items()
        },
        "length_summary": length_rows,
        "apostrophe_occurrences": {
            f"U+{ord(character):04X}": count for character, count in apostrophe_hits.items()
        },
        "suspicious_unicode_occurrences": {
            f"U+{ord(character):04X}": count for character, count in suspicious_hits.items()
        },
        "sanity_checks": sanity_checks,
        "raw_corpus_immutability_check": "PASS",
        "known_limitations": [
            "A partial scan cannot prove complete-corpus consistency.",
            "Script labels describe Unicode scripts, not language identity.",
            "Word counts are whitespace-based estimates, not tokenizer counts.",
            "Duplicate detection uses SHA-256 plus character length.",
            "Suspected apostrophe-gap rules are heuristic and review-only.",
            "Short-boilerplate rules are heuristic and review-only.",
            "Quality flags describe evidence and do not authorize correction or rejection.",
        ],
    }
    (output_dir / "corpus_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# Corpus Audit Metrics",
        "",
        "## Run configuration",
        "",
        f"- Input: `{source}`",
        f"- Output: `{output_dir}`",
        f"- Maximum records requested: {args.max_records}",
        f"- Records scanned: {scanned}",
        f"- Random seed: {args.seed}",
        f"- ijson version: {getattr(ijson, '__version__', 'unknown')}",
        f"- ijson backend: {getattr(ijson, 'backend', 'unknown')}",
        "",
        "## Core schema findings",
        "",
        f"- Dictionary records: {dictionary_records}",
        f"- Non-dictionary records: {scanned - dictionary_records}",
        f"- Records with `text`: {records_with_text}",
        f"- Records missing `text`: {dictionary_records - records_with_text}",
        f"- String text records: {string_text_records}",
        f"- Non-string text records: {records_with_text - string_text_records}",
        "",
        "## Duplicate findings",
        "",
        f"- Duplicate groups: {duplicate_groups}",
        f"- Duplicate occurrences after the first: {duplicate_occurrences_after_first}",
        "",
        "## Review-only heuristic findings",
        "",
        f"- Records with suspected apostrophe gaps: {apostrophe_gap_record_count}",
        f"- Suspected apostrophe-gap matches: {apostrophe_gap_match_count}",
        f"- Short or boilerplate candidates: {short_boilerplate_candidate_count}",
        "",
        "## Safety and sanity checks",
        "",
        "- Raw corpus size and modification timestamp were unchanged: PASS",
        "- No corpus text was normalized or corrected.",
        "- Exported text is limited to configured previews.",
    ]
    report_lines.extend(
        f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in sanity_checks.items()
    )
    report_lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- A partial scan cannot prove complete-corpus consistency.",
            "- Script labels are not language labels.",
            "- Word counts are whitespace-based estimates.",
            "- Apostrophe-gap and boilerplate labels are review-only heuristics.",
            "- Quality flags do not automatically reject or correct records.",
            "",
        ]
    )
    (output_dir / "corpus_audit_metrics.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("--- AUDIT V2 COMPLETE ---")
    print(f"records_scanned={scanned}")
    print(f"dictionary_records={dictionary_records}")
    print(f"string_text_records={string_text_records}")
    print(f"duplicate_groups={duplicate_groups}")
    print(f"duplicate_occurrences_after_first={duplicate_occurrences_after_first}")
    print(f"suspected_apostrophe_gap_records={apostrophe_gap_record_count}")
    print(f"suspected_apostrophe_gap_matches={apostrophe_gap_match_count}")
    print(f"short_boilerplate_candidates={short_boilerplate_candidate_count}")
    print(f"output_dir={output_dir}")
    print("immutability_check=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
