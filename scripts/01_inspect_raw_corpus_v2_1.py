#!/usr/bin/env python3
"""
Audit a large top-level JSON array corpus without modifying the source file.

Uzbek Contextual Spelling Correction Lab v2
Phase 1: Raw Corpus Investigation
Audit script v2.1

Key properties
--------------
- Streams the top-level JSON array with ijson.
- Opens the corpus read-only.
- Uses SQLite for exact duplicate tracking.
- Uses the JSON-array position as source_record_id (zero-based).
- Separates high-confidence and ambiguous apostrophe-gap heuristics.
- Aggregates apostrophe-gap signals by corpus-position block.
- Separates phrase-containing records from standalone source/credit records.
- Assigns review-only short-record role heuristics.
- Writes machine-readable CSV/JSON outputs and a compact Markdown report.
- Never edits, normalizes, or corrects corpus text.

Synthetic/gold or replacement decisions are outside this audit script.
All heuristic outputs are review-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import itertools
import json
import math
import os
import re
import shutil
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import ijson
except ImportError:
    ijson = None  # type: ignore[assignment]


SCRIPT_VERSION = "2.1"
SOURCE_RECORD_ID_BASE = 0

DEFAULT_INPUT = Path("/mnt/uzbekvoice_storage/text_data/normalized.json")
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/uzbekvoice_storage/projects/"
    "uzbek-contextual-spelling-correction-lab-v2/outputs/"
    "01_corpus_audit/scan_200k_v2_1"
)

APOSTROPHE_VARIANTS: Mapping[str, str] = {
    "U+0027_APOSTROPHE": "'",
    "U+0060_GRAVE_ACCENT": "`",
    "U+00B4_ACUTE_ACCENT": "´",
    "U+02BB_MODIFIER_LETTER_TURNED_COMMA": "ʻ",
    "U+02BC_MODIFIER_LETTER_APOSTROPHE": "ʼ",
    "U+2018_LEFT_SINGLE_QUOTATION_MARK": "‘",
    "U+2019_RIGHT_SINGLE_QUOTATION_MARK": "’",
}

APOSTROPHE_CLASS = r"'`´ʻʼ‘’"
LATIN_LETTER_CLASS = r"A-Za-zÀ-ɏḀ-ỿ"

# These rules intentionally identify review signals only. They do not modify text.
# Horizontal spacing is used so a match does not silently cross a line boundary.
APOSTROPHE_GAP_RULES: Mapping[str, re.Pattern[str]] = {
    "BO_L_PATTERN": re.compile(
        rf"(?<![{LATIN_LETTER_CLASS}])bo[ \t]+l(?=[{LATIN_LETTER_CLASS}])",
        re.IGNORECASE,
    ),
    "E_LON_PATTERN": re.compile(
        rf"(?<![{LATIN_LETTER_CLASS}])e[ \t]+lon(?=[{LATIN_LETTER_CLASS}]|s?\b)",
        re.IGNORECASE,
    ),
    "KO_R_PATTERN": re.compile(
        rf"(?<![{LATIN_LETTER_CLASS}])ko[ \t]+r(?=[{LATIN_LETTER_CLASS}])",
        re.IGNORECASE,
    ),
    "MA_LUM_PATTERN": re.compile(
        rf"(?<![{LATIN_LETTER_CLASS}])ma[ \t]+lum(?=[{LATIN_LETTER_CLASS}]|\b)",
        re.IGNORECASE,
    ),
    "O_T_PATTERN": re.compile(
        rf"(?<![{LATIN_LETTER_CLASS}])o[ \t]+t(?=[{LATIN_LETTER_CLASS}])",
        re.IGNORECASE,
    ),
    "QO_L_PATTERN": re.compile(
        rf"(?<![{LATIN_LETTER_CLASS}])qo[ \t]+l(?=[{LATIN_LETTER_CLASS}])",
        re.IGNORECASE,
    ),
    "TO_G_R_PATTERN": re.compile(
        rf"(?<![{LATIN_LETTER_CLASS}])to[ \t]+g[ \t]+r(?=[{LATIN_LETTER_CLASS}])",
        re.IGNORECASE,
    ),
    "O_Q_PATTERN": re.compile(
        rf"(?<![{LATIN_LETTER_CLASS}])o[ \t]+q(?=[{LATIN_LETTER_CLASS}])",
        re.IGNORECASE,
    ),
    "O_Z_PATTERN": re.compile(
        rf"(?<![{LATIN_LETTER_CLASS}])o[ \t]+z(?=[{LATIN_LETTER_CLASS}])",
        re.IGNORECASE,
    ),
}

HIGH_CONFIDENCE_RULE_IDS = frozenset(
    {
        "BO_L_PATTERN",
        "E_LON_PATTERN",
        "KO_R_PATTERN",
        "MA_LUM_PATTERN",
        "O_T_PATTERN",
        "QO_L_PATTERN",
        "TO_G_R_PATTERN",
    }
)
AMBIGUOUS_RULE_IDS = frozenset({"O_Q_PATTERN", "O_Z_PATTERN"})

if set(APOSTROPHE_GAP_RULES) != HIGH_CONFIDENCE_RULE_IDS | AMBIGUOUS_RULE_IDS:
    raise RuntimeError("Apostrophe-gap rule grouping is inconsistent.")

REPEATED_HORIZONTAL_WHITESPACE_RE = re.compile(r"[ \t]{2,}")
LATIN_RE = re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]")
CYRILLIC_RE = re.compile(r"[\u0400-\u052F\u2DE0-\u2DFF\uA640-\uA69F]")
TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?…][\"'’”)\]]*\s*$")

SUSPICIOUS_UNICODE_CHARS: Mapping[str, str] = {
    "\u00A0": "U+00A0_NO_BREAK_SPACE",
    "\u00AD": "U+00AD_SOFT_HYPHEN",
    "\u200B": "U+200B_ZERO_WIDTH_SPACE",
    "\u200C": "U+200C_ZERO_WIDTH_NON_JOINER",
    "\u200D": "U+200D_ZERO_WIDTH_JOINER",
    "\u202F": "U+202F_NARROW_NO_BREAK_SPACE",
    "\u2060": "U+2060_WORD_JOINER",
    "\uFEFF": "U+FEFF_ZERO_WIDTH_NO_BREAK_SPACE_OR_BOM",
    "\uFFFD": "U+FFFD_REPLACEMENT_CHARACTER",
}

# Broad phrase detection: the record contains a source/credit signal somewhere.
SOURCE_CREDIT_MARKERS: Mapping[str, re.Pattern[str]] = {
    "MANBA": re.compile(r"\bmanba\s*:", re.IGNORECASE),
    "MATBUOT_XIZMATI": re.compile(r"\bmatbuot\s+xizmati\b", re.IGNORECASE),
    "AXBOROT_XIZMATI": re.compile(r"\baxborot\s+xizmati\b", re.IGNORECASE),
    "XABAR_BERDI": re.compile(r"\bxabar\s+berdi\b", re.IGNORECASE),
    "XALQ_SOZI": re.compile(
        rf"\bxalq\s+so[{APOSTROPHE_CLASS}]zi\b", re.IGNORECASE
    ),
    "UZA": re.compile(
        rf"\b(?:uza|o[{APOSTROPHE_CLASS}]za)\b", re.IGNORECASE
    ),
    "MILLIY_AXBOROT_AGENTLIGI": re.compile(
        r"\bmilliy\s+axborot\s+agentligi\b", re.IGNORECASE
    ),
    "GAZETA_OR_GAZETASI": re.compile(
        r"\bgazeta(?:si)?\b", re.IGNORECASE
    ),
    "ARXIV_SURAT": re.compile(
        r"\barxiv\s+(?:surat|fotosurat|foto)\b", re.IGNORECASE
    ),
    "FOTO_CREDIT": re.compile(
        r"\b(?:foto|surat)\s*:", re.IGNORECASE
    ),
    "MUALLIF": re.compile(r"\bmuallif\s*:", re.IGNORECASE),
    "MUXBIR": re.compile(r"\b(?:muxbirimiz|muxbir)\b", re.IGNORECASE),
}

STANDALONE_SERVICE_RE = re.compile(
    r"^(?:(?:[A-ZА-ЯЁ][^\n]{0,80})\s+)?"
    r"(?:matbuot\s+xizmati|axborot\s+xizmati)\.?$",
    re.IGNORECASE,
)
STANDALONE_ARCHIVE_RE = re.compile(
    r"^(?:arxiv\s+)?(?:surat|fotosurat|foto)(?:\s*:\s*[^\n]{1,100})?\.?$",
    re.IGNORECASE,
)
STANDALONE_MANBA_RE = re.compile(r"^manba\s*:\s*[^\n]{1,180}$", re.IGNORECASE)
STANDALONE_ENDING_SOURCE_RE = re.compile(
    rf"(?:\bxalq\s+so[{APOSTROPHE_CLASS}]zi"
    rf"|\b(?:uza|o[{APOSTROPHE_CLASS}]za)"
    rf"|\b[^\n]{{1,80}}\s+gazeta(?:si)?)\.?$",
    re.IGNORECASE,
)

NAME_TOKEN_RE = re.compile(
    rf"^[{LATIN_LETTER_CLASS}][{LATIN_LETTER_CLASS}{APOSTROPHE_CLASS}\-]*$"
)


@dataclass(frozen=True)
class FileFingerprint:
    size_bytes: int
    mtime_ns: int
    inode: int
    device: int
    first_chunk_sha256: str
    last_chunk_sha256: str


class DeterministicSampler:
    """Keep rows with the lowest stable hash scores for each category."""

    def __init__(self, limit_per_category: int, seed: str) -> None:
        if limit_per_category < 1:
            raise ValueError("limit_per_category must be positive")
        self.limit = limit_per_category
        self.seed = seed
        self._heaps: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)

    def add(self, category: str, source_record_id: int, row: Mapping[str, Any]) -> None:
        payload = f"{self.seed}|{category}|{source_record_id}".encode("utf-8")
        score = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
        serialised = json.dumps(dict(row), ensure_ascii=False, sort_keys=True)
        item = (-score, -source_record_id, serialised)
        heap = self._heaps[category]
        if len(heap) < self.limit:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    def categories(self) -> List[str]:
        return sorted(self._heaps)

    def rows(self, category: str) -> List[Dict[str, Any]]:
        result: List[Tuple[int, int, Dict[str, Any]]] = []
        for neg_score, neg_record_id, serialised in self._heaps.get(category, []):
            result.append(
                (-neg_score, -neg_record_id, json.loads(serialised))
            )
        result.sort(key=lambda item: (item[0], item[1]))
        return [row for _, _, row in result]


class LengthHistogram:
    """Exact streaming length statistics without storing one value per record."""

    def __init__(self) -> None:
        self.counts: Counter[int] = Counter()
        self.n = 0
        self.total = 0
        self.minimum: Optional[int] = None
        self.maximum: Optional[int] = None

    def add(self, value: int) -> None:
        self.counts[value] += 1
        self.n += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def _value_at_sorted_index(self, index: int) -> int:
        if not 0 <= index < self.n:
            raise IndexError(index)
        cumulative = 0
        for value in sorted(self.counts):
            cumulative += self.counts[value]
            if cumulative > index:
                return value
        raise RuntimeError("Length histogram is internally inconsistent.")

    def percentile(self, q: float) -> Optional[float]:
        if self.n == 0:
            return None
        if not 0.0 <= q <= 1.0:
            raise ValueError(q)
        position = (self.n - 1) * q
        lower_index = math.floor(position)
        upper_index = math.ceil(position)
        lower = self._value_at_sorted_index(lower_index)
        upper = self._value_at_sorted_index(upper_index)
        if lower_index == upper_index:
            return float(lower)
        fraction = position - lower_index
        return lower + (upper - lower) * fraction

    def summary(self) -> Dict[str, Any]:
        if self.n == 0:
            return {
                "count": 0,
                "minimum": None,
                "median": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "maximum": None,
                "mean": None,
            }
        return {
            "count": self.n,
            "minimum": self.minimum,
            "median": self.percentile(0.50),
            "p90": self.percentile(0.90),
            "p95": self.percentile(0.95),
            "p99": self.percentile(0.99),
            "maximum": self.maximum,
            "mean": self.total / self.n,
        }


class ExtremeRecords:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._shortest: List[Tuple[int, int, str]] = []
        self._longest: List[Tuple[int, int, str]] = []

    def add(self, source_record_id: int, character_count: int, text_preview: str) -> None:
        shortest_item = (-character_count, -source_record_id, text_preview)
        if len(self._shortest) < self.limit:
            heapq.heappush(self._shortest, shortest_item)
        elif shortest_item > self._shortest[0]:
            heapq.heapreplace(self._shortest, shortest_item)

        longest_item = (character_count, source_record_id, text_preview)
        if len(self._longest) < self.limit:
            heapq.heappush(self._longest, longest_item)
        elif longest_item > self._longest[0]:
            heapq.heapreplace(self._longest, longest_item)

    def shortest_rows(self) -> List[Dict[str, Any]]:
        rows = [
            {
                "source_record_id": -neg_id,
                "character_count": -neg_length,
                "text_preview": preview,
            }
            for neg_length, neg_id, preview in self._shortest
        ]
        return sorted(rows, key=lambda row: (row["character_count"], row["source_record_id"]))

    def longest_rows(self) -> List[Dict[str, Any]]:
        rows = [
            {
                "source_record_id": record_id,
                "character_count": length,
                "text_preview": preview,
            }
            for length, record_id, preview in self._longest
        ]
        return sorted(
            rows,
            key=lambda row: (-row["character_count"], row["source_record_id"]),
        )


class BlockAccumulator:
    FIELDNAMES = [
        "block_start_record_id",
        "block_end_record_id",
        "records_scanned_in_block",
        "records_with_high_confidence_gap",
        "records_with_ambiguous_gap",
        "high_confidence_match_count",
        "ambiguous_match_count",
        "percentage_with_high_confidence_gap",
        "percentage_with_ambiguous_gap",
    ]

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.current_start: Optional[int] = None
        self.current_end: Optional[int] = None
        self.records = 0
        self.high_records = 0
        self.ambiguous_records = 0
        self.high_matches = 0
        self.ambiguous_matches = 0

    def add(
        self,
        source_record_id: int,
        high_match_count: int,
        ambiguous_match_count: int,
    ) -> Optional[Dict[str, Any]]:
        expected_start = (source_record_id // self.block_size) * self.block_size
        completed = None
        if self.current_start is None:
            self.current_start = expected_start
        elif expected_start != self.current_start:
            completed = self.finalize()
            self.current_start = expected_start

        self.current_end = source_record_id
        self.records += 1
        if high_match_count > 0:
            self.high_records += 1
        if ambiguous_match_count > 0:
            self.ambiguous_records += 1
        self.high_matches += high_match_count
        self.ambiguous_matches += ambiguous_match_count
        return completed

    def finalize(self) -> Optional[Dict[str, Any]]:
        if self.records == 0 or self.current_start is None or self.current_end is None:
            return None
        row = {
            "block_start_record_id": self.current_start,
            "block_end_record_id": self.current_end,
            "records_scanned_in_block": self.records,
            "records_with_high_confidence_gap": self.high_records,
            "records_with_ambiguous_gap": self.ambiguous_records,
            "high_confidence_match_count": self.high_matches,
            "ambiguous_match_count": self.ambiguous_matches,
            "percentage_with_high_confidence_gap": round(
                100.0 * self.high_records / self.records, 6
            ),
            "percentage_with_ambiguous_gap": round(
                100.0 * self.ambiguous_records / self.records, 6
            ),
        }
        self.current_start = None
        self.current_end = None
        self.records = 0
        self.high_records = 0
        self.ambiguous_records = 0
        self.high_matches = 0
        self.ambiguous_matches = 0
        return row


class ExactDuplicateTracker:
    """
    SQLite-backed exact duplicate tracker.

    SHA-256 narrows the lookup. The original full text is stored and compared,
    so a digest collision cannot be counted as an exact duplicate.
    """

    def __init__(self, db_path: Path, commit_every: int = 10_000) -> None:
        self.db_path = db_path
        self.commit_every = commit_every
        self.pending = 0
        self.connection = sqlite3.connect(str(db_path))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-65536")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS seen_texts (
                text_id INTEGER PRIMARY KEY,
                text_hash BLOB NOT NULL,
                character_count INTEGER NOT NULL,
                text_value TEXT NOT NULL,
                first_record_id INTEGER NOT NULL,
                occurrence_count INTEGER NOT NULL DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_seen_texts_hash
            ON seen_texts(text_hash);

            CREATE TABLE IF NOT EXISTS duplicate_occurrences (
                text_id INTEGER NOT NULL,
                source_record_id INTEGER NOT NULL,
                FOREIGN KEY(text_id) REFERENCES seen_texts(text_id)
            );

            CREATE INDEX IF NOT EXISTS idx_duplicate_occurrences_text_id
            ON duplicate_occurrences(text_id);
            """
        )
        self.connection.commit()

    def observe(self, source_record_id: int, text: str) -> Optional[Dict[str, Any]]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        rows = self.connection.execute(
            """
            SELECT text_id, text_value, first_record_id, occurrence_count
            FROM seen_texts
            WHERE text_hash = ?
            """,
            (digest,),
        ).fetchall()

        for text_id, stored_text, first_record_id, occurrence_count in rows:
            if stored_text == text:
                new_count = occurrence_count + 1
                self.connection.execute(
                    """
                    UPDATE seen_texts
                    SET occurrence_count = ?
                    WHERE text_id = ?
                    """,
                    (new_count, text_id),
                )
                self.connection.execute(
                    """
                    INSERT INTO duplicate_occurrences(text_id, source_record_id)
                    VALUES (?, ?)
                    """,
                    (text_id, source_record_id),
                )
                self._maybe_commit()
                return {
                    "text_id": text_id,
                    "first_record_id": first_record_id,
                    "occurrence_count": new_count,
                }

        cursor = self.connection.execute(
            """
            INSERT INTO seen_texts(
                text_hash,
                character_count,
                text_value,
                first_record_id,
                occurrence_count
            )
            VALUES (?, ?, ?, ?, 1)
            """,
            (digest, len(text), text, source_record_id),
        )
        self._maybe_commit()
        return None

    def _maybe_commit(self) -> None:
        self.pending += 1
        if self.pending >= self.commit_every:
            self.connection.commit()
            self.pending = 0

    def summary(self) -> Dict[str, int]:
        self.connection.commit()
        row = self.connection.execute(
            """
            SELECT
                COUNT(*) AS duplicate_groups,
                COALESCE(SUM(occurrence_count - 1), 0)
                    AS duplicate_occurrences_after_first
            FROM seen_texts
            WHERE occurrence_count > 1
            """
        ).fetchone()
        assert row is not None
        return {
            "duplicate_groups": int(row[0]),
            "duplicate_occurrences_after_first": int(row[1]),
        }

    def export_groups(self, destination: Path, preview_chars: int) -> None:
        self.connection.commit()
        query = """
            SELECT
                text_id,
                first_record_id,
                occurrence_count,
                character_count,
                hex(text_hash),
                text_value
            FROM seen_texts
            WHERE occurrence_count > 1
            ORDER BY occurrence_count DESC, first_record_id ASC
        """
        fieldnames = [
            "duplicate_group_id",
            "first_record_id",
            "occurrence_count",
            "duplicate_occurrences_after_first",
            "character_count",
            "sha256",
            "duplicate_record_ids_after_first",
            "text_preview",
        ]
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for (
                text_id,
                first_record_id,
                occurrence_count,
                character_count,
                digest_hex,
                text_value,
            ) in self.connection.execute(query):
                duplicate_ids = [
                    str(row[0])
                    for row in self.connection.execute(
                        """
                        SELECT source_record_id
                        FROM duplicate_occurrences
                        WHERE text_id = ?
                        ORDER BY source_record_id
                        """,
                        (text_id,),
                    )
                ]
                writer.writerow(
                    {
                        "duplicate_group_id": text_id,
                        "first_record_id": first_record_id,
                        "occurrence_count": occurrence_count,
                        "duplicate_occurrences_after_first": occurrence_count - 1,
                        "character_count": character_count,
                        "sha256": digest_hex.lower(),
                        "duplicate_record_ids_after_first": ";".join(duplicate_ids),
                        "text_preview": visible_preview(text_value, preview_chars),
                    }
                )

    def close(self) -> None:
        self.connection.commit()
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream-audit the Uzbek corpus without modifying it. "
            "source_record_id is the zero-based JSON-array position."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Top-level JSON-array corpus path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"New output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=200_000,
        help="Maximum array items to scan. Use 0 for the complete corpus.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=10_000,
        help="Corpus-position block size. Default: 10000.",
    )
    parser.add_argument(
        "--short-record-max-chars",
        type=int,
        default=120,
        help="Maximum trimmed character length for short-role heuristics.",
    )
    parser.add_argument(
        "--standalone-credit-max-chars",
        type=int,
        default=220,
        help="Maximum trimmed length for standalone source/credit heuristics.",
    )
    parser.add_argument(
        "--example-limit-per-category",
        type=int,
        default=12,
        help="Deterministic example count retained for each category.",
    )
    parser.add_argument(
        "--extreme-record-limit",
        type=int,
        default=25,
        help="Number of shortest and longest record examples.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=320,
        help="Maximum characters in human-readable previews.",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=90,
        help="Context characters retained around an apostrophe-gap match.",
    )
    parser.add_argument(
        "--sample-seed",
        default="uzbek-contextual-spelling-correction-lab-v2-audit-v2.1",
        help="Stable string used by deterministic example sampling.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Write progress to stderr every N records. Use 0 to disable.",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help=(
            "Explicitly delete and recreate --output-dir if it already exists. "
            "Without this flag, a non-empty output directory causes an error."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_records < 0:
        raise SystemExit("ERROR: --max-records must be 0 or a positive integer.")
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

    if not input_path.is_file():
        raise SystemExit(f"ERROR: input corpus does not exist or is not a file: {input_path}")
    if not os.access(input_path, os.R_OK):
        raise SystemExit(f"ERROR: input corpus is not readable: {input_path}")
    if input_path == output_path:
        raise SystemExit("ERROR: input path and output directory cannot be identical.")
    if output_path == Path("/"):
        raise SystemExit("ERROR: refusing to use filesystem root as output directory.")
    if input_path in output_path.parents:
        # Output below a file is impossible, but retain a clear safety check.
        raise SystemExit("ERROR: output directory cannot be placed inside the input file path.")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise SystemExit(f"ERROR: output path exists and is not a directory: {path}")
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


def sha256_chunk(path: Path, offset: int, length: int) -> str:
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(length)
    return hashlib.sha256(data).hexdigest()


def fingerprint_file(path: Path, chunk_bytes: int = 1_048_576) -> FileFingerprint:
    stat = path.stat()
    first_length = min(chunk_bytes, stat.st_size)
    last_offset = max(0, stat.st_size - chunk_bytes)
    return FileFingerprint(
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        inode=stat.st_ino,
        device=stat.st_dev,
        first_chunk_sha256=sha256_chunk(path, 0, first_length),
        last_chunk_sha256=sha256_chunk(path, last_offset, stat.st_size - last_offset),
    )


def inspect_json_boundaries(path: Path, chunk_bytes: int = 1_048_576) -> Dict[str, Any]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        first_chunk = handle.read(min(chunk_bytes, size))
        handle.seek(max(0, size - chunk_bytes))
        last_chunk = handle.read(min(chunk_bytes, size))

    has_bom = first_chunk.startswith(b"\xef\xbb\xbf")
    first_decoded = first_chunk.decode("utf-8-sig")
    last_decoded = last_chunk.decode("utf-8")
    first_non_whitespace = next((ch for ch in first_decoded if not ch.isspace()), None)
    last_non_whitespace = next(
        (ch for ch in reversed(last_decoded) if not ch.isspace()), None
    )
    if first_non_whitespace != "[" or last_non_whitespace != "]":
        raise SystemExit(
            "ERROR: input boundaries do not look like a top-level JSON array: "
            f"first={first_non_whitespace!r}, last={last_non_whitespace!r}"
        )
    return {
        "utf8_bom_present": has_bom,
        "first_meaningful_character": first_non_whitespace,
        "last_meaningful_character": last_non_whitespace,
        "boundary_probe_bytes": chunk_bytes,
    }


def visible_preview(text: str, limit: int) -> str:
    clipped = text[:limit]
    clipped = (
        clipped.replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    if len(text) > limit:
        clipped += "…"
    return clipped


def match_context(text: str, start: int, end: int, context_chars: int) -> str:
    left = max(0, start - context_chars)
    right = min(len(text), end + context_chars)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(text) else ""
    return prefix + visible_preview(text[left:right], right - left) + suffix


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_sampler_categories(
    sampler: DeterministicSampler,
    output_dir: Path,
    prefix: str,
) -> None:
    for category in sampler.categories():
        rows = sampler.rows(category)
        if not rows:
            continue
        all_fields: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    all_fields.append(key)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", category)
        write_csv(output_dir / f"{prefix}{safe_name}.csv", all_fields, rows)


def classify_script(text: str) -> str:
    has_latin = LATIN_RE.search(text) is not None
    has_cyrillic = CYRILLIC_RE.search(text) is not None
    if has_latin and has_cyrillic:
        return "MIXED_LATIN_CYRILLIC"
    if has_latin:
        return "LATIN"
    if has_cyrillic:
        return "CYRILLIC"
    if any(ch.isalpha() for ch in text):
        return "OTHER_ALPHABETIC"
    return "NO_ALPHABETIC_CHARACTERS"


def suspicious_unicode_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for char, label in SUSPICIOUS_UNICODE_CHARS.items():
        count = text.count(char)
        if count:
            counts[label] += count

    for char in text:
        code = ord(char)
        if code < 32 and char not in "\n\r\t":
            counts[f"U+{code:04X}_CONTROL_CHARACTER"] += 1
        elif 0x7F <= code <= 0x9F:
            counts[f"U+{code:04X}_CONTROL_CHARACTER"] += 1
    return counts


def find_apostrophe_gap_matches(text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    high: List[Dict[str, Any]] = []
    ambiguous: List[Dict[str, Any]] = []
    for rule_id, pattern in APOSTROPHE_GAP_RULES.items():
        target = high if rule_id in HIGH_CONFIDENCE_RULE_IDS else ambiguous
        for match in pattern.finditer(text):
            target.append(
                {
                    "rule_id": rule_id,
                    "matched_fragment": match.group(0),
                    "match_start": match.start(),
                    "match_end": match.end(),
                }
            )
    high.sort(key=lambda item: (item["match_start"], item["rule_id"]))
    ambiguous.sort(key=lambda item: (item["match_start"], item["rule_id"]))
    return high, ambiguous


def matched_source_credit_markers(text: str) -> List[str]:
    return [
        marker_id
        for marker_id, pattern in SOURCE_CREDIT_MARKERS.items()
        if pattern.search(text)
    ]


def tokenise_for_name_heuristic(text: str) -> List[str]:
    cleaned = text.strip().rstrip(".!?…")
    return [token for token in re.split(r"\s+", cleaned) if token]


def is_name_like(text: str) -> bool:
    tokens = tokenise_for_name_heuristic(text)
    if not 2 <= len(tokens) <= 4:
        return False
    for token in tokens:
        if not NAME_TOKEN_RE.fullmatch(token):
            return False
        letters = "".join(ch for ch in token if ch.isalpha())
        if not letters:
            return False
        if not (letters.isupper() or (letters[0].isupper() and letters[1:].islower())):
            return False
    return True


def is_standalone_source_or_credit(
    text: str,
    matched_markers: Sequence[str],
    max_chars: int,
    name_like: bool,
) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > max_chars:
        return False
    if stripped.count("\n") > 2:
        return False

    if STANDALONE_MANBA_RE.fullmatch(stripped):
        return True
    if STANDALONE_ARCHIVE_RE.fullmatch(stripped):
        return True
    if STANDALONE_SERVICE_RE.fullmatch(stripped):
        return True
    if STANDALONE_ENDING_SOURCE_RE.search(stripped):
        return True
    if "FOTO_CREDIT" in matched_markers or "MUALLIF" in matched_markers:
        return True

    # A name-only line ending in punctuation can be a byline, but this remains
    # intentionally review-only because names and ordinary short noun phrases overlap.
    if name_like and TERMINAL_PUNCTUATION_RE.search(stripped):
        return True
    return False


def short_record_role(
    text: str,
    max_chars: int,
    standalone_credit: bool,
    name_like: bool,
) -> Optional[str]:
    stripped = text.strip()
    if len(stripped) > max_chars:
        return None
    if standalone_credit:
        return "SHORT_CREDIT_LIKE"
    if name_like and TERMINAL_PUNCTUATION_RE.search(stripped):
        return "SHORT_CREDIT_LIKE"
    if TERMINAL_PUNCTUATION_RE.search(stripped):
        return "SHORT_SENTENCE_LIKE"
    return "SHORT_FRAGMENT_OR_UNKNOWN"


def review_status_for_gap(high_count: int, ambiguous_count: int) -> str:
    if high_count and ambiguous_count:
        return "MIXED_CONFIDENCE_REVIEW_REQUIRED"
    if high_count:
        return "HIGH_CONFIDENCE_REVIEW_REQUIRED"
    if ambiguous_count:
        return "AMBIGUOUS_REVIEW_REQUIRED"
    raise ValueError("review_status requested for a record without a gap signal")


def percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 6) if denominator else 0.0


def markdown_table(rows: Sequence[Tuple[str, Any]]) -> str:
    lines = ["| Metric | Value |", "|---|---:|"]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def format_stat(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def build_markdown_report(summary: Mapping[str, Any]) -> str:
    length = summary["text_length_characters"]
    quality = summary["quality_flags"]
    apostrophe = summary["apostrophe_gap"]
    source_credit = summary["source_credit"]
    short_roles = summary["short_record_roles"]
    duplicates = summary["duplicates"]

    lines = [
        "# Phase 1 Raw Corpus Audit — v2.1",
        "",
        "This report is descriptive only. No corpus text was corrected, normalised, "
        "deleted, or automatically approved.",
        "",
        "## Run identity",
        "",
        markdown_table(
            [
                ("script_version", summary["script_version"]),
                ("input_path", summary["input"]["path"]),
                ("records_scanned", summary["records_scanned"]),
                ("source_record_id_base", summary["source_record_id_base"]),
                ("max_records_requested", summary["max_records_requested"]),
                ("block_size", summary["block_size"]),
                ("elapsed_seconds", format_stat(summary["elapsed_seconds"])),
                ("immutability_check", summary["immutability_check"]),
            ]
        ),
        "",
        "## Structural findings",
        "",
        markdown_table(
            [
                ("dictionary_records", summary["record_structure"]["dictionary_records"]),
                ("non_dictionary_records", summary["record_structure"]["non_dictionary_records"]),
                ("records_with_text_field", summary["record_structure"]["records_with_text_field"]),
                ("missing_text_field", summary["record_structure"]["missing_text_field"]),
                ("string_text_records", summary["record_structure"]["string_text_records"]),
                ("non_string_text_records", summary["record_structure"]["non_string_text_records"]),
                ("empty_or_whitespace_texts", summary["record_structure"]["empty_or_whitespace_texts"]),
            ]
        ),
        "",
        "## Text length in characters",
        "",
        markdown_table([(key, format_stat(value)) for key, value in length.items()]),
        "",
        "## Duplicate findings",
        "",
        markdown_table(
            [
                ("duplicate_groups", duplicates["duplicate_groups"]),
                (
                    "duplicate_occurrences_after_first",
                    duplicates["duplicate_occurrences_after_first"],
                ),
                ("tracking_mode", duplicates["tracking_mode"]),
            ]
        ),
        "",
        "## Boundary and whitespace findings",
        "",
        markdown_table([(key, value) for key, value in quality.items()]),
        "",
        "## Apostrophe-gap review signals",
        "",
        markdown_table(
            [
                (
                    "records_with_high_confidence_gap",
                    apostrophe["records_with_high_confidence_gap"],
                ),
                (
                    "records_with_ambiguous_gap",
                    apostrophe["records_with_ambiguous_gap"],
                ),
                (
                    "records_with_any_gap_signal",
                    apostrophe["records_with_any_gap_signal"],
                ),
                (
                    "high_confidence_match_count",
                    apostrophe["high_confidence_match_count"],
                ),
                ("ambiguous_match_count", apostrophe["ambiguous_match_count"]),
            ]
        ),
        "",
        "High-confidence and ambiguous labels are heuristic review queues, not "
        "automatic corrections.",
        "",
        "## Source and credit heuristics",
        "",
        markdown_table([(key, value) for key, value in source_credit.items()]),
        "",
        "`CONTAINS_SOURCE_OR_CREDIT_PHRASE` is broad. "
        "`STANDALONE_SOURCE_OR_CREDIT` is stricter and length/structure constrained. "
        "Both remain review-only.",
        "",
        "## Short-record role heuristics",
        "",
        markdown_table([(key, value) for key, value in short_roles.items()]),
        "",
        "Short-role labels do not delete, exclude, or approve records.",
        "",
        "## Script distribution",
        "",
        markdown_table(
            [(key, value) for key, value in summary["script_distribution"].items()]
        ),
        "",
        "## Main output files",
        "",
    ]
    for name, description in summary["output_manifest"].items():
        lines.append(f"- `{name}` — {description}")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- The audit establishes corpus properties and review signals.",
            "- It does not establish that flagged text is definitely wrong.",
            "- It does not establish that unflagged text is correct.",
            "- It does not modify the immutable raw corpus.",
            "- It does not enable automatic replacement.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    if ijson is None:
        raise SystemExit(
            "ERROR: ijson is required to run the audit. Install the pinned project "
            "dependency first: python3 -m pip install -r requirements.txt"
        )
    validate_args(args)
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prepare_output_dir(output_dir, args.overwrite_output_dir)

    started_at_epoch = time.time()
    pre_fingerprint = fingerprint_file(input_path)
    boundary_info = inspect_json_boundaries(input_path)

    counters: Counter[str] = Counter()
    quality_flag_counts: Counter[str] = Counter()
    script_counts: Counter[str] = Counter()
    apostrophe_variant_counts: Counter[str] = Counter()
    suspicious_unicode_total: Counter[str] = Counter()
    apostrophe_rule_counts: Counter[str] = Counter()
    source_marker_counts: Counter[str] = Counter()
    short_role_counts: Counter[str] = Counter()

    length_histogram = LengthHistogram()
    extremes = ExtremeRecords(args.extreme_record_limit)
    block_accumulator = BlockAccumulator(args.block_size)

    quality_sampler = DeterministicSampler(
        args.example_limit_per_category, args.sample_seed + "|quality"
    )
    category_sampler = DeterministicSampler(
        args.example_limit_per_category, args.sample_seed + "|category"
    )
    apostrophe_sampler = DeterministicSampler(
        args.example_limit_per_category, args.sample_seed + "|apostrophe"
    )

    db_path = output_dir / "duplicate_index.sqlite3"
    duplicate_tracker = ExactDuplicateTracker(db_path)

    apostrophe_records_path = output_dir / "apostrophe_gap_records.csv"
    block_distribution_path = output_dir / "apostrophe_gap_block_distribution.csv"
    source_credit_records_path = output_dir / "source_credit_records.csv"
    short_record_roles_path = output_dir / "short_record_roles.csv"

    apostrophe_record_fields = [
        "source_record_id",
        "character_count",
        "high_confidence_match_count",
        "ambiguous_match_count",
        "high_confidence_rule_ids",
        "ambiguous_rule_ids",
        "matches_per_1000_characters",
        "review_status",
        "text_preview",
    ]
    source_credit_fields = [
        "source_record_id",
        "character_count",
        "contains_source_or_credit_phrase",
        "standalone_source_or_credit",
        "matched_marker_ids",
        "person_name_like",
        "review_status",
        "text_preview",
    ]
    short_role_fields = [
        "source_record_id",
        "character_count",
        "short_record_role",
        "standalone_source_or_credit",
        "person_name_like",
        "review_status",
        "text_preview",
    ]

    block_rows_written = 0
    block_validation_totals: Counter[str] = Counter()

    def account_block(row: Mapping[str, Any]) -> None:
        nonlocal block_rows_written
        block_rows_written += 1
        block_validation_totals["records_scanned_in_block"] += int(
            row["records_scanned_in_block"]
        )
        block_validation_totals["records_with_high_confidence_gap"] += int(
            row["records_with_high_confidence_gap"]
        )
        block_validation_totals["records_with_ambiguous_gap"] += int(
            row["records_with_ambiguous_gap"]
        )
        block_validation_totals["high_confidence_match_count"] += int(
            row["high_confidence_match_count"]
        )
        block_validation_totals["ambiguous_match_count"] += int(
            row["ambiguous_match_count"]
        )

    try:
        with (
            apostrophe_records_path.open("w", encoding="utf-8", newline="") as apostrophe_handle,
            block_distribution_path.open("w", encoding="utf-8", newline="") as block_handle,
            source_credit_records_path.open("w", encoding="utf-8", newline="") as source_handle,
            short_record_roles_path.open("w", encoding="utf-8", newline="") as short_handle,
            input_path.open("rb") as corpus_handle,
        ):
            apostrophe_writer = csv.DictWriter(
                apostrophe_handle, fieldnames=apostrophe_record_fields
            )
            block_writer = csv.DictWriter(
                block_handle, fieldnames=BlockAccumulator.FIELDNAMES
            )
            source_writer = csv.DictWriter(source_handle, fieldnames=source_credit_fields)
            short_writer = csv.DictWriter(short_handle, fieldnames=short_role_fields)

            apostrophe_writer.writeheader()
            block_writer.writeheader()
            source_writer.writeheader()
            short_writer.writeheader()

            items: Iterator[Any] = ijson.items(corpus_handle, "item")
            selected_items: Iterable[Any]
            if args.max_records:
                selected_items = itertools.islice(items, args.max_records)
            else:
                selected_items = items

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
                            "missing_text_field",
                            source_record_id,
                            {
                                "source_record_id": source_record_id,
                                "record_type": type(record).__name__,
                                "record_preview": visible_preview(
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
                            text_preview = visible_preview(text, args.preview_chars)
                            length_histogram.add(character_count)
                            extremes.add(source_record_id, character_count, text_preview)

                            if not text.strip():
                                counters["empty_or_whitespace_texts"] += 1
                                quality_flag_counts["empty_or_whitespace_text"] += 1
                                quality_sampler.add(
                                    "empty_or_whitespace_text",
                                    source_record_id,
                                    {
                                        "source_record_id": source_record_id,
                                        "character_count": character_count,
                                        "text_preview": text_preview,
                                    },
                                )

                            has_leading = bool(text) and text[0].isspace()
                            has_trailing = bool(text) and text[-1].isspace()
                            repeated_horizontal = (
                                REPEATED_HORIZONTAL_WHITESPACE_RE.search(text) is not None
                            )

                            if has_leading:
                                quality_flag_counts["leading_whitespace"] += 1
                                quality_sampler.add(
                                    "leading_whitespace",
                                    source_record_id,
                                    {
                                        "source_record_id": source_record_id,
                                        "character_count": character_count,
                                        "text_preview": text_preview,
                                    },
                                )
                            if has_trailing:
                                quality_flag_counts["trailing_whitespace"] += 1
                                quality_sampler.add(
                                    "trailing_whitespace",
                                    source_record_id,
                                    {
                                        "source_record_id": source_record_id,
                                        "character_count": character_count,
                                        "text_preview": text_preview,
                                    },
                                )
                            if repeated_horizontal:
                                quality_flag_counts["repeated_horizontal_whitespace"] += 1
                                quality_sampler.add(
                                    "repeated_horizontal_whitespace",
                                    source_record_id,
                                    {
                                        "source_record_id": source_record_id,
                                        "character_count": character_count,
                                        "text_preview": text_preview,
                                    },
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

                            script_label = classify_script(text)
                            script_counts[script_label] += 1

                            for variant_id, char in APOSTROPHE_VARIANTS.items():
                                count = text.count(char)
                                if count:
                                    apostrophe_variant_counts[variant_id] += count

                            unicode_counts = suspicious_unicode_counts(text)
                            if unicode_counts:
                                quality_flag_counts["suspicious_unicode_record"] += 1
                                suspicious_unicode_total.update(unicode_counts)
                                quality_sampler.add(
                                    "suspicious_unicode_record",
                                    source_record_id,
                                    {
                                        "source_record_id": source_record_id,
                                        "character_count": character_count,
                                        "suspicious_codepoints": ";".join(
                                            sorted(unicode_counts)
                                        ),
                                        "suspicious_occurrence_count": sum(
                                            unicode_counts.values()
                                        ),
                                        "text_preview": text_preview,
                                    },
                                )

                            duplicate_info = duplicate_tracker.observe(
                                source_record_id, text
                            )
                            if duplicate_info is not None:
                                quality_flag_counts["exact_duplicate_after_first"] += 1
                                quality_sampler.add(
                                    "exact_duplicate_after_first",
                                    source_record_id,
                                    {
                                        "source_record_id": source_record_id,
                                        "first_record_id": duplicate_info[
                                            "first_record_id"
                                        ],
                                        "occurrence_count_including_current": duplicate_info[
                                            "occurrence_count"
                                        ],
                                        "character_count": character_count,
                                        "text_preview": text_preview,
                                    },
                                )

                            high_matches, ambiguous_matches = find_apostrophe_gap_matches(
                                text
                            )
                            high_count = len(high_matches)
                            ambiguous_count = len(ambiguous_matches)

                            if high_count:
                                counters["records_with_high_confidence_gap"] += 1
                            if ambiguous_count:
                                counters["records_with_ambiguous_gap"] += 1
                            if high_count or ambiguous_count:
                                counters["records_with_any_gap_signal"] += 1
                                high_rule_ids = sorted(
                                    {match["rule_id"] for match in high_matches}
                                )
                                ambiguous_rule_ids = sorted(
                                    {match["rule_id"] for match in ambiguous_matches}
                                )
                                total_gap_matches = high_count + ambiguous_count
                                apostrophe_writer.writerow(
                                    {
                                        "source_record_id": source_record_id,
                                        "character_count": character_count,
                                        "high_confidence_match_count": high_count,
                                        "ambiguous_match_count": ambiguous_count,
                                        "high_confidence_rule_ids": ";".join(
                                            high_rule_ids
                                        ),
                                        "ambiguous_rule_ids": ";".join(
                                            ambiguous_rule_ids
                                        ),
                                        "matches_per_1000_characters": round(
                                            1000.0
                                            * total_gap_matches
                                            / max(1, character_count),
                                            6,
                                        ),
                                        "review_status": review_status_for_gap(
                                            high_count, ambiguous_count
                                        ),
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
                                            "context_preview": match_context(
                                                text,
                                                match["match_start"],
                                                match["match_end"],
                                                args.context_chars,
                                            ),
                                        },
                                    )

                            name_like = is_name_like(text)
                            if name_like:
                                quality_flag_counts["person_name_like"] += 1
                                quality_sampler.add(
                                    "person_name_like",
                                    source_record_id,
                                    {
                                        "source_record_id": source_record_id,
                                        "character_count": character_count,
                                        "text_preview": text_preview,
                                    },
                                )

                            marker_ids = matched_source_credit_markers(text)
                            for marker_id in marker_ids:
                                source_marker_counts[marker_id] += 1
                            contains_source_credit = bool(marker_ids)
                            standalone_source_credit = is_standalone_source_or_credit(
                                text,
                                marker_ids,
                                args.standalone_credit_max_chars,
                                name_like,
                            )
                            if contains_source_credit:
                                counters[
                                    "contains_source_or_credit_phrase"
                                ] += 1
                            if standalone_source_credit:
                                counters[
                                    "standalone_source_or_credit"
                                ] += 1

                            if contains_source_credit or standalone_source_credit:
                                source_row = {
                                    "source_record_id": source_record_id,
                                    "character_count": character_count,
                                    "contains_source_or_credit_phrase": str(
                                        contains_source_credit
                                    ).lower(),
                                    "standalone_source_or_credit": str(
                                        standalone_source_credit
                                    ).lower(),
                                    "matched_marker_ids": ";".join(marker_ids),
                                    "person_name_like": str(name_like).lower(),
                                    "review_status": "REVIEW_REQUIRED",
                                    "text_preview": text_preview,
                                }
                                source_writer.writerow(source_row)
                                if contains_source_credit:
                                    category_sampler.add(
                                        "CONTAINS_SOURCE_OR_CREDIT_PHRASE",
                                        source_record_id,
                                        source_row,
                                    )
                                if standalone_source_credit:
                                    category_sampler.add(
                                        "STANDALONE_SOURCE_OR_CREDIT",
                                        source_record_id,
                                        source_row,
                                    )

                            role = short_record_role(
                                text,
                                args.short_record_max_chars,
                                standalone_source_credit,
                                name_like,
                            )
                            if role is not None:
                                short_role_counts[role] += 1
                                short_row = {
                                    "source_record_id": source_record_id,
                                    "character_count": character_count,
                                    "short_record_role": role,
                                    "standalone_source_or_credit": str(
                                        standalone_source_credit
                                    ).lower(),
                                    "person_name_like": str(name_like).lower(),
                                    "review_status": "REVIEW_REQUIRED",
                                    "text_preview": text_preview,
                                }
                                short_writer.writerow(short_row)
                                category_sampler.add(
                                    role, source_record_id, short_row
                                )
                        else:
                            counters["non_string_text_records"] += 1
                            quality_flag_counts["non_string_text"] += 1
                            quality_sampler.add(
                                "non_string_text",
                                source_record_id,
                                {
                                    "source_record_id": source_record_id,
                                    "text_value_type": type(text).__name__,
                                    "record_preview": visible_preview(
                                        json.dumps(record, ensure_ascii=False),
                                        args.preview_chars,
                                    ),
                                },
                            )
                else:
                    counters["non_dictionary_records"] += 1
                    quality_flag_counts["non_dictionary_record"] += 1
                    quality_sampler.add(
                        "non_dictionary_record",
                        source_record_id,
                        {
                            "source_record_id": source_record_id,
                            "record_type": type(record).__name__,
                            "record_preview": visible_preview(
                                json.dumps(record, ensure_ascii=False),
                                args.preview_chars,
                            ),
                        },
                    )

                completed_block = block_accumulator.add(
                    source_record_id,
                    len(high_matches),
                    len(ambiguous_matches),
                )
                if completed_block is not None:
                    block_writer.writerow(completed_block)
                    account_block(completed_block)

                if (
                    args.progress_every
                    and counters["records_scanned"] % args.progress_every == 0
                ):
                    elapsed = time.time() - started_at_epoch
                    rate = counters["records_scanned"] / max(elapsed, 1e-9)
                    print(
                        (
                            f"[audit v{SCRIPT_VERSION}] "
                            f"records={counters['records_scanned']:,} "
                            f"rate={rate:,.1f} records/s "
                            f"high_gap_records="
                            f"{counters['records_with_high_confidence_gap']:,} "
                            f"ambiguous_gap_records="
                            f"{counters['records_with_ambiguous_gap']:,}"
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

            final_block = block_accumulator.finalize()
            if final_block is not None:
                block_writer.writerow(final_block)
                account_block(final_block)

        duplicate_summary = duplicate_tracker.summary()
        duplicate_tracker.export_groups(
            output_dir / "duplicate_groups.csv", args.preview_chars
        )
        duplicate_tracker.close()

        quality_flag_counts["trailing_whitespace"] = quality_flag_counts.get(
            "trailing_whitespace", 0
        )
        quality_flag_counts["leading_whitespace"] = quality_flag_counts.get(
            "leading_whitespace", 0
        )
        quality_flag_counts["repeated_horizontal_whitespace"] = (
            quality_flag_counts.get("repeated_horizontal_whitespace", 0)
        )

        write_sampler_categories(
            quality_sampler,
            output_dir / "quality_flag_examples",
            prefix="",
        )
        write_sampler_categories(
            category_sampler,
            output_dir / "category_examples",
            prefix="",
        )
        write_sampler_categories(
            apostrophe_sampler,
            output_dir / "category_examples",
            prefix="apostrophe_",
        )

        write_csv(
            output_dir / "shortest_record_examples.csv",
            ["source_record_id", "character_count", "text_preview"],
            extremes.shortest_rows(),
        )
        write_csv(
            output_dir / "longest_record_examples.csv",
            ["source_record_id", "character_count", "text_preview"],
            extremes.longest_rows(),
        )
        write_csv(
            output_dir / "boundary_whitespace_summary.csv",
            ["metric", "record_count"],
            [
                {"metric": key, "record_count": quality_flag_counts.get(key, 0)}
                for key in (
                    "leading_whitespace",
                    "trailing_whitespace",
                    "starts_with_ascii_space",
                    "starts_with_tab",
                    "starts_with_newline",
                    "ends_with_exactly_one_ascii_space",
                    "ends_with_multiple_ascii_spaces",
                    "ends_with_tab",
                    "ends_with_newline",
                    "no_trailing_whitespace",
                    "repeated_horizontal_whitespace",
                )
            ],
        )

        post_fingerprint = fingerprint_file(input_path)
        immutability_passed = pre_fingerprint == post_fingerprint
        if not immutability_passed:
            raise RuntimeError(
                "Input corpus fingerprint changed during the audit. "
                "Results must not be approved."
            )

        elapsed_seconds = time.time() - started_at_epoch
        records_scanned = counters["records_scanned"]
        high_match_total = sum(
            apostrophe_rule_counts[rule_id]
            for rule_id in HIGH_CONFIDENCE_RULE_IDS
        )
        ambiguous_match_total = sum(
            apostrophe_rule_counts[rule_id]
            for rule_id in AMBIGUOUS_RULE_IDS
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

        add_check(
            "record_type_partition",
            counters["dictionary_records"] + counters["non_dictionary_records"],
            records_scanned,
        )
        add_check(
            "dictionary_text_field_partition",
            counters["records_with_text_field"] + counters["missing_text_field"],
            counters["dictionary_records"],
        )
        add_check(
            "text_type_partition",
            counters["string_text_records"] + counters["non_string_text_records"],
            counters["records_with_text_field"],
        )
        add_check(
            "trailing_boundary_partition",
            quality_flag_counts["trailing_whitespace"]
            + quality_flag_counts["no_trailing_whitespace"],
            counters["string_text_records"],
        )
        add_check(
            "duplicate_occurrence_count",
            quality_flag_counts["exact_duplicate_after_first"],
            duplicate_summary["duplicate_occurrences_after_first"],
        )
        add_check(
            "block_record_total",
            block_validation_totals["records_scanned_in_block"],
            records_scanned,
        )
        add_check(
            "block_high_confidence_record_total",
            block_validation_totals["records_with_high_confidence_gap"],
            counters["records_with_high_confidence_gap"],
        )
        add_check(
            "block_ambiguous_record_total",
            block_validation_totals["records_with_ambiguous_gap"],
            counters["records_with_ambiguous_gap"],
        )
        add_check(
            "block_high_confidence_match_total",
            block_validation_totals["high_confidence_match_count"],
            high_match_total,
        )
        add_check(
            "block_ambiguous_match_total",
            block_validation_totals["ambiguous_match_count"],
            ambiguous_match_total,
        )

        output_manifest = {
            "summary.json": "Machine-readable audit metrics and configuration.",
            "report.md": "Compact human-readable Phase 1 report.",
            "validation_checks.json": (
                "Internal partition and block-total checks; every item must PASS."
            ),
            "apostrophe_gap_records.csv": (
                "One row per record with high-confidence or ambiguous gap signals."
            ),
            "apostrophe_gap_block_distribution.csv": (
                "Gap-signal distribution by corpus-position block."
            ),
            "source_credit_records.csv": (
                "Records containing source/credit phrases or matching the stricter "
                "standalone heuristic."
            ),
            "short_record_roles.csv": (
                "Review-only short sentence, credit, and fragment/unknown roles."
            ),
            "duplicate_groups.csv": "Exact duplicate groups and record positions.",
            "duplicate_index.sqlite3": (
                "SQLite exact-duplicate index; large output, do not commit to Git."
            ),
            "boundary_whitespace_summary.csv": "Boundary whitespace metrics.",
            "shortest_record_examples.csv": "Shortest record examples.",
            "longest_record_examples.csv": "Longest record examples.",
            "quality_flag_examples/": (
                "Separate deterministic CSV examples for each quality flag."
            ),
            "category_examples/": (
                "Separate deterministic examples for new v2.1 heuristic categories."
            ),
        }

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
            "short_record_max_chars": args.short_record_max_chars,
            "standalone_credit_max_chars": args.standalone_credit_max_chars,
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
            "duplicates": {
                **duplicate_summary,
                "tracking_mode": "SQLite + SHA-256 lookup + full-text equality",
                "database_path": str(db_path),
            },
            "quality_flags": {
                key: int(value) for key, value in sorted(quality_flag_counts.items())
            },
            "script_distribution": {
                key: int(value) for key, value in sorted(script_counts.items())
            },
            "apostrophe_variants": {
                key: int(apostrophe_variant_counts.get(key, 0))
                for key in APOSTROPHE_VARIANTS
            },
            "suspicious_unicode_occurrences": {
                key: int(value)
                for key, value in sorted(suspicious_unicode_total.items())
            },
            "apostrophe_gap": {
                "records_with_high_confidence_gap": counters[
                    "records_with_high_confidence_gap"
                ],
                "records_with_ambiguous_gap": counters[
                    "records_with_ambiguous_gap"
                ],
                "records_with_any_gap_signal": counters[
                    "records_with_any_gap_signal"
                ],
                "high_confidence_match_count": high_match_total,
                "ambiguous_match_count": ambiguous_match_total,
                "rule_match_counts": {
                    rule_id: int(apostrophe_rule_counts.get(rule_id, 0))
                    for rule_id in APOSTROPHE_GAP_RULES
                },
                "high_confidence_rule_ids": sorted(HIGH_CONFIDENCE_RULE_IDS),
                "ambiguous_rule_ids": sorted(AMBIGUOUS_RULE_IDS),
                "review_only": True,
                "text_corrected": False,
            },
            "source_credit": {
                "CONTAINS_SOURCE_OR_CREDIT_PHRASE": counters[
                    "contains_source_or_credit_phrase"
                ],
                "STANDALONE_SOURCE_OR_CREDIT": counters[
                    "standalone_source_or_credit"
                ],
                "marker_record_counts": {
                    key: int(value) for key, value in sorted(source_marker_counts.items())
                },
                "review_only": True,
            },
            "short_record_roles": {
                "SHORT_SENTENCE_LIKE": short_role_counts["SHORT_SENTENCE_LIKE"],
                "SHORT_CREDIT_LIKE": short_role_counts["SHORT_CREDIT_LIKE"],
                "SHORT_FRAGMENT_OR_UNKNOWN": short_role_counts[
                    "SHORT_FRAGMENT_OR_UNKNOWN"
                ],
                "total_short_records": sum(short_role_counts.values()),
                "review_only": True,
            },
            "block_distribution": {
                "block_rows_written": block_rows_written,
                "records_accounted_for": block_validation_totals[
                    "records_scanned_in_block"
                ],
                "validation_totals": {
                    key: int(value)
                    for key, value in sorted(block_validation_totals.items())
                },
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
                "version": getattr(ijson, "__version__", "unknown"),
                "backend": getattr(ijson, "backend", "unknown"),
                "item_prefix": "item",
            },
            "output_manifest": output_manifest,
            "safety": {
                "raw_corpus_modified": False,
                "automatic_correction_performed": False,
                "automatic_exclusion_performed": False,
                "heuristic_labels_are_gold": False,
                "review_required": True,
            },
        }

        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        (output_dir / "validation_checks.json").write_text(
            json.dumps(validation_checks, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        (output_dir / "report.md").write_text(
            build_markdown_report(summary), encoding="utf-8"
        )
        (output_dir / "run_configuration.json").write_text(
            json.dumps(
                {
                    "script_version": SCRIPT_VERSION,
                    "arguments": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        incomplete_marker = output_dir / "_AUDIT_INCOMPLETE"
        if incomplete_marker.exists():
            incomplete_marker.unlink()
        (output_dir / "_AUDIT_COMPLETE").write_text(
            "Audit completed successfully. Immutability check: PASS.\n",
            encoding="utf-8",
        )
        return summary

    except Exception:
        try:
            duplicate_tracker.close()
        except Exception:
            pass
        failure_path = output_dir / "_AUDIT_FAILED.txt"
        failure_path.write_text(
            "The audit failed. Do not approve partial outputs. "
            "See the terminal traceback.\n",
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
                "dictionary_records": summary["record_structure"][
                    "dictionary_records"
                ],
                "string_text_records": summary["record_structure"][
                    "string_text_records"
                ],
                "duplicate_groups": summary["duplicates"]["duplicate_groups"],
                "duplicate_occurrences_after_first": summary["duplicates"][
                    "duplicate_occurrences_after_first"
                ],
                "records_with_high_confidence_gap": summary["apostrophe_gap"][
                    "records_with_high_confidence_gap"
                ],
                "records_with_ambiguous_gap": summary["apostrophe_gap"][
                    "records_with_ambiguous_gap"
                ],
                "block_rows_written": summary["block_distribution"][
                    "block_rows_written"
                ],
                "immutability_check": summary["immutability_check"],
                "output_directory": summary["output_directory"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
