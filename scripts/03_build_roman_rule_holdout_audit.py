#!/usr/bin/env python3
"""
Build a fresh holdout audit for the conservative Roman-expansion rule.

Important
---------
This script is read-only regarding:
- the source corpus;
- Phase 2 normalized records;
- existing Phase 3 inventories;
- the sentence segmenter.

It scans documents not used by segment_10k_v1 or segment_10k_v2:
rows 1,000-1,999 from each selected shard.

Candidate rule
--------------
Protect a Roman-expansion boundary only when:

1. an initial is followed by a suspicious ordinal expansion;
2. the right context contains at least 12 tokens;
3. no metadata/bibliography term is detected;
4. no joined ordinal corruption is detected.

This rule came from the zero-false-positive manual-data experiment,
but this holdout is independent evidence.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


INPUT_ROOT = Path(
    "outputs/02_curated_corpus/"
    "normalize_full_v1/normalized_records"
)

OUTPUT_ROOT = Path(
    "outputs/03_sentence_inventory/"
    "roman_rule_holdout_10k_v1"
)

PART_NUMBERS = [
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
]

ROW_START = 1_000
ROWS_PER_PART = 1_000
CONTEXT_CHARACTERS = 220
RANDOM_SEED = 20260615

PROTECT_SAMPLE_SIZE = 40
REJECT_SAMPLE_SIZE = 20


WORD_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿʻʼ‘’']+"
)

ROMAN_EVENT_RE = re.compile(
    r"(?P<initial>"
    r"(?:^|\s)"
    r"(?:"
    r"[A-Z]|"
    r"Sh|SH|Ch|CH|"
    r"O['ʻʼ‘’]|"
    r"G['ʻʼ‘’]"
    r")\."
    r")"
    r"\s+"
    r"(?P<ordinal>"
    r"bir\s+minginchi|"
    r"besh\s+yuzinchi|"
    r"birinchi|"
    r"beshinchi|"
    r"o'ninchi|"
    r"yuzinchi"
    r")"
    r"(?P<boundary>\.)"
)

METADATA_RE = re.compile(
    r"\b(?:"
    r"professor|akademigi|muallif|mualliflar|"
    r"taqrizchi|taqrizchilar|redaktor|"
    r"nashriyot|bibliograf|adabiyotlar|"
    r"tom|bet|sahifa|formula|jadval|"
    r"kg|mkg|mln"
    r")\b",
    re.IGNORECASE,
)

JOINED_ORDINAL_RE = re.compile(
    r"(?:"
    r"birinchi|beshinchi|o'ninchi|"
    r"yuzinchi|minginchi"
    r")[A-Za-zʻʼ‘’']+",
    re.IGNORECASE,
)

WHITESPACE_RE = re.compile(r"\s+")


def compact(text: str) -> str:
    """Collapse whitespace for readable audit context."""
    return WHITESPACE_RE.sub(" ", text).strip()


def pipe_join(values: list[str]) -> str:
    """Store a list safely in one CSV field."""
    return "|".join(str(value) for value in values)


def extract_event_rows(
    *,
    source_record_id: int,
    text: str,
    parent_review_flags: list[str],
    part_number: int,
    local_row_number: int,
) -> list[dict[str, Any]]:
    """Extract all Roman-expansion candidates from one document."""
    rows: list[dict[str, Any]] = []

    for match_index, match in enumerate(
        ROMAN_EVENT_RE.finditer(text)
    ):
        boundary_position = match.start("boundary")

        right_start = match.end("boundary")

        while (
            right_start < len(text)
            and text[right_start].isspace()
        ):
            right_start += 1

        left_start = max(
            0,
            match.start("initial") - CONTEXT_CHARACTERS,
        )

        right_end = min(
            len(text),
            right_start + CONTEXT_CHARACTERS,
        )

        left_context = compact(
            text[left_start:match.end("boundary")]
        )

        right_context = compact(
            text[right_start:right_end]
        )

        right_tokens = WORD_RE.findall(right_context)

        has_metadata = bool(
            METADATA_RE.search(right_context)
        )

        has_joined_ordinal = bool(
            JOINED_ORDINAL_RE.search(right_context)
        )

        predicted_protection = (
            len(right_tokens) >= 12
            and not has_metadata
            and not has_joined_ordinal
        )

        rejection_reasons: list[str] = []

        if len(right_tokens) < 12:
            rejection_reasons.append(
                "RIGHT_TOKEN_COUNT_LT_12"
            )

        if has_metadata:
            rejection_reasons.append(
                "METADATA_SIGNAL"
            )

        if has_joined_ordinal:
            rejection_reasons.append(
                "JOINED_ORDINAL_DAMAGE"
            )

        rows.append(
            {
                "event_id": (
                    f"{source_record_id}:"
                    f"{boundary_position}:"
                    f"{match_index}"
                ),
                "source_record_id": source_record_id,
                "part_number": part_number,
                "local_row_number": local_row_number,
                "boundary_parent_char_position": (
                    boundary_position
                ),
                "initial_text": compact(
                    match.group("initial")
                ),
                "ordinal_text": compact(
                    match.group("ordinal")
                ),
                "boundary_marker": ".",
                "left_context": left_context,
                "right_context": right_context,
                "right_first_token": (
                    right_tokens[0]
                    if right_tokens
                    else ""
                ),
                "right_token_count": len(right_tokens),
                "has_metadata_signal": has_metadata,
                "has_joined_ordinal_damage": (
                    has_joined_ordinal
                ),
                "parent_review_flags": pipe_join(
                    parent_review_flags
                ),
                "rule_prediction": (
                    "PROTECT_BOUNDARY"
                    if predicted_protection
                    else "REJECT_OR_REVIEW"
                ),
                "rejection_reasons": pipe_join(
                    rejection_reasons
                ),
                "human_decision": "",
                "review_note": "",
            }
        )

    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write rows with a stable schema."""
    if not rows:
        raise RuntimeError(
            f"Refusing to write empty CSV: {path}"
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_ROOT}\n"
            "It was not overwritten."
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    all_events: list[dict[str, Any]] = []
    metrics = Counter()
    per_part_documents = Counter()
    per_part_events = Counter()

    for part_number in PART_NUMBERS:
        input_path = (
            INPUT_ROOT
            / f"part-{part_number:05d}.jsonl"
        )

        if not input_path.exists():
            raise FileNotFoundError(input_path)

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for local_row_number, line in enumerate(handle):
                if local_row_number < ROW_START:
                    continue

                if (
                    local_row_number
                    >= ROW_START + ROWS_PER_PART
                ):
                    break

                parent = json.loads(line)

                source_record_id = int(
                    parent["source_record_id"]
                )

                text = parent["normalized_text"]

                parent_review_flags = list(
                    parent.get("review_flags", [])
                )

                event_rows = extract_event_rows(
                    source_record_id=source_record_id,
                    text=text,
                    parent_review_flags=(
                        parent_review_flags
                    ),
                    part_number=part_number,
                    local_row_number=local_row_number,
                )

                all_events.extend(event_rows)

                metrics["documents_scanned"] += 1
                per_part_documents[part_number] += 1
                per_part_events[part_number] += len(
                    event_rows
                )

    expected_documents = (
        len(PART_NUMBERS) * ROWS_PER_PART
    )

    if metrics["documents_scanned"] != expected_documents:
        raise RuntimeError(
            f"Expected {expected_documents} documents, "
            f"found {metrics['documents_scanned']}"
        )

    if not all_events:
        raise RuntimeError(
            "No Roman-expansion candidates were detected."
        )

    all_events.sort(
        key=lambda row: (
            int(row["source_record_id"]),
            int(row["boundary_parent_char_position"]),
            row["event_id"],
        )
    )

    event_ids = [
        row["event_id"]
        for row in all_events
    ]

    if len(event_ids) != len(set(event_ids)):
        raise RuntimeError(
            "Duplicate holdout event IDs detected."
        )

    protect_rows = [
        row
        for row in all_events
        if row["rule_prediction"]
        == "PROTECT_BOUNDARY"
    ]

    reject_rows = [
        row
        for row in all_events
        if row["rule_prediction"]
        == "REJECT_OR_REVIEW"
    ]

    rng = random.Random(RANDOM_SEED)

    protect_sample = rng.sample(
        protect_rows,
        min(PROTECT_SAMPLE_SIZE, len(protect_rows)),
    )

    reject_sample = rng.sample(
        reject_rows,
        min(REJECT_SAMPLE_SIZE, len(reject_rows)),
    )

    review_sample = protect_sample + reject_sample

    review_sample.sort(
        key=lambda row: (
            0
            if row["rule_prediction"]
            == "PROTECT_BOUNDARY"
            else 1,
            int(row["source_record_id"]),
            int(row["boundary_parent_char_position"]),
        )
    )

    all_events_path = (
        OUTPUT_ROOT / "all_candidates.csv"
    )

    review_sample_path = (
        OUTPUT_ROOT / "manual_review_sample.csv"
    )

    metrics_path = OUTPUT_ROOT / "metrics.json"

    write_csv(all_events_path, all_events)
    write_csv(review_sample_path, review_sample)

    prediction_counts = Counter(
        row["rule_prediction"]
        for row in all_events
    )

    rejection_reason_counts = Counter()

    for row in all_events:
        for reason in row[
            "rejection_reasons"
        ].split("|"):
            if reason:
                rejection_reason_counts[reason] += 1

    final_metrics = {
        "experiment_name": (
            "roman_rule_holdout_10k_v1"
        ),
        "input_directory": str(INPUT_ROOT),
        "parts_sampled": PART_NUMBERS,
        "row_start_in_each_part": ROW_START,
        "rows_per_part": ROWS_PER_PART,
        "documents_scanned": metrics[
            "documents_scanned"
        ],
        "candidate_events": len(all_events),
        "unique_event_ids": len(set(event_ids)),
        "prediction_counts": dict(
            sorted(prediction_counts.items())
        ),
        "rejection_reason_counts": dict(
            sorted(rejection_reason_counts.items())
        ),
        "manual_review_sample_rows": len(
            review_sample
        ),
        "manual_review_protect_rows": len(
            protect_sample
        ),
        "manual_review_reject_rows": len(
            reject_sample
        ),
        "per_part_document_counts": {
            str(part): per_part_documents[part]
            for part in PART_NUMBERS
        },
        "per_part_event_counts": {
            str(part): per_part_events[part]
            for part in PART_NUMBERS
        },
        "candidate_rule": {
            "minimum_right_token_count": 12,
            "reject_metadata_signal": True,
            "reject_joined_ordinal_damage": True,
        },
    }

    metrics_path.write_text(
        json.dumps(
            final_metrics,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n--- fresh Roman-rule holdout ---")
    print(
        "documents_scanned:",
        final_metrics["documents_scanned"],
    )
    print(
        "candidate_events:",
        final_metrics["candidate_events"],
    )
    print(
        "prediction_counts:",
        final_metrics["prediction_counts"],
    )
    print(
        "rejection_reason_counts:",
        final_metrics[
            "rejection_reason_counts"
        ],
    )
    print(
        "manual_review_sample_rows:",
        final_metrics[
            "manual_review_sample_rows"
        ],
    )
    print(
        "output:",
        OUTPUT_ROOT,
    )

    print("\n--- predicted-protection preview ---")

    for number, row in enumerate(
        protect_sample[:12],
        start=1,
    ):
        print("\n" + "=" * 100)
        print(
            f"{number}. source="
            f"{row['source_record_id']} "
            f"position="
            f"{row['boundary_parent_char_position']}"
        )
        print(
            "right_token_count:",
            row["right_token_count"],
        )
        print(
            f"{row['left_context']} "
            f"[.] {row['right_context']}"
        )

    print("\nAllowed human_decision values:")
    print("PROTECT_BOUNDARY")
    print("KEEP_SPLIT")
    print("SOURCE_TOO_CORRUPTED")
    print("UNCERTAIN")
    print("\nNo segmenter rule was changed.")


if __name__ == "__main__":
    main()
