#!/usr/bin/env python3
"""
Audit the Phase 3 LOWERCASE_AFTER_TERMINAL split family.

Purpose
-------
Measure recurring contexts behind lowercase-after-period boundaries
before adding any new segmentation rule.

This script is read-only with respect to the sentence inventory.
It creates an analysis CSV and compact deterministic examples.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


INVENTORY_ROOT = Path(
    "outputs/03_sentence_inventory/segment_10k_v1"
)

SENTENCES_PATH = INVENTORY_ROOT / "sentences.jsonl"

MANUAL_AUDIT_PATH = (
    INVENTORY_ROOT
    / "manual_boundary_event_audit_v2.csv"
)

OUTPUT_PATH = (
    INVENTORY_ROOT
    / "lowercase_split_family_audit_v1.csv"
)

RANDOM_SEED = 20260615
MAX_PRINTED_EXAMPLES_PER_FAMILY = 8
CONTEXT_CHARACTERS = 180

WORD_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿʻʼ‘’'-]+"
)

SUSPICIOUS_ORDINAL_END_RE = re.compile(
    r"(?:"
    r"bir\s+minginchi|"
    r"besh\s+yuzinchi|"
    r"birinchi|"
    r"beshinchi|"
    r"o'ninchi|"
    r"yuzinchi"
    r")\.\s*$",
    re.IGNORECASE,
)

INITIAL_NEAR_ORDINAL_RE = re.compile(
    r"(?:"
    r"\b[A-Z]\.|"
    r"\b(?:Sh|SH|Ch|CH)\."
    r").{0,80}"
    r"(?:"
    r"bir\s+minginchi|"
    r"besh\s+yuzinchi|"
    r"birinchi|"
    r"beshinchi|"
    r"o'ninchi|"
    r"yuzinchi"
    r")\.\s*$",
    re.IGNORECASE,
)

NUMBER_WORDS = {
    "nol",
    "bir",
    "ikki",
    "uch",
    "to'rt",
    "tort",
    "besh",
    "olti",
    "yetti",
    "sakkiz",
    "to'qqiz",
    "toqqiz",
    "o'n",
    "on",
    "yigirma",
    "o'ttiz",
    "ottiz",
    "qirq",
    "ellik",
    "oltmish",
    "yetmish",
    "sakson",
    "to'qson",
    "toqson",
    "yuz",
    "ming",
}

DOTTED_CONTINUATION_WORDS = {
    "uz",
    "ru",
    "com",
    "net",
    "org",
    "day",
    "info",
    "news",
}


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def first_word(text: str) -> str:
    match = WORD_RE.search(text)
    return match.group(0) if match else ""


def last_word(text: str) -> str:
    matches = list(WORD_RE.finditer(text))
    return matches[-1].group(0) if matches else ""


def classify_event(
    *,
    left_text: str,
    right_text: str,
    left_token_count: int,
    right_token_count: int,
    parent_review_flags: list[str],
) -> list[str]:
    """
    Assign analysis families.

    These are candidate signals, not automatic correction labels.
    """
    families: list[str] = []

    left_token = last_word(
        left_text.rstrip(".!?…")
    )
    right_token = first_word(right_text)

    left_lower = left_token.lower()
    right_lower = right_token.lower()

    if (
        left_token
        and right_lower in DOTTED_CONTINUATION_WORDS
    ):
        families.append(
            "DOTTED_NAME_OR_DOMAIN_CANDIDATE"
        )

    if SUSPICIOUS_ORDINAL_END_RE.search(left_text):
        families.append(
            "SUSPICIOUS_ORDINAL_AT_BOUNDARY"
        )

    if INITIAL_NEAR_ORDINAL_RE.search(left_text):
        families.append(
            "ROMAN_EXPANSION_NEAR_INITIAL"
        )

    if right_lower in NUMBER_WORDS:
        families.append(
            "NUMBER_LED_NEXT_SEGMENT"
        )

    if (
        left_token_count <= 3
        or right_token_count <= 3
    ):
        families.append(
            "SHORT_OR_METADATA_CONTEXT"
        )

    if (
        left_token
        and right_token
        and left_token[0].islower()
        and right_token[0].islower()
        and len(left_token) >= 3
        and len(right_token) >= 3
        and right_lower not in NUMBER_WORDS
    ):
        families.append(
            "LOWERCASE_WORD_PAIR_CANDIDATE"
        )

    if parent_review_flags:
        families.append(
            "PARENT_REVIEW_FLAGGED"
        )

    if not families:
        families.append(
            "OTHER_LOWERCASE_SPLIT"
        )

    return sorted(set(families))


def load_manual_labels() -> dict[
    tuple[int, int], dict[str, str]
]:
    labels: dict[
        tuple[int, int], dict[str, str]
    ] = {}

    with MANUAL_AUDIT_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            if (
                row["audit_group"]
                != "LOWERCASE_AFTER_TERMINAL_SPLIT"
            ):
                continue

            key = (
                int(row["source_record_id"]),
                int(row["sentence_index_in_record"]),
            )

            labels[key] = {
                "human_boundary_label": (
                    row["human_boundary_label"]
                ),
                "human_health_label": (
                    row["human_health_label"]
                ),
                "review_note": row["review_note"],
            }

    return labels


def main() -> None:
    rows_by_document: dict[
        int, list[dict[str, Any]]
    ] = defaultdict(list)

    with SENTENCES_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            row = json.loads(line)
            rows_by_document[
                row["source_record_id"]
            ].append(row)

    for document_rows in rows_by_document.values():
        document_rows.sort(
            key=lambda row: row[
                "sentence_index_in_record"
            ]
        )

    manual_labels = load_manual_labels()

    events: list[dict[str, Any]] = []
    family_counts = Counter()
    family_manual_labels = defaultdict(Counter)
    token_pair_counts = Counter()
    terminal_counts = Counter()

    for source_record_id, document_rows in sorted(
        rows_by_document.items()
    ):
        for index, row in enumerate(document_rows):
            if index + 1 >= len(document_rows):
                continue

            flags = set(row["boundary_flags"])

            if (
                "LOWERCASE_AFTER_TERMINAL"
                not in flags
            ):
                continue

            if (
                "LOWERCASE_AFTER_STRONG_PUNCTUATION_PROTECTED"
                in flags
            ):
                continue

            next_row = document_rows[index + 1]

            left_text = row["sentence_text"]
            right_text = next_row["sentence_text"]

            left_token = last_word(
                left_text.rstrip(".!?…")
            )
            right_token = first_word(right_text)

            families = classify_event(
                left_text=left_text,
                right_text=right_text,
                left_token_count=row[
                    "sentence_token_count"
                ],
                right_token_count=next_row[
                    "sentence_token_count"
                ],
                parent_review_flags=row[
                    "parent_review_flags"
                ],
            )

            key = (
                source_record_id,
                row["sentence_index_in_record"],
            )

            manual = manual_labels.get(
                key,
                {
                    "human_boundary_label": "",
                    "human_health_label": "",
                    "review_note": "",
                },
            )

            event = {
                "source_record_id": source_record_id,
                "sentence_index_in_record": row[
                    "sentence_index_in_record"
                ],
                "sentence_id": row["sentence_id"],
                "boundary_marker": row[
                    "terminal_punctuation"
                ],
                "left_last_token": left_token,
                "right_first_token": right_token,
                "family_labels": "|".join(families),
                "left_sentence_character_count": row[
                    "sentence_character_count"
                ],
                "left_sentence_token_count": row[
                    "sentence_token_count"
                ],
                "right_sentence_character_count": (
                    next_row[
                        "sentence_character_count"
                    ]
                ),
                "right_sentence_token_count": (
                    next_row[
                        "sentence_token_count"
                    ]
                ),
                "parent_review_flags": "|".join(
                    row["parent_review_flags"]
                ),
                "left_context": compact(
                    left_text[-CONTEXT_CHARACTERS:]
                ),
                "right_context": compact(
                    right_text[:CONTEXT_CHARACTERS]
                ),
                **manual,
            }

            events.append(event)
            terminal_counts[
                row["terminal_punctuation"]
            ] += 1

            token_pair_counts[
                (
                    left_token.lower(),
                    right_token.lower(),
                )
            ] += 1

            for family in families:
                family_counts[family] += 1

                if manual[
                    "human_boundary_label"
                ]:
                    family_manual_labels[family][
                        manual[
                            "human_boundary_label"
                        ]
                    ] += 1

    expected_event_count = 7_333

    if len(events) != expected_event_count:
        raise RuntimeError(
            f"Expected {expected_event_count:,} events, "
            f"found {len(events):,}."
        )

    fieldnames = [
        "source_record_id",
        "sentence_index_in_record",
        "sentence_id",
        "boundary_marker",
        "left_last_token",
        "right_first_token",
        "family_labels",
        "left_sentence_character_count",
        "left_sentence_token_count",
        "right_sentence_character_count",
        "right_sentence_token_count",
        "parent_review_flags",
        "left_context",
        "right_context",
        "human_boundary_label",
        "human_health_label",
        "review_note",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(events)

    print("\n--- lowercase split family audit ---")
    print(f"events: {len(events):,}")
    print(
        "manually_labeled_events:",
        sum(
            bool(
                event["human_boundary_label"]
            )
            for event in events
        ),
    )
    print(
        "terminal_counts:",
        dict(terminal_counts),
    )

    print("\n--- family counts ---")
    for family, count in family_counts.most_common():
        percentage = 100 * count / len(events)
        print(
            f"{family:42s} "
            f"{count:6,d} "
            f"{percentage:7.3f}%"
        )

    print("\n--- manual outcomes inside each family ---")
    for family in sorted(family_manual_labels):
        print(
            f"{family:42s} "
            f"{dict(family_manual_labels[family])}"
        )

    print("\n--- top token pairs around the period ---")
    for (
        left_token,
        right_token,
    ), count in token_pair_counts.most_common(30):
        print(
            f"{left_token!r:22s} "
            f"-> {right_token!r:22s} "
            f"{count:,}"
        )

    random_generator = random.Random(RANDOM_SEED)

    print("\n--- deterministic family examples ---")

    events_by_family: dict[
        str, list[dict[str, Any]]
    ] = defaultdict(list)

    for event in events:
        for family in event[
            "family_labels"
        ].split("|"):
            events_by_family[family].append(event)

    for family in sorted(events_by_family):
        candidates = sorted(
            events_by_family[family],
            key=lambda event: (
                int(event["source_record_id"]),
                int(
                    event[
                        "sentence_index_in_record"
                    ]
                ),
            ),
        )

        sample_size = min(
            MAX_PRINTED_EXAMPLES_PER_FAMILY,
            len(candidates),
        )

        examples = random_generator.sample(
            candidates,
            sample_size,
        )

        print("\n" + family)

        for number, event in enumerate(
            examples,
            start=1,
        ):
            print(
                f"  {number}. "
                f"source={event['source_record_id']} "
                f"manual="
                f"{event['human_boundary_label'] or 'UNLABELED'}"
            )
            print(
                "     "
                + event["left_context"]
                + " "
                + f"[{event['boundary_marker']}]"
                + " "
                + event["right_context"]
            )

    print("\n--- output ---")
    print(f"path: {OUTPUT_PATH}")
    print(f"size_bytes: {OUTPUT_PATH.stat().st_size:,}")
    print("\nNo segmentation rule was changed.")


if __name__ == "__main__":
    main()
