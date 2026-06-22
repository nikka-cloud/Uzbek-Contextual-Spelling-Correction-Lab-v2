#!/usr/bin/env python3
"""
Create a context-rich manual audit of Phase 3 boundary decisions.

Unlike the earlier sentence-level audit, this export identifies the
exact punctuation decision and shows text on both sides.

It does not modify the sentence inventory or source corpus.
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

OUTPUT_PATH = (
    INVENTORY_ROOT
    / "manual_boundary_event_audit_v2.csv"
)

RANDOM_SEED = 20260615
ROWS_PER_GROUP = 25
CONTEXT_CHARACTERS = 220

GROUP_ORDER = [
    "UNFLAGGED_SPLIT",
    "LOWERCASE_AFTER_TERMINAL_SPLIT",
    "INITIAL_BOUNDARY_PROTECTED",
    "SUSPICIOUS_ORDINAL_PERIOD_PROTECTED",
    "ABBREVIATION_BOUNDARY_PROTECTED",
    "LOWERCASE_AFTER_STRONG_PUNCTUATION_PROTECTED",
    "QUOTE_OR_SPEECH_CONTINUATION",
    "NO_TERMINAL_PUNCTUATION",
    "VERY_LONG_SEGMENT",
    "VERY_SHORT_SEGMENT_CANDIDATE",
]

INITIAL_RE = re.compile(
    r"\b(?:"
    r"[A-Z]|"
    r"Sh|SH|Ch|CH|"
    r"O['ʻʼ‘’]|"
    r"G['ʻʼ‘’]"
    r")\."
)

ORDINAL_RE = re.compile(
    r"(?:"
    r"bir\s+minginchi|"
    r"besh\s+yuzinchi|"
    r"birinchi|"
    r"beshinchi|"
    r"o'ninchi|"
    r"yuzinchi"
    r")\.",
    re.IGNORECASE,
)

ABBREVIATION_RE = re.compile(
    r"\b(?:AJ|MCHJ|prof|dr|kv)\.",
    re.IGNORECASE,
)

STRONG_LOWERCASE_RE = re.compile(
    r"[?!](?:[\"»”’)\]]*)\s+"
    r"[a-zà-öø-ÿʻʼ‘’']"
)

SPEECH_CONTINUATION_RE = re.compile(
    r"[.!?](?:[\"»”’)\]]*)\s+"
    r"(?:"
    r"deb|dedi|deya|degan|deyiladi|"
    r"deganidek|shiori"
    r")\b",
    re.IGNORECASE,
)

PROTECTED_PATTERNS = {
    "INITIAL_BOUNDARY_PROTECTED": INITIAL_RE,
    "SUSPICIOUS_ORDINAL_PERIOD_PROTECTED": ORDINAL_RE,
    "ABBREVIATION_BOUNDARY_PROTECTED": ABBREVIATION_RE,
    "LOWERCASE_AFTER_STRONG_PUNCTUATION_PROTECTED": (
        STRONG_LOWERCASE_RE
    ),
    "QUOTE_OR_SPEECH_CONTINUATION": (
        SPEECH_CONTINUATION_RE
    ),
}


def compact(text: str) -> str:
    """Display whitespace consistently without changing stored text."""
    return re.sub(r"\s+", " ", text).strip()


def punctuation_position(
    matched_text: str,
    group_name: str,
) -> tuple[int, str]:
    """Locate the protected punctuation inside one regex match."""
    if (
        group_name
        == "LOWERCASE_AFTER_STRONG_PUNCTUATION_PROTECTED"
    ):
        allowed_markers = "?!"

    elif group_name == "QUOTE_OR_SPEECH_CONTINUATION":
        allowed_markers = ".!?"

    else:
        allowed_markers = "."

    positions = [
        matched_text.find(marker)
        for marker in allowed_markers
        if matched_text.find(marker) >= 0
    ]

    if not positions:
        raise ValueError(
            "No expected punctuation found in "
            f"{matched_text!r} for group {group_name!r}; "
            f"expected one of {allowed_markers!r}"
        )

    position = min(positions)

    return position, matched_text[position]


def protected_event(
    *,
    group_name: str,
    row: dict[str, Any],
    match: re.Match[str],
) -> dict[str, Any]:
    """Build one exact protected-boundary audit event."""
    sentence_text = row["sentence_text"]

    punctuation_offset, marker = punctuation_position(
        match.group(0),
        group_name,
    )

    local_position = match.start() + punctuation_offset
    parent_position = (
        row["sentence_start_char"] + local_position
    )

    left_start = max(
        0,
        local_position - CONTEXT_CHARACTERS,
    )
    right_end = min(
        len(sentence_text),
        local_position + 1 + CONTEXT_CHARACTERS,
    )

    left_context = compact(
        sentence_text[left_start:local_position]
    )
    right_context = compact(
        sentence_text[
            local_position + 1:right_end
        ]
    )

    return {
        "audit_group": group_name,
        "decision_type": "PROTECT_NO_SPLIT",
        "source_record_id": row["source_record_id"],
        "sentence_id": row["sentence_id"],
        "sentence_index_in_record": (
            row["sentence_index_in_record"]
        ),
        "boundary_parent_char_position": parent_position,
        "boundary_marker": marker,
        "left_context": left_context,
        "right_context": right_context,
        "sentence_character_count": (
            row["sentence_character_count"]
        ),
        "sentence_token_count": row["sentence_token_count"],
        "boundary_flags": "|".join(
            row["boundary_flags"]
        ),
        "parent_review_flags": "|".join(
            row["parent_review_flags"]
        ),
        "sentence_text": sentence_text,
        "human_boundary_label": "",
        "human_health_label": "",
        "review_note": "",
    }


def split_event(
    *,
    group_name: str,
    row: dict[str, Any],
    next_row: dict[str, Any],
) -> dict[str, Any]:
    """Build one exact split-boundary audit event."""
    sentence_text = row["sentence_text"]
    next_text = next_row["sentence_text"]
    marker = row["terminal_punctuation"]

    left_context = compact(
        sentence_text[-CONTEXT_CHARACTERS:]
    )
    right_context = compact(
        next_text[:CONTEXT_CHARACTERS]
    )

    return {
        "audit_group": group_name,
        "decision_type": "SPLIT",
        "source_record_id": row["source_record_id"],
        "sentence_id": row["sentence_id"],
        "sentence_index_in_record": (
            row["sentence_index_in_record"]
        ),
        "boundary_parent_char_position": (
            row["sentence_end_char"]
        ),
        "boundary_marker": marker,
        "left_context": left_context,
        "right_context": right_context,
        "sentence_character_count": (
            row["sentence_character_count"]
        ),
        "sentence_token_count": row["sentence_token_count"],
        "boundary_flags": "|".join(
            row["boundary_flags"]
        ),
        "parent_review_flags": "|".join(
            row["parent_review_flags"]
        ),
        "sentence_text": sentence_text,
        "human_boundary_label": "",
        "human_health_label": "",
        "review_note": "",
    }


def health_event(
    *,
    group_name: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Build a whole-sentence health and fragment review row."""
    sentence_text = row["sentence_text"]

    return {
        "audit_group": group_name,
        "decision_type": "WHOLE_SENTENCE_REVIEW",
        "source_record_id": row["source_record_id"],
        "sentence_id": row["sentence_id"],
        "sentence_index_in_record": (
            row["sentence_index_in_record"]
        ),
        "boundary_parent_char_position": "",
        "boundary_marker": "",
        "left_context": "",
        "right_context": "",
        "sentence_character_count": (
            row["sentence_character_count"]
        ),
        "sentence_token_count": row["sentence_token_count"],
        "boundary_flags": "|".join(
            row["boundary_flags"]
        ),
        "parent_review_flags": "|".join(
            row["parent_review_flags"]
        ),
        "sentence_text": sentence_text,
        "human_boundary_label": "",
        "human_health_label": "",
        "review_note": "",
    }


def main() -> None:
    rows_by_document: dict[int, list[dict[str, Any]]] = (
        defaultdict(list)
    )

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

    events: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    pattern_not_located = Counter()
    seen_event_keys: set[tuple[Any, ...]] = set()

    for source_record_id, document_rows in sorted(
        rows_by_document.items()
    ):
        for index, row in enumerate(document_rows):
            flags = set(row["boundary_flags"])

            # Actual split decisions are audited using the next
            # sentence as right-hand context.
            if index + 1 < len(document_rows):
                next_row = document_rows[index + 1]

                if (
                    row["segmentation_method"]
                    == "RULE_TERMINAL_PUNCTUATION"
                ):
                    if not flags:
                        event = split_event(
                            group_name="UNFLAGGED_SPLIT",
                            row=row,
                            next_row=next_row,
                        )
                        events["UNFLAGGED_SPLIT"].append(
                            event
                        )

                    if (
                        "LOWERCASE_AFTER_TERMINAL"
                        in flags
                        and
                        "LOWERCASE_AFTER_STRONG_PUNCTUATION_PROTECTED"
                        not in flags
                    ):
                        event = split_event(
                            group_name=(
                                "LOWERCASE_AFTER_TERMINAL_SPLIT"
                            ),
                            row=row,
                            next_row=next_row,
                        )
                        events[
                            "LOWERCASE_AFTER_TERMINAL_SPLIT"
                        ].append(event)

            # Protected punctuation decisions are found inside
            # the retained sentence.
            for group_name, pattern in (
                PROTECTED_PATTERNS.items()
            ):
                if group_name not in flags:
                    continue

                matches = list(
                    pattern.finditer(row["sentence_text"])
                )

                if not matches:
                    pattern_not_located[group_name] += 1
                    continue

                for match in matches:
                    event = protected_event(
                        group_name=group_name,
                        row=row,
                        match=match,
                    )

                    event_key = (
                        group_name,
                        row["sentence_id"],
                        event[
                            "boundary_parent_char_position"
                        ],
                    )

                    if event_key in seen_event_keys:
                        continue

                    seen_event_keys.add(event_key)
                    events[group_name].append(event)

            if "NO_TERMINAL_PUNCTUATION" in flags:
                events["NO_TERMINAL_PUNCTUATION"].append(
                    health_event(
                        group_name=(
                            "NO_TERMINAL_PUNCTUATION"
                        ),
                        row=row,
                    )
                )

            if "VERY_LONG_SEGMENT" in flags:
                events["VERY_LONG_SEGMENT"].append(
                    health_event(
                        group_name="VERY_LONG_SEGMENT",
                        row=row,
                    )
                )

            if (
                row["sentence_character_count"] <= 15
                or row["sentence_token_count"] <= 2
            ):
                events[
                    "VERY_SHORT_SEGMENT_CANDIDATE"
                ].append(
                    health_event(
                        group_name=(
                            "VERY_SHORT_SEGMENT_CANDIDATE"
                        ),
                        row=row,
                    )
                )

    random_generator = random.Random(RANDOM_SEED)
    selected: list[dict[str, Any]] = []

    for group_name in GROUP_ORDER:
        candidates = sorted(
            events[group_name],
            key=lambda event: (
                event["source_record_id"],
                event["sentence_index_in_record"],
                str(
                    event[
                        "boundary_parent_char_position"
                    ]
                ),
                event["sentence_id"],
            ),
        )

        sample_size = min(
            ROWS_PER_GROUP,
            len(candidates),
        )

        if sample_size:
            selected.extend(
                random_generator.sample(
                    candidates,
                    sample_size,
                )
            )

    selected.sort(
        key=lambda event: (
            GROUP_ORDER.index(
                event["audit_group"]
            ),
            event["source_record_id"],
            event["sentence_index_in_record"],
            str(
                event["boundary_parent_char_position"]
            ),
        )
    )

    fieldnames = [
        "audit_group",
        "decision_type",
        "source_record_id",
        "sentence_id",
        "sentence_index_in_record",
        "boundary_parent_char_position",
        "boundary_marker",
        "left_context",
        "right_context",
        "sentence_character_count",
        "sentence_token_count",
        "boundary_flags",
        "parent_review_flags",
        "sentence_text",
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
        writer.writerows(selected)

    selected_counts = Counter(
        event["audit_group"]
        for event in selected
    )

    print("\n--- available and selected events ---")
    for group_name in GROUP_ORDER:
        print(
            f"{group_name:52s} "
            f"available={len(events[group_name]):7,d} "
            f"selected={selected_counts[group_name]:3,d}"
        )

    print("\n--- protected patterns not located ---")
    print(
        dict(pattern_not_located)
        if pattern_not_located
        else "NONE"
    )

    print("\n--- output ---")
    print(f"path: {OUTPUT_PATH}")
    print(f"rows: {len(selected):,}")
    print(f"size_bytes: {OUTPUT_PATH.stat().st_size:,}")

    print("\n--- first boundary events ---")
    for event in selected[:8]:
        print("\n" + "-" * 100)
        print(
            f"group={event['audit_group']} "
            f"decision={event['decision_type']}"
        )
        print(
            event["left_context"]
            + " "
            + f"[{event['boundary_marker']}]"
            + " "
            + event["right_context"]
        )

    print("\nAllowed human_boundary_label values:")
    print("CORRECT_DECISION")
    print("WRONG_SPLIT")
    print("WRONG_PROTECTION")
    print("UNCERTAIN")
    print("SOURCE_TOO_CORRUPTED")

    print("\nSuggested human_health_label values:")
    print("HEALTHY_CANDIDATE")
    print("SPELLING_OR_OCR_DAMAGE")
    print("FORMAT_OR_LIST")
    print("FRAGMENT")
    print("UNCERTAIN")


if __name__ == "__main__":
    main()
