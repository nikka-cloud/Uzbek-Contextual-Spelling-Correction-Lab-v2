#!/usr/bin/env python3

"""
Build a routing-signal audit from the 700 manually reviewed
corpus-viability examples.

This script does NOT route the full corpus.

Its purpose is to measure how candidate sentence-quality signals
behave against the existing human health labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any
import csv
import json
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    ROOT
    / "outputs"
    / "03_sentence_inventory"
    / "corpus_viability_audit_v1"
    / "estimation_random_review_batches_v1"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "03_sentence_inventory"
    / "routing_signal_audit_v1"
)

FEATURE_ROWS_PATH = (
    OUTPUT_DIR / "review_rows_with_features_v1.csv"
)

LABEL_COUNTS_PATH = (
    OUTPUT_DIR / "manual_label_counts_v1.csv"
)

SIGNAL_SUMMARY_PATH = (
    OUTPUT_DIR / "boolean_signal_summary_v1.csv"
)

NUMERIC_SUMMARY_PATH = (
    OUTPUT_DIR
    / "numeric_feature_summary_by_label_v1.csv"
)

SIGNAL_EXAMPLES_PATH = (
    OUTPUT_DIR / "signal_examples_v1.csv"
)

REPORT_PATH = (
    OUTPUT_DIR / "routing_signal_audit_report_v1.md"
)

SUMMARY_JSON_PATH = (
    OUTPUT_DIR / "routing_signal_audit_summary_v1.json"
)


ALLOWED_LABELS = {
    "HEALTHY",
    "MINOR_DAMAGE_BUT_USABLE",
    "FORMAT_OR_LIST",
    "FRAGMENT",
    "HEAVY_CORRUPTION",
    "MIXED_OR_FOREIGN_TEXT",
    "UNCERTAIN",
}

USABLE_LABELS = {
    "HEALTHY",
    "MINOR_DAMAGE_BUT_USABLE",
}

HARMFUL_CLEAN_LANE_LABELS = {
    "FORMAT_OR_LIST",
    "FRAGMENT",
    "HEAVY_CORRUPTION",
    "MIXED_OR_FOREIGN_TEXT",
}

HIGH_RISK_BOUNDARY_FLAGS = {
    "LOWERCASE_AFTER_TERMINAL",
    "INITIAL_BOUNDARY_PROTECTED",
    "SHORT_TOKEN_BEFORE_PERIOD",
    "VERY_LONG_SEGMENT",
    "LOWERCASE_AFTER_STRONG_PUNCTUATION_PROTECTED",
}

APOSTROPHE_PARENT_FLAGS = {
    "AMBIGUOUS_APOSTROPHE_GAP",
    "HIGH_CONFIDENCE_APOSTROPHE_GAP",
}


URL_RE = re.compile(
    r"""
    (?:
        https?://
        |
        www\.
        |
        \b[a-zA-Z0-9-]+\.
        (?:uz|com|org|net|edu|gov|ru|io)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

LEADING_LIST_RE = re.compile(
    r"""
    ^\s*
    (?:
        [-–—•▪◦*]
        |
        \(?\d{1,4}[\).:\-]
        |
        [A-Za-zА-Яа-яЎўҚқҒғҲҳ][\).]
    )
    \s+
    """,
    re.VERBOSE,
)

TABLE_DELIMITER_RE = re.compile(
    r"\t|\||(?:\s{3,})"
)

HEADING_END_RE = re.compile(
    r"[:：]\s*$"
)

TERMINAL_PUNCTUATION = {
    ".",
    "!",
    "?",
    "…",
    "。",
    "！",
    "？",
    '"',
    "”",
    "’",
    "»",
}


FEATURE_FIELDNAMES = [
    "sample_id",
    "part_number",
    "source_record_id",
    "sentence_id",
    "human_health_label",
    "boundary_flags",
    "parent_review_flags",
    "sentence_text",
    "stored_character_count",
    "actual_character_count",
    "character_count_matches",
    "stored_token_count",
    "actual_token_count",
    "token_count_matches",
    "nonspace_character_count",
    "letter_count",
    "digit_count",
    "digit_ratio",
    "numeric_token_count",
    "numeric_token_ratio",
    "uppercase_count",
    "uppercase_ratio",
    "latin_letter_count",
    "latin_ratio",
    "cyrillic_letter_count",
    "cyrillic_ratio",
    "other_script_letter_count",
    "signal_mixed_latin_cyrillic",
    "signal_url",
    "signal_email",
    "signal_url_or_email",
    "signal_leading_list_marker",
    "signal_table_delimiter",
    "signal_heading_like",
    "signal_high_digit_ratio_20pct",
    "signal_high_digit_ratio_30pct",
    "signal_high_numeric_token_ratio_30pct",
    "signal_high_uppercase_ratio_70pct",
    "signal_no_terminal_punctuation",
    "signal_very_short_3_tokens",
    "signal_short_5_tokens",
    "signal_long_over_35_tokens",
    "signal_long_over_240_chars",
    "signal_boundary_high_risk_any",
    "signal_apostrophe_gap_any",
    "signal_quote_or_speech_continuation",
    "signal_dotted_name_or_domain",
]


BOOLEAN_SIGNAL_FIELDS = [
    field
    for field in FEATURE_FIELDNAMES
    if field.startswith("signal_")
]


NUMERIC_FEATURE_FIELDS = [
    "actual_character_count",
    "actual_token_count",
    "digit_ratio",
    "numeric_token_ratio",
    "uppercase_ratio",
    "latin_ratio",
    "cyrillic_ratio",
]


def safe_int(value: Any) -> int:
    text = str(value or "").strip()

    if not text:
        return 0

    return int(float(text))


def percentage(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return round(
        numerator / denominator * 100.0,
        2,
    )


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def script_name(character: str) -> str:
    if not character.isalpha():
        return "NON_LETTER"

    try:
        name = unicodedata.name(character)
    except ValueError:
        return "UNKNOWN"

    if "LATIN" in name:
        return "LATIN"

    if "CYRILLIC" in name:
        return "CYRILLIC"

    return "OTHER"


def contains_named_flag(
    raw_flags: str,
    expected_flags: set[str],
) -> bool:
    raw_upper = str(raw_flags or "").upper()

    return any(
        flag in raw_upper
        for flag in expected_flags
    )


def discover_batch_files() -> list[Path]:
    pattern = re.compile(
        r"^batch-\d{2}\.csv$"
    )

    files = sorted(
        path
        for path in INPUT_DIR.glob("batch-*.csv")
        if pattern.fullmatch(path.name)
    )

    if len(files) != 28:
        raise ValueError(
            "Expected exactly 28 active review batches, "
            f"found {len(files)}."
        )

    return files


def read_review_rows(
    batch_files: list[Path],
) -> list[dict[str, str]]:
    required_columns = {
        "sample_id",
        "part_number",
        "source_record_id",
        "sentence_id",
        "sentence_text",
        "human_health_label",
        "boundary_flags",
        "parent_review_flags",
        "sentence_character_count",
        "sentence_token_count",
    }

    all_rows: list[dict[str, str]] = []

    for batch_path in batch_files:
        with batch_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(
                reader.fieldnames or []
            )

            missing = required_columns - fieldnames

            if missing:
                raise ValueError(
                    f"{batch_path.name} is missing "
                    f"columns: {sorted(missing)}"
                )

            rows = list(reader)

        if len(rows) != 25:
            raise ValueError(
                f"{batch_path.name} has {len(rows)} rows; "
                "expected 25."
            )

        all_rows.extend(rows)

    return all_rows


def validate_review_rows(
    rows: list[dict[str, str]],
) -> None:
    if len(rows) != 700:
        raise ValueError(
            f"Expected 700 reviewed rows, found {len(rows)}."
        )

    sample_ids = [
        str(row["sample_id"]).strip()
        for row in rows
    ]

    if len(sample_ids) != len(set(sample_ids)):
        duplicates = [
            sample_id
            for sample_id, count
            in Counter(sample_ids).items()
            if count > 1
        ]

        raise ValueError(
            "Duplicate sample IDs found: "
            f"{duplicates[:10]}"
        )

    empty_sentences = [
        row["sample_id"]
        for row in rows
        if not str(row["sentence_text"]).strip()
    ]

    if empty_sentences:
        raise ValueError(
            "Empty sentence_text values found: "
            f"{empty_sentences[:10]}"
        )

    invalid_labels = [
        {
            "sample_id": row["sample_id"],
            "label": row["human_health_label"],
        }
        for row in rows
        if str(
            row["human_health_label"]
        ).strip() not in ALLOWED_LABELS
    ]

    if invalid_labels:
        raise ValueError(
            "Invalid labels found: "
            f"{invalid_labels[:10]}"
        )


def build_feature_row(
    row: dict[str, str],
) -> dict[str, Any]:
    text = str(
        row["sentence_text"] or ""
    ).strip()

    tokens = text.split()

    actual_character_count = len(text)
    actual_token_count = len(tokens)

    nonspace_characters = [
        character
        for character in text
        if not character.isspace()
    ]

    letters = [
        character
        for character in text
        if character.isalpha()
    ]

    digit_count = sum(
        character.isdigit()
        for character in text
    )

    numeric_token_count = sum(
        any(character.isdigit() for character in token)
        for token in tokens
    )

    uppercase_count = sum(
        character.isupper()
        for character in letters
    )

    scripts = Counter(
        script_name(character)
        for character in letters
    )

    latin_count = scripts["LATIN"]
    cyrillic_count = scripts["CYRILLIC"]
    other_script_count = (
        scripts["OTHER"]
        + scripts["UNKNOWN"]
    )

    letter_count = len(letters)
    nonspace_count = len(
        nonspace_characters
    )

    digit_ratio = (
        digit_count / nonspace_count
        if nonspace_count
        else 0.0
    )

    numeric_token_ratio = (
        numeric_token_count / actual_token_count
        if actual_token_count
        else 0.0
    )

    uppercase_ratio = (
        uppercase_count / letter_count
        if letter_count
        else 0.0
    )

    latin_ratio = (
        latin_count / letter_count
        if letter_count
        else 0.0
    )

    cyrillic_ratio = (
        cyrillic_count / letter_count
        if letter_count
        else 0.0
    )

    stripped = text.rstrip()

    last_character = (
        stripped[-1]
        if stripped
        else ""
    )

    boundary_flags = str(
        row.get("boundary_flags", "")
    )

    parent_flags = str(
        row.get("parent_review_flags", "")
    )

    url_signal = bool(
        URL_RE.search(text)
    )

    email_signal = bool(
        EMAIL_RE.search(text)
    )

    heading_like = bool(
        actual_token_count <= 12
        and HEADING_END_RE.search(text)
    )

    return {
        "sample_id": str(
            row["sample_id"]
        ).strip(),
        "part_number": str(
            row["part_number"]
        ).strip(),
        "source_record_id": str(
            row["source_record_id"]
        ).strip(),
        "sentence_id": str(
            row["sentence_id"]
        ).strip(),
        "human_health_label": str(
            row["human_health_label"]
        ).strip(),
        "boundary_flags": boundary_flags,
        "parent_review_flags": parent_flags,
        "sentence_text": text,
        "stored_character_count": safe_int(
            row["sentence_character_count"]
        ),
        "actual_character_count": (
            actual_character_count
        ),
        "character_count_matches": (
            safe_int(
                row["sentence_character_count"]
            )
            == actual_character_count
        ),
        "stored_token_count": safe_int(
            row["sentence_token_count"]
        ),
        "actual_token_count": actual_token_count,
        "token_count_matches": (
            safe_int(
                row["sentence_token_count"]
            )
            == actual_token_count
        ),
        "nonspace_character_count": nonspace_count,
        "letter_count": letter_count,
        "digit_count": digit_count,
        "digit_ratio": round(
            digit_ratio,
            6,
        ),
        "numeric_token_count": (
            numeric_token_count
        ),
        "numeric_token_ratio": round(
            numeric_token_ratio,
            6,
        ),
        "uppercase_count": uppercase_count,
        "uppercase_ratio": round(
            uppercase_ratio,
            6,
        ),
        "latin_letter_count": latin_count,
        "latin_ratio": round(
            latin_ratio,
            6,
        ),
        "cyrillic_letter_count": (
            cyrillic_count
        ),
        "cyrillic_ratio": round(
            cyrillic_ratio,
            6,
        ),
        "other_script_letter_count": (
            other_script_count
        ),
        "signal_mixed_latin_cyrillic": (
            latin_count > 0
            and cyrillic_count > 0
        ),
        "signal_url": url_signal,
        "signal_email": email_signal,
        "signal_url_or_email": (
            url_signal or email_signal
        ),
        "signal_leading_list_marker": bool(
            LEADING_LIST_RE.search(text)
        ),
        "signal_table_delimiter": bool(
            TABLE_DELIMITER_RE.search(text)
        ),
        "signal_heading_like": heading_like,
        "signal_high_digit_ratio_20pct": (
            digit_ratio >= 0.20
        ),
        "signal_high_digit_ratio_30pct": (
            digit_ratio >= 0.30
        ),
        "signal_high_numeric_token_ratio_30pct": (
            numeric_token_ratio >= 0.30
        ),
        "signal_high_uppercase_ratio_70pct": (
            letter_count >= 5
            and uppercase_ratio >= 0.70
        ),
        "signal_no_terminal_punctuation": (
            last_character
            not in TERMINAL_PUNCTUATION
        ),
        "signal_very_short_3_tokens": (
            actual_token_count <= 3
        ),
        "signal_short_5_tokens": (
            actual_token_count <= 5
        ),
        "signal_long_over_35_tokens": (
            actual_token_count > 35
        ),
        "signal_long_over_240_chars": (
            actual_character_count > 240
        ),
        "signal_boundary_high_risk_any": (
            contains_named_flag(
                boundary_flags,
                HIGH_RISK_BOUNDARY_FLAGS,
            )
        ),
        "signal_apostrophe_gap_any": (
            contains_named_flag(
                parent_flags,
                APOSTROPHE_PARENT_FLAGS,
            )
        ),
        "signal_quote_or_speech_continuation": (
            "QUOTE_OR_SPEECH_CONTINUATION"
            in boundary_flags.upper()
        ),
        "signal_dotted_name_or_domain": (
            "DOTTED_NAME_OR_DOMAIN_PROTECTED"
            in boundary_flags.upper()
        ),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_label_counts(
    feature_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts = Counter(
        row["human_health_label"]
        for row in feature_rows
    )

    total = len(feature_rows)

    return [
        {
            "human_health_label": label,
            "count": counts[label],
            "percentage": percentage(
                counts[label],
                total,
            ),
        }
        for label in sorted(ALLOWED_LABELS)
    ]


def build_signal_summary(
    feature_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for signal in BOOLEAN_SIGNAL_FIELDS:
        flagged_rows = [
            row
            for row in feature_rows
            if is_true(row[signal])
        ]

        flagged_count = len(flagged_rows)

        label_counts = Counter(
            row["human_health_label"]
            for row in flagged_rows
        )

        usable_count = sum(
            label_counts[label]
            for label in USABLE_LABELS
        )

        harmful_count = sum(
            label_counts[label]
            for label
            in HARMFUL_CLEAN_LANE_LABELS
        )

        results.append(
            {
                "signal": signal,
                "flagged_rows": flagged_count,
                "coverage_pct": percentage(
                    flagged_count,
                    len(feature_rows),
                ),
                "usable_rows": usable_count,
                "usable_rate_pct": percentage(
                    usable_count,
                    flagged_count,
                ),
                "healthy_rows": (
                    label_counts["HEALTHY"]
                ),
                "healthy_rate_pct": percentage(
                    label_counts["HEALTHY"],
                    flagged_count,
                ),
                "minor_damage_rows": (
                    label_counts[
                        "MINOR_DAMAGE_BUT_USABLE"
                    ]
                ),
                "format_or_list_rows": (
                    label_counts["FORMAT_OR_LIST"]
                ),
                "fragment_rows": (
                    label_counts["FRAGMENT"]
                ),
                "heavy_corruption_rows": (
                    label_counts["HEAVY_CORRUPTION"]
                ),
                "mixed_or_foreign_rows": (
                    label_counts[
                        "MIXED_OR_FOREIGN_TEXT"
                    ]
                ),
                "uncertain_rows": (
                    label_counts["UNCERTAIN"]
                ),
                "harmful_clean_lane_rows": (
                    harmful_count
                ),
                "harmful_clean_lane_rate_pct": (
                    percentage(
                        harmful_count,
                        flagged_count,
                    )
                ),
            }
        )

    return sorted(
        results,
        key=lambda row: (
            -row["harmful_clean_lane_rate_pct"],
            -row["flagged_rows"],
            row["signal"],
        ),
    )


def percentile(
    values: list[float],
    fraction: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    position = (
        len(ordered) - 1
    ) * fraction

    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(ordered) - 1,
    )

    weight = position - lower_index

    return (
        ordered[lower_index]
        * (1.0 - weight)
        + ordered[upper_index]
        * weight
    )


def build_numeric_summary(
    feature_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in feature_rows:
        grouped[
            row["human_health_label"]
        ].append(row)

    results: list[dict[str, Any]] = []

    for label in sorted(grouped):
        rows = grouped[label]

        for feature in NUMERIC_FEATURE_FIELDS:
            values = [
                float(row[feature])
                for row in rows
            ]

            results.append(
                {
                    "human_health_label": label,
                    "feature": feature,
                    "count": len(values),
                    "minimum": round(
                        min(values),
                        6,
                    ),
                    "p25": round(
                        percentile(values, 0.25),
                        6,
                    ),
                    "median": round(
                        median(values),
                        6,
                    ),
                    "mean": round(
                        mean(values),
                        6,
                    ),
                    "p75": round(
                        percentile(values, 0.75),
                        6,
                    ),
                    "maximum": round(
                        max(values),
                        6,
                    ),
                }
            )

    return results


def build_signal_examples(
    feature_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for signal in BOOLEAN_SIGNAL_FIELDS:
        for label in sorted(ALLOWED_LABELS):
            matching = [
                row
                for row in feature_rows
                if (
                    row["human_health_label"] == label
                    and is_true(row[signal])
                )
            ]

            for row in matching[:3]:
                results.append(
                    {
                        "signal": signal,
                        "human_health_label": label,
                        "sample_id": row["sample_id"],
                        "part_number": row["part_number"],
                        "boundary_flags": (
                            row["boundary_flags"]
                        ),
                        "parent_review_flags": (
                            row["parent_review_flags"]
                        ),
                        "sentence_text": (
                            row["sentence_text"]
                        ),
                    }
                )

    return results


def write_report(
    label_counts: list[dict[str, Any]],
    signal_summary: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
) -> None:
    character_mismatches = sum(
        not is_true(
            row["character_count_matches"]
        )
        for row in feature_rows
    )

    token_mismatches = sum(
        not is_true(
            row["token_count_matches"]
        )
        for row in feature_rows
    )

    lines = [
        "# Routing Signal Audit v1",
        "",
        "## Status",
        "",
        "`MEASUREMENT_ONLY_NOT_ROUTING`",
        "",
        "This audit measures candidate routing signals "
        "against the 700 manual health labels.",
        "",
        "It does not approve thresholds and does not "
        "route the full corpus.",
        "",
        "## Validation",
        "",
        "- Active review batches: 28",
        "- Reviewed rows: 700",
        "- Duplicate sample IDs: 0",
        "- Invalid labels: 0",
        f"- Character-count mismatches: "
        f"{character_mismatches}",
        f"- Token-count mismatches: "
        f"{token_mismatches}",
        "",
        "## Manual-label distribution",
        "",
        "| Label | Count | Percentage |",
        "|---|---:|---:|",
    ]

    for row in label_counts:
        lines.append(
            f"| {row['human_health_label']} "
            f"| {row['count']} "
            f"| {row['percentage']}% |"
        )

    lines.extend(
        [
            "",
            "## Candidate boolean signals",
            "",
            "| Signal | Flagged | Coverage | "
            "Usable rate | Harmful clean-lane rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for row in signal_summary:
        lines.append(
            f"| {row['signal']} "
            f"| {row['flagged_rows']} "
            f"| {row['coverage_pct']}% "
            f"| {row['usable_rate_pct']}% "
            f"| {row['harmful_clean_lane_rate_pct']}% |"
        )

    lines.extend(
        [
            "",
            "## Interpretation warning",
            "",
            "A high harmful rate does not automatically "
            "make a signal a deletion rule.",
            "",
            "A high usable rate does not automatically "
            "make a signal a clean-acceptance rule.",
            "",
            "Signals must be combined and false-positive "
            "examples must be inspected before deployment.",
            "",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    batch_files = discover_batch_files()
    review_rows = read_review_rows(
        batch_files
    )
    validate_review_rows(review_rows)

    feature_rows = [
        build_feature_row(row)
        for row in review_rows
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_counts = build_label_counts(
        feature_rows
    )

    signal_summary = build_signal_summary(
        feature_rows
    )

    numeric_summary = build_numeric_summary(
        feature_rows
    )

    signal_examples = build_signal_examples(
        feature_rows
    )

    write_csv(
        FEATURE_ROWS_PATH,
        feature_rows,
        FEATURE_FIELDNAMES,
    )

    write_csv(
        LABEL_COUNTS_PATH,
        label_counts,
        [
            "human_health_label",
            "count",
            "percentage",
        ],
    )

    signal_summary_fields = list(
        signal_summary[0]
    )

    write_csv(
        SIGNAL_SUMMARY_PATH,
        signal_summary,
        signal_summary_fields,
    )

    write_csv(
        NUMERIC_SUMMARY_PATH,
        numeric_summary,
        list(numeric_summary[0]),
    )

    write_csv(
        SIGNAL_EXAMPLES_PATH,
        signal_examples,
        [
            "signal",
            "human_health_label",
            "sample_id",
            "part_number",
            "boundary_flags",
            "parent_review_flags",
            "sentence_text",
        ],
    )

    write_report(
        label_counts=label_counts,
        signal_summary=signal_summary,
        feature_rows=feature_rows,
    )

    summary = {
        "status": (
            "MEASUREMENT_ONLY_NOT_ROUTING"
        ),
        "active_batch_files": len(
            batch_files
        ),
        "review_rows": len(feature_rows),
        "duplicate_sample_ids": 0,
        "invalid_labels": 0,
        "character_count_mismatches": sum(
            not is_true(
                row["character_count_matches"]
            )
            for row in feature_rows
        ),
        "token_count_mismatches": sum(
            not is_true(
                row["token_count_matches"]
            )
            for row in feature_rows
        ),
        "manual_label_counts": {
            row["human_health_label"]: (
                row["count"]
            )
            for row in label_counts
        },
        "output_files": [
            str(
                FEATURE_ROWS_PATH.relative_to(
                    ROOT
                )
            ),
            str(
                LABEL_COUNTS_PATH.relative_to(
                    ROOT
                )
            ),
            str(
                SIGNAL_SUMMARY_PATH.relative_to(
                    ROOT
                )
            ),
            str(
                NUMERIC_SUMMARY_PATH.relative_to(
                    ROOT
                )
            ),
            str(
                SIGNAL_EXAMPLES_PATH.relative_to(
                    ROOT
                )
            ),
            str(
                REPORT_PATH.relative_to(ROOT)
            ),
        ],
    }

    SUMMARY_JSON_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("--- ROUTING SIGNAL AUDIT BUILT ---")
    print(
        f"active_batch_files: {len(batch_files)}"
    )
    print(
        f"review_rows: {len(feature_rows)}"
    )
    print("duplicate_sample_ids: 0")
    print("invalid_labels: 0")

    print(
        "character_count_mismatches:",
        summary[
            "character_count_mismatches"
        ],
    )

    print(
        "token_count_mismatches:",
        summary[
            "token_count_mismatches"
        ],
    )

    print()
    print("Manual labels:")

    for row in label_counts:
        print(
            f"- {row['human_health_label']}: "
            f"{row['count']} "
            f"({row['percentage']}%)"
        )

    print()
    print(
        "Candidate signal summary "
        "(measurement only):"
    )

    for row in signal_summary:
        print(
            f"- {row['signal']}: "
            f"flagged={row['flagged_rows']}, "
            f"coverage={row['coverage_pct']}%, "
            f"usable={row['usable_rate_pct']}%, "
            f"harmful={row['harmful_clean_lane_rate_pct']}%"
        )

    print()
    print("Created files:")

    for path in [
        FEATURE_ROWS_PATH,
        LABEL_COUNTS_PATH,
        SIGNAL_SUMMARY_PATH,
        NUMERIC_SUMMARY_PATH,
        SIGNAL_EXAMPLES_PATH,
        REPORT_PATH,
        SUMMARY_JSON_PATH,
    ]:
        print(
            "-",
            path.relative_to(ROOT),
        )

    print()
    print(
        "ROUTING SIGNAL AUDIT: PASS"
    )
    print(
        "STATUS: MEASUREMENT_ONLY_NOT_ROUTING"
    )


if __name__ == "__main__":
    main()
