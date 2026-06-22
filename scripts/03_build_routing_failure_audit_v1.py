#!/usr/bin/env python3

"""
Analyse harmful rows missed by the current candidate routing signals.

This is an exploratory development audit.

It does not route the full corpus and does not approve any new
feature as a production routing rule.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import csv
import re


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "outputs"
    / "03_sentence_inventory"
    / "routing_signal_audit_v1"
    / "review_rows_with_features_v1.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "03_sentence_inventory"
    / "routing_failure_audit_v1"
)

UNCUGHT_PATH = (
    OUTPUT_DIR
    / "currently_uncaught_harmful_rows_v1.csv"
)

FALSE_QUARANTINE_PATH = (
    OUTPUT_DIR
    / "currently_flagged_usable_rows_v1.csv"
)

EXPLORATORY_SUMMARY_PATH = (
    OUTPUT_DIR
    / "exploratory_signal_summary_v1.csv"
)

TOKEN_CONTRAST_PATH = (
    OUTPUT_DIR
    / "token_label_contrast_v1.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "routing_failure_audit_report_v1.md"
)


HARMFUL_LABELS = {
    "FORMAT_OR_LIST",
    "FRAGMENT",
    "HEAVY_CORRUPTION",
    "MIXED_OR_FOREIGN_TEXT",
}

USABLE_LABELS = {
    "HEALTHY",
    "MINOR_DAMAGE_BUT_USABLE",
}

CURRENT_NEGATIVE_SIGNALS = [
    "signal_high_uppercase_ratio_70pct",
    "signal_url_or_email",
    "signal_leading_list_marker",
    "signal_very_short_3_tokens",
    "signal_boundary_high_risk_any",
    "signal_dotted_name_or_domain",
    "signal_long_over_35_tokens",
    "signal_long_over_240_chars",
]

UZBEK_CORE_WORDS = {
    "u",
    "ular",
    "men",
    "sen",
    "biz",
    "siz",
    "bu",
    "shu",
    "o'sha",
    "bir",
    "va",
    "ham",
    "esa",
    "ammo",
    "lekin",
    "yoki",
    "agar",
    "chunki",
    "uchun",
    "bilan",
    "orqali",
    "haqida",
    "bo'yicha",
    "kabi",
    "deb",
    "deya",
    "edi",
    "ekan",
    "kerak",
    "mumkin",
    "uning",
    "o'z",
    "o'zi",
    "bo'lib",
    "bo'lgan",
    "bo'ladi",
    "qiladi",
    "qilgan",
    "qilish",
    "hamda",
    "so'ng",
    "keyin",
    "oldin",
}

TOKEN_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿʻʼ‘’'`-]+"
)

ROMAN_TOKEN_RE = re.compile(
    r"^[IVXLCDM]+[.)]?$",
    re.IGNORECASE,
)

INTERNAL_TERMINAL_RE = re.compile(
    r"[?!…](?=.+)"
)


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


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


def normalize_token(token: str) -> str:
    return (
        token.lower()
        .replace("ʻ", "'")
        .replace("ʼ", "'")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("`", "'")
        .strip("-'")
    )


def tokenize(text: str) -> list[str]:
    return [
        normalize_token(token)
        for token in TOKEN_RE.findall(text)
        if normalize_token(token)
    ]


def starts_with_lowercase(text: str) -> bool:
    for character in text.lstrip():
        if character.isalpha():
            return character.islower()

        if character in "\"'“‘«([{":
            continue

        return False

    return False


def build_exploratory_features(
    row: dict[str, str],
) -> dict[str, Any]:
    text = row["sentence_text"].strip()
    tokens = tokenize(text)

    first_token = (
        tokens[0]
        if tokens
        else ""
    )

    single_letter_tokens = [
        token
        for token in tokens
        if len(token) == 1
    ]

    long_tokens_20 = [
        token
        for token in tokens
        if len(token) >= 20
    ]

    long_tokens_30 = [
        token
        for token in tokens
        if len(token) >= 30
    ]

    core_count = sum(
        token in UZBEK_CORE_WORDS
        for token in tokens
    )

    core_ratio = (
        core_count / len(tokens)
        if tokens
        else 0.0
    )

    apostrophe_tokens = sum(
        "'" in token
        for token in tokens
    )

    current_negative_any = any(
        truth(row[signal])
        for signal in CURRENT_NEGATIVE_SIGNALS
    )

    return {
        **row,
        "current_negative_any": (
            current_negative_any
        ),
        "explore_starts_lowercase": (
            starts_with_lowercase(text)
        ),
        "explore_first_token_single_letter": (
            len(first_token) == 1
        ),
        "explore_first_token_roman_like": bool(
            ROMAN_TOKEN_RE.fullmatch(
                first_token
            )
        ),
        "explore_internal_question_or_exclamation": bool(
            INTERNAL_TERMINAL_RE.search(text)
        ),
        "explore_contains_colon": (
            ":" in text
            or "：" in text
        ),
        "explore_many_commas_4plus": (
            text.count(",") >= 4
        ),
        "explore_single_letter_tokens_2plus": (
            len(single_letter_tokens) >= 2
        ),
        "explore_token_length_20plus": bool(
            long_tokens_20
        ),
        "explore_token_length_30plus": bool(
            long_tokens_30
        ),
        "explore_zero_uzbek_core_words": (
            core_count == 0
        ),
        "explore_low_uzbek_core_ratio_5pct": (
            len(tokens) >= 5
            and core_ratio <= 0.05
        ),
        "explore_zero_apostrophe_tokens": (
            apostrophe_tokens == 0
        ),
        "explore_core_word_count": (
            core_count
        ),
        "explore_core_word_ratio": round(
            core_ratio,
            6,
        ),
        "explore_apostrophe_token_count": (
            apostrophe_tokens
        ),
        "explore_first_token": first_token,
        "explore_single_letter_token_count": (
            len(single_letter_tokens)
        ),
        "explore_long_token_20_count": (
            len(long_tokens_20)
        ),
    }


EXPLORATORY_BOOLEAN_SIGNALS = [
    "explore_starts_lowercase",
    "explore_first_token_single_letter",
    "explore_first_token_roman_like",
    "explore_internal_question_or_exclamation",
    "explore_contains_colon",
    "explore_many_commas_4plus",
    "explore_single_letter_tokens_2plus",
    "explore_token_length_20plus",
    "explore_token_length_30plus",
    "explore_zero_uzbek_core_words",
    "explore_low_uzbek_core_ratio_5pct",
    "explore_zero_apostrophe_tokens",
]


def read_rows() -> list[dict[str, str]]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 700:
        raise ValueError(
            f"Expected 700 rows, found {len(rows)}."
        )

    missing_signals = [
        signal
        for signal in CURRENT_NEGATIVE_SIGNALS
        if signal not in rows[0]
    ]

    if missing_signals:
        raise ValueError(
            "Input is missing signals: "
            f"{missing_signals}"
        )

    return rows


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


def build_signal_summary(
    rows: list[dict[str, Any]],
    uncaught_harmful: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []

    for signal in EXPLORATORY_BOOLEAN_SIGNALS:
        flagged = [
            row
            for row in rows
            if truth(row[signal])
        ]

        harmful = [
            row
            for row in flagged
            if row["human_health_label"]
            in HARMFUL_LABELS
        ]

        usable = [
            row
            for row in flagged
            if row["human_health_label"]
            in USABLE_LABELS
        ]

        newly_caught = [
            row
            for row in uncaught_harmful
            if truth(row[signal])
        ]

        label_counts = Counter(
            row["human_health_label"]
            for row in flagged
        )

        results.append(
            {
                "signal": signal,
                "flagged_all_rows": len(flagged),
                "flagged_harmful_rows": len(harmful),
                "flagged_usable_rows": len(usable),
                "harmful_precision_pct": percentage(
                    len(harmful),
                    len(flagged),
                ),
                "currently_uncaught_harmful_matched": (
                    len(newly_caught)
                ),
                "currently_uncaught_harmful_coverage_pct": (
                    percentage(
                        len(newly_caught),
                        len(uncaught_harmful),
                    )
                ),
                "healthy_rows": (
                    label_counts["HEALTHY"]
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
            }
        )

    return sorted(
        results,
        key=lambda row: (
            -row[
                "currently_uncaught_harmful_matched"
            ],
            -row["harmful_precision_pct"],
            row["signal"],
        ),
    )


def build_token_contrast(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    token_label_rows: dict[
        str,
        Counter,
    ] = defaultdict(Counter)

    for row in rows:
        label = row["human_health_label"]

        unique_tokens = set(
            tokenize(row["sentence_text"])
        )

        for token in unique_tokens:
            if len(token) < 2:
                continue

            token_label_rows[token][label] += 1

    results = []

    for token, counts in token_label_rows.items():
        total_rows = sum(counts.values())

        if total_rows < 2:
            continue

        usable_rows = sum(
            counts[label]
            for label in USABLE_LABELS
        )

        harmful_rows = sum(
            counts[label]
            for label in HARMFUL_LABELS
        )

        results.append(
            {
                "token": token,
                "total_rows": total_rows,
                "usable_rows": usable_rows,
                "harmful_rows": harmful_rows,
                "harmful_rate_pct": percentage(
                    harmful_rows,
                    total_rows,
                ),
                "healthy_rows": (
                    counts["HEALTHY"]
                ),
                "minor_damage_rows": (
                    counts[
                        "MINOR_DAMAGE_BUT_USABLE"
                    ]
                ),
                "format_or_list_rows": (
                    counts["FORMAT_OR_LIST"]
                ),
                "fragment_rows": (
                    counts["FRAGMENT"]
                ),
                "heavy_corruption_rows": (
                    counts["HEAVY_CORRUPTION"]
                ),
                "mixed_or_foreign_rows": (
                    counts[
                        "MIXED_OR_FOREIGN_TEXT"
                    ]
                ),
            }
        )

    return sorted(
        results,
        key=lambda row: (
            -row["mixed_or_foreign_rows"],
            -row["harmful_rate_pct"],
            -row["total_rows"],
            row["token"],
        ),
    )


def write_report(
    rows: list[dict[str, Any]],
    uncaught_harmful: list[dict[str, Any]],
    flagged_usable: list[dict[str, Any]],
    signal_summary: list[dict[str, Any]],
) -> None:
    harmful_rows = [
        row
        for row in rows
        if row["human_health_label"]
        in HARMFUL_LABELS
    ]

    caught_harmful = [
        row
        for row in harmful_rows
        if truth(row["current_negative_any"])
    ]

    usable_rows = [
        row
        for row in rows
        if row["human_health_label"]
        in USABLE_LABELS
    ]

    lines = [
        "# Routing Failure Audit v1",
        "",
        "## Status",
        "",
        "`EXPLORATORY_DEVELOPMENT_ONLY`",
        "",
        "This audit studies failures of the current "
        "candidate negative signals.",
        "",
        "It does not approve production routing rules.",
        "",
        "## Current negative-signal union",
        "",
        f"- Total harmful rows: {len(harmful_rows)}",
        f"- Harmful rows caught: {len(caught_harmful)}",
        f"- Harmful recall: "
        f"{percentage(len(caught_harmful), len(harmful_rows))}%",
        f"- Harmful rows missed: {len(uncaught_harmful)}",
        f"- Harmful miss rate: "
        f"{percentage(len(uncaught_harmful), len(harmful_rows))}%",
        f"- Total usable rows: {len(usable_rows)}",
        f"- Usable rows flagged by the union: "
        f"{len(flagged_usable)}",
        f"- Usable flag rate: "
        f"{percentage(len(flagged_usable), len(usable_rows))}%",
        "",
        "## Exploratory signals",
        "",
        "| Signal | New missed-harmful matches | "
        "Missed-harmful coverage | Harmful precision | "
        "Usable rows flagged |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in signal_summary:
        lines.append(
            f"| {row['signal']} "
            f"| {row['currently_uncaught_harmful_matched']} "
            f"| {row['currently_uncaught_harmful_coverage_pct']}% "
            f"| {row['harmful_precision_pct']}% "
            f"| {row['flagged_usable_rows']} |"
        )

    lines.extend(
        [
            "",
            "## Warning",
            "",
            "These exploratory features were studied using "
            "manual labels from the same 700-row development set.",
            "",
            "They must not be treated as unbiased production rules.",
            "",
            "After policy development, a fresh unseen manual-review "
            "sample is required for final validation.",
            "",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    source_rows = read_rows()

    rows = [
        build_exploratory_features(row)
        for row in source_rows
    ]

    harmful_rows = [
        row
        for row in rows
        if row["human_health_label"]
        in HARMFUL_LABELS
    ]

    uncaught_harmful = [
        row
        for row in harmful_rows
        if not truth(
            row["current_negative_any"]
        )
    ]

    flagged_usable = [
        row
        for row in rows
        if (
            row["human_health_label"]
            in USABLE_LABELS
            and truth(
                row["current_negative_any"]
            )
        )
    ]

    signal_summary = build_signal_summary(
        rows,
        uncaught_harmful,
    )

    token_contrast = build_token_contrast(
        rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_fieldnames = list(rows[0])

    write_csv(
        UNCUGHT_PATH,
        uncaught_harmful,
        feature_fieldnames,
    )

    write_csv(
        FALSE_QUARANTINE_PATH,
        flagged_usable,
        feature_fieldnames,
    )

    write_csv(
        EXPLORATORY_SUMMARY_PATH,
        signal_summary,
        list(signal_summary[0]),
    )

    write_csv(
        TOKEN_CONTRAST_PATH,
        token_contrast,
        list(token_contrast[0]),
    )

    write_report(
        rows=rows,
        uncaught_harmful=uncaught_harmful,
        flagged_usable=flagged_usable,
        signal_summary=signal_summary,
    )

    harmful_total = len(harmful_rows)
    harmful_caught = (
        harmful_total
        - len(uncaught_harmful)
    )

    print("--- ROUTING FAILURE AUDIT BUILT ---")
    print(f"review_rows: {len(rows)}")
    print(f"harmful_total: {harmful_total}")
    print(
        f"harmful_caught_by_current_union: "
        f"{harmful_caught}"
    )
    print(
        "current_union_harmful_recall_pct:",
        percentage(
            harmful_caught,
            harmful_total,
        ),
    )
    print(
        f"currently_uncaught_harmful: "
        f"{len(uncaught_harmful)}"
    )
    print(
        f"usable_flagged_by_current_union: "
        f"{len(flagged_usable)}"
    )

    print()
    print(
        "Exploratory signals "
        "(not approved as rules):"
    )

    for row in signal_summary:
        print(
            f"- {row['signal']}: "
            f"new_harmful="
            f"{row['currently_uncaught_harmful_matched']}, "
            f"new_coverage="
            f"{row['currently_uncaught_harmful_coverage_pct']}%, "
            f"harmful_precision="
            f"{row['harmful_precision_pct']}%, "
            f"usable_flagged="
            f"{row['flagged_usable_rows']}"
        )

    print()
    print("Created files:")
    print(f"- {UNCUGHT_PATH.relative_to(ROOT)}")
    print(
        f"- {FALSE_QUARANTINE_PATH.relative_to(ROOT)}"
    )
    print(
        f"- {EXPLORATORY_SUMMARY_PATH.relative_to(ROOT)}"
    )
    print(
        f"- {TOKEN_CONTRAST_PATH.relative_to(ROOT)}"
    )
    print(f"- {REPORT_PATH.relative_to(ROOT)}")

    print()
    print("ROUTING FAILURE AUDIT: PASS")
    print(
        "STATUS: EXPLORATORY_DEVELOPMENT_ONLY"
    )


if __name__ == "__main__":
    main()
