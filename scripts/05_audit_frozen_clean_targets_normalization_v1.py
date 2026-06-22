from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from typing import Dict, List


INPUT_PATH = Path(
    "outputs/05_synthetic_typo_generation/v10_deterministic_sweep/"
    "frozen_clean_targets_v1.csv"
)

OUT_DIR = Path(
    "outputs/05_synthetic_typo_generation/"
    "normalization_audit_frozen_clean_targets_v1"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_OUT = OUT_DIR / "frozen_clean_targets_normalization_audit_all_v1.csv"
FLAGGED_OUT = OUT_DIR / "frozen_clean_targets_normalization_flagged_v1.csv"
SUMMARY_OUT = OUT_DIR / "frozen_clean_targets_normalization_summary_v1.md"


TEXT_COLUMNS = ["sentence_text", "target", "sentence", "text"]

APOSTROPHE_VARIANTS = {
    "‘": "'",
    "’": "'",
    "ʻ": "'",
    "ʼ": "'",
    "`": "'",
    "´": "'",
    "ʹ": "'",
}

DOUBLE_QUOTE_VARIANTS = {
    "“": '"',
    "”": '"',
    "„": '"',
    "«": '"',
    "»": '"',
}

SPACE_VARIANTS = {
    "\u00A0": " ",   # non-breaking space
    "\u2007": " ",
    "\u202F": " ",
}

INVISIBLE_CHARS = [
    "\u200B",  # zero-width space
    "\u200C",  # zero-width non-joiner
    "\u200D",  # zero-width joiner
    "\uFEFF",  # BOM
]

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
MULTISPACE_RE = re.compile(r" {2,}")


def get_text(row: Dict[str, str]) -> str:
    for col in TEXT_COLUMNS:
        if col in row and row[col]:
            return row[col]
    return ""


def deterministic_normalize(text: str) -> str:
    s = unicodedata.normalize("NFC", text)

    for bad, good in SPACE_VARIANTS.items():
        s = s.replace(bad, good)

    for ch in INVISIBLE_CHARS:
        s = s.replace(ch, "")

    for bad, good in APOSTROPHE_VARIANTS.items():
        s = s.replace(bad, good)

    for bad, good in DOUBLE_QUOTE_VARIANTS.items():
        s = s.replace(bad, good)

    s = MULTISPACE_RE.sub(" ", s)
    s = s.strip()

    return s


def flags_for(text: str, normalized: str) -> List[str]:
    flags = []

    if text != unicodedata.normalize("NFC", text):
        flags.append("unicode_nfc_changes")

    for bad in APOSTROPHE_VARIANTS:
        if bad in text:
            if bad == "`":
                flags.append("contains_backtick_apostrophe")
            else:
                flags.append(f"contains_apostrophe_variant_U+{ord(bad):04X}")

    for bad in DOUBLE_QUOTE_VARIANTS:
        if bad in text:
            flags.append(f"contains_double_quote_variant_U+{ord(bad):04X}")

    for bad in SPACE_VARIANTS:
        if bad in text:
            flags.append(f"contains_space_variant_U+{ord(bad):04X}")

    for ch in INVISIBLE_CHARS:
        if ch in text:
            flags.append(f"contains_invisible_char_U+{ord(ch):04X}")

    if CYRILLIC_RE.search(text):
        flags.append("contains_cyrillic_letters")

    if MULTISPACE_RE.search(text):
        flags.append("contains_multiple_spaces")

    if text != text.strip():
        flags.append("leading_or_trailing_space")

    if normalized != text:
        flags.append("normalizer_would_change_text")

    return flags


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    with INPUT_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    flagged_rows = []

    for i, row in enumerate(rows, start=1):
        text = get_text(row)
        normalized = deterministic_normalize(text)
        flags = flags_for(text, normalized)

        out = {
            "row_num": i,
            "review_id": row.get("review_id", ""),
            "source_record_id": row.get("source_record_id", ""),
            "sentence_id": row.get("sentence_id", ""),
            "sentence_text": text,
            "normalized_candidate": normalized,
            "changed_by_normalizer": str(text != normalized),
            "flag_count": len(flags),
            "flags": "|".join(flags),
        }

        out_rows.append(out)

        if flags:
            flagged_rows.append(out)

    fieldnames = [
        "row_num",
        "review_id",
        "source_record_id",
        "sentence_id",
        "sentence_text",
        "normalized_candidate",
        "changed_by_normalizer",
        "flag_count",
        "flags",
    ]

    with ALL_OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    with FLAGGED_OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(flagged_rows)

    flag_counter: Dict[str, int] = {}
    for r in flagged_rows:
        for flag in r["flags"].split("|"):
            if flag:
                flag_counter[flag] = flag_counter.get(flag, 0) + 1

    summary = []
    summary.append("# Frozen Clean Targets Normalization Audit v1")
    summary.append("")
    summary.append("## Input")
    summary.append(f"- input_file: `{INPUT_PATH}`")
    summary.append("")
    summary.append("## Counts")
    summary.append(f"- total_rows: `{len(rows)}`")
    summary.append(f"- flagged_rows: `{len(flagged_rows)}`")
    summary.append(f"- clean_normalization_rows: `{len(rows) - len(flagged_rows)}`")
    summary.append("")
    summary.append("## Flag counts")

    if flag_counter:
        for k, v in sorted(flag_counter.items(), key=lambda x: (-x[1], x[0])):
            summary.append(f"- {k}: `{v}`")
    else:
        summary.append("- no flags found")

    summary.append("")
    summary.append("## Outputs")
    summary.append(f"- all_rows: `{ALL_OUT}`")
    summary.append(f"- flagged_rows: `{FLAGGED_OUT}`")
    summary.append("")

    SUMMARY_OUT.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))

    print("\nFIRST 30 FLAGGED ROWS")
    if flagged_rows:
        for r in flagged_rows[:30]:
            print()
            print("ROW:", r["row_num"])
            print("review_id:", r["review_id"])
            print("flags:", r["flags"])
            print("TEXT:", r["sentence_text"])
            print("NORM:", r["normalized_candidate"])
    else:
        print("None")


if __name__ == "__main__":
    main()
