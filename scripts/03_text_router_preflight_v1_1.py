from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1"
)

# Corrected input: features joined with human labels.
CALIBRATION_PATH = (
    ROOT
    / "text_health_features_v1"
    / "calibration_text_health_features_labeled_v1.csv"
)

# Must remain blind and unchanged.
VALIDATION_PATH = (
    ROOT
    / "text_health_features_v1"
    / "validation_text_health_features_blind_v1.csv"
)

EXPECTED_CALIBRATION_ROWS = 699
EXPECTED_VALIDATION_ROWS = 301

EXPECTED_LABEL_COUNTS = {
    "HEALTHY": 239,
    "MINOR_DAMAGE_BUT_USABLE": 251,
    "FORMAT_OR_LIST": 116,
    "FRAGMENT": 31,
    "HEAVY_CORRUPTION": 39,
    "MIXED_OR_FOREIGN_TEXT": 23,
    "UNCERTAIN": 0,
}

ROUTE_MAP = {
    "HEALTHY": "DIRECT_HEALTHY",
    "MINOR_DAMAGE_BUT_USABLE": "REPAIRABLE_USABLE",
    "FORMAT_OR_LIST": "QUARANTINE",
    "FRAGMENT": "QUARANTINE",
    "HEAVY_CORRUPTION": "REJECT",
    "MIXED_OR_FOREIGN_TEXT": "REJECT",
    "UNCERTAIN": "QUARANTINE",
}

ID_CANDIDATES = [
    "review_sample_id",
    "sentence_id",
    "sample_id",
]

LABEL_CANDIDATES = [
    "human_health_label",
    "reference_label",
    "gold_label",
]

REQUIRED_FEATURES = [
    "computed_character_count",
    "computed_token_count",
    "alphabetic_character_ratio",
    "punctuation_ratio",
    "very_short_token_ratio",
    "number_expression_density",
    "garbage_token_ratio",
    "merged_word_indicator_count",
]


def fail(reason: str) -> None:
    print("\nPREFLIGHT: FAIL")
    print("REASON:", reason)
    sys.exit(1)


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        fail(f"Input file does not exist: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            fail(f"CSV has no header: {path}")

        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]

    return fieldnames, rows


def resolve_column(
    fieldnames: list[str],
    candidates: list[str],
    *,
    description: str,
    required: bool = True,
) -> str | None:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate

    if required:
        fail(
            f"Could not resolve {description}. "
            f"Tried columns: {candidates}"
        )

    return None


def clean(value: object) -> str:
    return str(value or "").strip()


def duplicate_values(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(
        value
        for value, count in counts.items()
        if value and count > 1
    )


calibration_fields, calibration_rows = load_csv(CALIBRATION_PATH)
validation_fields, validation_rows = load_csv(VALIDATION_PATH)

calibration_id_column = resolve_column(
    calibration_fields,
    ID_CANDIDATES,
    description="calibration ID column",
)

validation_id_column = resolve_column(
    validation_fields,
    ID_CANDIDATES,
    description="validation ID column",
)

calibration_label_column = resolve_column(
    calibration_fields,
    LABEL_CANDIDATES,
    description="calibration label column",
)

validation_label_columns = [
    column
    for column in LABEL_CANDIDATES
    if column in validation_fields
]

print("=" * 112)
print("HIERARCHICAL TEXT ROUTER V1.1 — PREFLIGHT")
print("=" * 112)

print("\nINPUT FILES")
print("-" * 112)
print("calibration:", CALIBRATION_PATH)
print("validation :", VALIDATION_PATH)

print("\nRESOLVED SCHEMA")
print("-" * 112)
print("calibration_id_column   :", calibration_id_column)
print("calibration_label_column:", calibration_label_column)
print("validation_id_column    :", validation_id_column)
print(
    "validation_label_columns:",
    validation_label_columns or "NONE",
)

# ---------------------------------------------------------------------
# Row-count validation
# ---------------------------------------------------------------------

if len(calibration_rows) != EXPECTED_CALIBRATION_ROWS:
    fail(
        "Unexpected calibration row count. "
        f"Expected {EXPECTED_CALIBRATION_ROWS}, "
        f"found {len(calibration_rows)}."
    )

if len(validation_rows) != EXPECTED_VALIDATION_ROWS:
    fail(
        "Unexpected validation row count. "
        f"Expected {EXPECTED_VALIDATION_ROWS}, "
        f"found {len(validation_rows)}."
    )

# ---------------------------------------------------------------------
# ID validation and split leakage checks
# ---------------------------------------------------------------------

calibration_ids = [
    clean(row.get(calibration_id_column))
    for row in calibration_rows
]

validation_ids = [
    clean(row.get(validation_id_column))
    for row in validation_rows
]

blank_calibration_ids = [
    index
    for index, value in enumerate(calibration_ids, start=1)
    if not value
]

blank_validation_ids = [
    index
    for index, value in enumerate(validation_ids, start=1)
    if not value
]

if blank_calibration_ids:
    fail(
        "Blank calibration IDs found at row positions: "
        f"{blank_calibration_ids[:20]}"
    )

if blank_validation_ids:
    fail(
        "Blank validation IDs found at row positions: "
        f"{blank_validation_ids[:20]}"
    )

duplicate_calibration_ids = duplicate_values(calibration_ids)
duplicate_validation_ids = duplicate_values(validation_ids)

if duplicate_calibration_ids:
    fail(
        "Duplicate calibration IDs found. Examples: "
        f"{duplicate_calibration_ids[:10]}"
    )

if duplicate_validation_ids:
    fail(
        "Duplicate validation IDs found. Examples: "
        f"{duplicate_validation_ids[:10]}"
    )

split_overlap = sorted(
    set(calibration_ids) & set(validation_ids)
)

if split_overlap:
    fail(
        "Calibration-validation ID leakage detected. Examples: "
        f"{split_overlap[:10]}"
    )

# ---------------------------------------------------------------------
# Calibration-label validation
# ---------------------------------------------------------------------

calibration_labels = [
    clean(row.get(calibration_label_column))
    for row in calibration_rows
]

blank_label_rows = [
    index
    for index, label in enumerate(calibration_labels, start=1)
    if not label
]

if blank_label_rows:
    fail(
        "Blank calibration labels found at row positions: "
        f"{blank_label_rows[:20]}"
    )

unexpected_labels = sorted(
    set(calibration_labels) - set(EXPECTED_LABEL_COUNTS)
)

if unexpected_labels:
    fail(
        f"Unexpected calibration labels found: {unexpected_labels}"
    )

label_counts = Counter(calibration_labels)

for label, expected_count in EXPECTED_LABEL_COUNTS.items():
    actual_count = label_counts.get(label, 0)

    if actual_count != expected_count:
        fail(
            f"Calibration count mismatch for {label}: "
            f"expected {expected_count}, found {actual_count}."
        )

# ---------------------------------------------------------------------
# Validation must remain blind
# ---------------------------------------------------------------------

exposed_validation_labels: list[tuple[int, str, str]] = []

for column in validation_label_columns:
    for row_number, row in enumerate(validation_rows, start=1):
        value = clean(row.get(column))

        if value:
            exposed_validation_labels.append(
                (row_number, column, value)
            )

if exposed_validation_labels:
    fail(
        "Validation labels are exposed. First examples: "
        f"{exposed_validation_labels[:10]}"
    )

# ---------------------------------------------------------------------
# Feature-schema checks
# ---------------------------------------------------------------------

missing_calibration_features = [
    feature
    for feature in REQUIRED_FEATURES
    if feature not in calibration_fields
]

missing_validation_features = [
    feature
    for feature in REQUIRED_FEATURES
    if feature not in validation_fields
]

if missing_calibration_features:
    fail(
        "Calibration file is missing required features: "
        f"{missing_calibration_features}"
    )

if missing_validation_features:
    fail(
        "Validation file is missing required features: "
        f"{missing_validation_features}"
    )

invalid_numeric_values: list[tuple[str, int, str, str]] = []

for split_name, rows in [
    ("CALIBRATION", calibration_rows),
    ("VALIDATION", validation_rows),
]:
    for row_number, row in enumerate(rows, start=1):
        for feature in REQUIRED_FEATURES:
            value = clean(row.get(feature))

            if not value:
                invalid_numeric_values.append(
                    (split_name, row_number, feature, "BLANK")
                )
                continue

            try:
                float(value)
            except ValueError:
                invalid_numeric_values.append(
                    (split_name, row_number, feature, value)
                )

if invalid_numeric_values:
    fail(
        "Invalid required feature values found. First examples: "
        f"{invalid_numeric_values[:10]}"
    )

shared_columns = sorted(
    set(calibration_fields) & set(validation_fields)
)

calibration_only_columns = sorted(
    set(calibration_fields) - set(validation_fields)
)

validation_only_columns = sorted(
    set(validation_fields) - set(calibration_fields)
)

# ---------------------------------------------------------------------
# Derived route distribution
# ---------------------------------------------------------------------

route_counts = Counter(
    ROUTE_MAP[label]
    for label in calibration_labels
)

print("\nROW AND SPLIT CHECKS")
print("-" * 112)
print(
    f"calibration_rows              : "
    f"{len(calibration_rows)} — PASS"
)
print(
    f"validation_rows               : "
    f"{len(validation_rows)} — PASS"
)
print("calibration_ids_unique        : PASS")
print("validation_ids_unique         : PASS")
print("calibration_validation_overlap: 0 — PASS")
print("validation_labels_exposed     : NO — PASS")

print("\nCALIBRATION LABEL DISTRIBUTION")
print("-" * 112)

for label in EXPECTED_LABEL_COUNTS:
    print(
        f"{label:<30}: "
        f"{label_counts.get(label, 0):>3}"
    )

print("\nDERIVED ROUTE DISTRIBUTION")
print("-" * 112)

for route in [
    "DIRECT_HEALTHY",
    "REPAIRABLE_USABLE",
    "QUARANTINE",
    "REJECT",
]:
    count = route_counts.get(route, 0)
    percentage = count / len(calibration_rows)

    print(
        f"{route:<30}: "
        f"{count:>3}/{len(calibration_rows)} "
        f"({percentage:>6.2%})"
    )

print("\nFEATURE SCHEMA")
print("-" * 112)
print("required_features_present     : PASS")
print("required_feature_values_valid : PASS")
print("shared_columns                :", len(shared_columns))
print(
    "calibration_only_columns      :",
    calibration_only_columns,
)
print(
    "validation_only_columns       :",
    validation_only_columns,
)

print("\nSAFETY STATUS")
print("-" * 112)
print("calibration_labels_available  : YES")
print("validation_labels_available   : NO")
print("validation_split_reserved     : YES")
print("input_files_modified          : NO")
print("master_csv_modified           : NO")

print("\nPREFLIGHT: PASS")
print(
    "STATUS: READY_TO_TRAIN_"
    "HIERARCHICAL_TEXT_ROUTER_V1_1"
)
