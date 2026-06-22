from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


INPUT_PATH = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1/"
    "text_health_features_v1/"
    "calibration_text_health_features_v1.csv"
)

OUTPUT_DIR = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1/"
    "text_health_routing_policy_v1"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR / "calibration_routing_predictions_v1.csv"
)

CONFUSION_PATH = (
    OUTPUT_DIR / "calibration_routing_confusion_v1.csv"
)

METRICS_PATH = (
    OUTPUT_DIR / "calibration_routing_metrics_v1.csv"
)

ERRORS_PATH = (
    OUTPUT_DIR / "calibration_routing_errors_v1.csv"
)

POLICY_PATH = (
    OUTPUT_DIR / "text_health_routing_policy_v1.json"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "text_health_routing_summary_v1.md"
)


ROUTES = [
    "DIRECT_HEALTHY",
    "REPAIRABLE_USABLE",
    "QUARANTINE",
    "REJECT",
]

GOLD_ROUTE_MAP = {
    "HEALTHY": "DIRECT_HEALTHY",
    "MINOR_DAMAGE_BUT_USABLE": "REPAIRABLE_USABLE",
    "FORMAT_OR_LIST": "QUARANTINE",
    "FRAGMENT": "QUARANTINE",
    "HEAVY_CORRUPTION": "REJECT",
    "MIXED_OR_FOREIGN_TEXT": "REJECT",
}

USABLE_ROUTES = {
    "DIRECT_HEALTHY",
    "REPAIRABLE_USABLE",
}

NON_USABLE_ROUTES = {
    "QUARANTINE",
    "REJECT",
}


POLICY_THRESHOLDS = {
    "extreme_low_alphabetic_ratio": 0.65,
    "direct_min_alphabetic_ratio": 0.78,
    "high_punctuation_ratio": 0.15,
    "repair_punctuation_ratio": 0.10,
    "extreme_short_token_ratio": 0.60,
    "high_short_token_ratio": 0.30,
    "repair_short_token_ratio": 0.15,
    "extreme_number_density": 0.60,
    "high_number_density": 0.35,
    "repair_number_density": 0.20,
    "high_garbage_ratio": 0.15,
    "severe_garbage_ratio": 0.25,
    "long_character_count": 350,
    "long_token_count": 40,
}


TOKEN_PATTERN = re.compile(
    r"[A-Za-zʻʼ’‘'`-]+",
    flags=re.UNICODE,
)


ENGLISH_MARKERS = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "for",
    "with",
    "from",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "between",
    "into",
    "relations",
    "history",
    "society",
    "press",
    "university",
    "journal",
    "volume",
    "edition",
    "chapter",
    "software",
    "management",
    "analysis",
}

RUSSIAN_TRANSLITERATION_MARKERS = {
    "poskolku",
    "poluchayemaya",
    "yavlyayetsya",
    "otritsatelnim",
    "soglasno",
    "prinyatoy",
    "velichina",
    "svidetelstvuyet",
    "nedostatke",
    "normalnogo",
    "razvitiya",
    "rasteniya",
    "usloviyam",
    "otnositelniy",
    "izbitok",
    "testirovaniye",
    "snijayet",
    "trudoyemkost",
    "kontrolya",
    "subektivnost",
    "otsenki",
    "povishayet",
    "rezultatov",
    "obespechivayet",
    "modelirovaniya",
    "soderjaniya",
    "kotorie",
    "takje",
}

OBVIOUS_FOREIGN_SHORT_MARKERS = {
    "apples",
    "software",
    "history",
    "society",
    "journal",
    "university",
    "edition",
    "chapter",
}


def parse_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def parse_float(
    row: dict[str, str],
    column: str,
    default: float = 0.0,
) -> float:
    value = str(row.get(column, "") or "").strip()

    if not value:
        return default

    try:
        return float(value)
    except ValueError:
        return default


def parse_int(
    row: dict[str, str],
    column: str,
    default: int = 0,
) -> int:
    return int(round(parse_float(row, column, default)))


def get_bool(
    row: dict[str, str],
    column: str,
) -> bool:
    return parse_bool(row.get(column, ""))


def normalise_token(token: str) -> str:
    return (
        token.lower()
        .replace("ʻ", "'")
        .replace("ʼ", "'")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .strip("-'")
    )


def analyse_foreign_evidence(
    text: str,
) -> dict[str, object]:
    tokens = [
        normalise_token(token)
        for token in TOKEN_PATTERN.findall(text)
    ]

    tokens = [
        token
        for token in tokens
        if token
    ]

    token_count = len(tokens)

    english_hits = [
        token
        for token in tokens
        if token in ENGLISH_MARKERS
    ]

    russian_hits = [
        token
        for token in tokens
        if token in RUSSIAN_TRANSLITERATION_MARKERS
    ]

    obvious_short_hits = [
        token
        for token in tokens
        if token in OBVIOUS_FOREIGN_SHORT_MARKERS
    ]

    all_hits = english_hits + russian_hits
    hit_count = len(all_hits)

    hit_ratio = (
        hit_count / token_count
        if token_count
        else 0.0
    )

    strong_foreign = (
        (
            token_count >= 5
            and hit_count >= 3
            and hit_ratio >= 0.30
        )
        or (
            token_count >= 8
            and hit_count >= 4
        )
    )

    possible_foreign = (
        (
            token_count >= 4
            and hit_count >= 2
            and hit_ratio >= 0.20
        )
        or (
            token_count <= 4
            and len(obvious_short_hits) >= 1
        )
    )

    return {
        "foreign_token_count": token_count,
        "foreign_marker_count": hit_count,
        "foreign_marker_ratio": hit_ratio,
        "foreign_markers": "|".join(all_hits),
        "strong_foreign_evidence": strong_foreign,
        "possible_foreign_evidence": possible_foreign,
    }


def is_clean_short_utterance(
    *,
    text: str,
    token_count: int,
    character_count: int,
    alphabetic_ratio: float,
    number_density: float,
    garbage_ratio: float,
    merged_count: int,
    list_like: bool,
    formula_like: bool,
    heading_like: bool,
    strong_foreign: bool,
    possible_foreign: bool,
) -> bool:
    stripped = text.strip()

    return (
        1 <= token_count <= 4
        and 4 <= character_count <= 40
        and alphabetic_ratio >= 0.72
        and number_density == 0.0
        and garbage_ratio == 0.0
        and merged_count == 0
        and not list_like
        and not formula_like
        and not heading_like
        and not strong_foreign
        and not possible_foreign
        and stripped.endswith((".", "?", "!"))
    )


def route_row(
    row: dict[str, str],
) -> dict[str, object]:
    text = str(row.get("sentence_text", "") or "").strip()

    character_count = parse_int(
        row,
        "computed_character_count",
    )

    token_count = parse_int(
        row,
        "computed_token_count",
    )

    alphabetic_ratio = parse_float(
        row,
        "alphabetic_character_ratio",
    )

    punctuation_ratio = parse_float(
        row,
        "punctuation_ratio",
    )

    short_token_ratio = parse_float(
        row,
        "very_short_token_ratio",
    )

    number_density = parse_float(
        row,
        "number_expression_density",
    )

    garbage_ratio = parse_float(
        row,
        "garbage_token_ratio",
    )

    merged_count = parse_int(
        row,
        "merged_word_indicator_count",
    )

    segmentation_review = get_bool(
        row,
        "metadata_segmentation_review_required",
    )

    exact_duplicate = get_bool(
        row,
        "metadata_exact_duplicate_text",
    )

    parent_flags = get_bool(
        row,
        "metadata_has_parent_flags",
    )

    boundary_flags = get_bool(
        row,
        "metadata_has_boundary_flags",
    )

    list_like = get_bool(
        row,
        "list_like_structure",
    )

    formula_like = get_bool(
        row,
        "formula_like_structure",
    )

    heading_like = get_bool(
        row,
        "heading_like_structure",
    )

    high_uppercase = get_bool(
        row,
        "signal_high_uppercase_ratio",
    )

    low_alphabetic_signal = get_bool(
        row,
        "signal_low_alphabetic_ratio",
    )

    high_punctuation_signal = get_bool(
        row,
        "signal_high_punctuation_ratio",
    )

    high_short_token_signal = get_bool(
        row,
        "signal_high_short_token_ratio",
    )

    high_garbage_signal = get_bool(
        row,
        "signal_high_garbage_ratio",
    )

    number_heavy_signal = get_bool(
        row,
        "signal_number_heavy",
    )

    merged_damage_signal = get_bool(
        row,
        "signal_merged_word_damage",
    )

    foreign = analyse_foreign_evidence(text)

    strong_foreign = bool(
        foreign["strong_foreign_evidence"]
    )

    possible_foreign = bool(
        foreign["possible_foreign_evidence"]
    )

    clean_short_utterance = is_clean_short_utterance(
        text=text,
        token_count=token_count,
        character_count=character_count,
        alphabetic_ratio=alphabetic_ratio,
        number_density=number_density,
        garbage_ratio=garbage_ratio,
        merged_count=merged_count,
        list_like=list_like,
        formula_like=formula_like,
        heading_like=heading_like,
        strong_foreign=strong_foreign,
        possible_foreign=possible_foreign,
    )

    duplicate_action = (
        "KEEP_ONE_CANONICAL_COPY"
        if exact_duplicate
        else "NOT_DUPLICATE"
    )

    reasons: list[str] = []

    # ------------------------------------------------------------------
    # REJECT:
    # Only use high-confidence evidence. Uncertain severe material should
    # be quarantined rather than permanently rejected.
    # ------------------------------------------------------------------

    if strong_foreign:
        reasons.append("STRONG_FOREIGN_LANGUAGE_EVIDENCE")

        return {
            "predicted_route": "REJECT",
            "route_reasons": "|".join(reasons),
            "duplicate_action": duplicate_action,
            "clean_short_utterance": clean_short_utterance,
            **foreign,
        }

    severe_corruption = (
        (
            token_count >= 5
            and garbage_ratio
            >= POLICY_THRESHOLDS["severe_garbage_ratio"]
            and (
                merged_count >= 1
                or short_token_ratio >= 0.20
            )
        )
        or (
            token_count >= 6
            and merged_count >= 3
            and (
                garbage_ratio >= 0.05
                or short_token_ratio >= 0.20
            )
        )
        or (
            token_count >= 8
            and short_token_ratio >= 0.35
            and garbage_ratio >= 0.08
            and merged_count >= 2
        )
    )

    if severe_corruption:
        reasons.append("MULTI_SIGNAL_SEVERE_CORRUPTION")

        return {
            "predicted_route": "REJECT",
            "route_reasons": "|".join(reasons),
            "duplicate_action": duplicate_action,
            "clean_short_utterance": clean_short_utterance,
            **foreign,
        }

    # ------------------------------------------------------------------
    # QUARANTINE:
    # Structurally unsuitable, uncertain, incomplete or strongly damaged.
    # ------------------------------------------------------------------

    quarantine_reasons: list[str] = []

    if possible_foreign:
        quarantine_reasons.append(
            "POSSIBLE_FOREIGN_LANGUAGE_EVIDENCE"
        )

    if token_count == 0 or character_count == 0:
        quarantine_reasons.append("EMPTY_OR_UNTOKENISABLE")

    if formula_like:
        quarantine_reasons.append("FORMULA_LIKE_STRUCTURE")

    if (
        alphabetic_ratio
        < POLICY_THRESHOLDS[
            "extreme_low_alphabetic_ratio"
        ]
        and (
            token_count <= 4
            or punctuation_ratio >= 0.20
            or short_token_ratio >= 0.50
        )
    ):
        quarantine_reasons.append(
            "EXTREME_LOW_ALPHABETIC_SHORT_SEGMENT"
        )

    if (
        high_punctuation_signal
        and token_count <= 4
        and not clean_short_utterance
    ):
        quarantine_reasons.append(
            "HIGH_PUNCTUATION_SHORT_NON_UTTERANCE"
        )

    if (
        short_token_ratio
        >= POLICY_THRESHOLDS[
            "extreme_short_token_ratio"
        ]
        and not clean_short_utterance
    ):
        quarantine_reasons.append(
            "EXTREME_SHORT_TOKEN_RATIO"
        )

    if (
        number_density
        >= POLICY_THRESHOLDS[
            "extreme_number_density"
        ]
    ):
        quarantine_reasons.append(
            "EXTREME_NUMBER_EXPRESSION_DENSITY"
        )

    if (
        list_like
        and (
            number_density >= 0.25
            or token_count <= 8
            or segmentation_review
            or punctuation_ratio >= 0.10
        )
    ):
        quarantine_reasons.append(
            "STRUCTURED_LIST_LIKE_CONTENT"
        )

    if heading_like and token_count <= 10:
        quarantine_reasons.append(
            "SHORT_HEADING_LIKE_CONTENT"
        )

    if (
        high_uppercase
        and token_count <= 10
        and not clean_short_utterance
    ):
        quarantine_reasons.append(
            "SHORT_HIGH_UPPERCASE_CONTENT"
        )

    if (
        segmentation_review
        and (
            token_count <= 3
            or punctuation_ratio >= 0.15
            or short_token_ratio >= 0.30
            or merged_count >= 2
        )
    ):
        quarantine_reasons.append(
            "SEGMENTATION_WARNING_WITH_DAMAGE"
        )

    if (
        number_density
        >= POLICY_THRESHOLDS[
            "high_number_density"
        ]
        and (
            short_token_ratio >= 0.20
            or list_like
            or segmentation_review
        )
    ):
        quarantine_reasons.append(
            "NUMBER_HEAVY_WITH_STRUCTURE_WARNING"
        )

    if (
        garbage_ratio
        >= POLICY_THRESHOLDS[
            "high_garbage_ratio"
        ]
    ):
        quarantine_reasons.append(
            "HIGH_GARBAGE_TOKEN_RATIO"
        )

    if (
        merged_count >= 2
        and (
            garbage_ratio >= 0.05
            or short_token_ratio >= 0.20
            or segmentation_review
        )
    ):
        quarantine_reasons.append(
            "MULTIPLE_MERGED_WORD_DAMAGE"
        )

    if quarantine_reasons:
        return {
            "predicted_route": "QUARANTINE",
            "route_reasons": "|".join(
                sorted(set(quarantine_reasons))
            ),
            "duplicate_action": duplicate_action,
            "clean_short_utterance": clean_short_utterance,
            **foreign,
        }

    # ------------------------------------------------------------------
    # REPAIRABLE_USABLE:
    # The meaning is intact, but at least one warning or repair signal
    # exists. These sentences must not directly become clean targets.
    # ------------------------------------------------------------------

    repair_reasons: list[str] = []

    if merged_damage_signal or merged_count >= 1:
        repair_reasons.append("MERGED_WORD_REPAIR_NEEDED")

    if garbage_ratio > 0.0 or high_garbage_signal:
        repair_reasons.append("LOCAL_GARBAGE_REPAIR_NEEDED")

    if segmentation_review:
        repair_reasons.append("SEGMENTATION_REVIEW_WARNING")

    if parent_flags:
        repair_reasons.append("PARENT_RECORD_WARNING")

    if boundary_flags:
        repair_reasons.append("BOUNDARY_WARNING")

    if (
        number_density
        >= POLICY_THRESHOLDS[
            "repair_number_density"
        ]
        or number_heavy_signal
    ):
        repair_reasons.append("NUMBER_HEAVY_SENTENCE")

    if (
        short_token_ratio
        >= POLICY_THRESHOLDS[
            "repair_short_token_ratio"
        ]
        or high_short_token_signal
    ):
        repair_reasons.append("ELEVATED_SHORT_TOKEN_RATIO")

    if (
        alphabetic_ratio
        < POLICY_THRESHOLDS[
            "direct_min_alphabetic_ratio"
        ]
        or low_alphabetic_signal
    ):
        repair_reasons.append("LOWER_ALPHABETIC_RATIO")

    if (
        punctuation_ratio
        > POLICY_THRESHOLDS[
            "repair_punctuation_ratio"
        ]
        and not clean_short_utterance
    ):
        repair_reasons.append("ELEVATED_PUNCTUATION_RATIO")

    if list_like:
        repair_reasons.append("LIST_LIKE_BUT_SENTENTIAL")

    if heading_like:
        repair_reasons.append("HEADING_LIKE_BUT_SENTENTIAL")

    if high_uppercase:
        repair_reasons.append("HIGH_UPPERCASE_RATIO")

    if (
        character_count
        > POLICY_THRESHOLDS[
            "long_character_count"
        ]
        or token_count
        > POLICY_THRESHOLDS[
            "long_token_count"
        ]
    ):
        repair_reasons.append("LONG_SEGMENT_REVIEW")

    if repair_reasons:
        return {
            "predicted_route": "REPAIRABLE_USABLE",
            "route_reasons": "|".join(
                sorted(set(repair_reasons))
            ),
            "duplicate_action": duplicate_action,
            "clean_short_utterance": clean_short_utterance,
            **foreign,
        }

    reasons.append("NO_MATERIAL_WARNING_SIGNAL")

    return {
        "predicted_route": "DIRECT_HEALTHY",
        "route_reasons": "|".join(reasons),
        "duplicate_action": duplicate_action,
        "clean_short_utterance": clean_short_utterance,
        **foreign,
    }


def safe_divide(
    numerator: int,
    denominator: int,
) -> float:
    return (
        numerator / denominator
        if denominator
        else 0.0
    )


def precision_recall_f1(
    *,
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> tuple[float, float, float]:
    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1


if not INPUT_PATH.exists():
    raise SystemExit(
        f"Missing calibration feature file: {INPUT_PATH}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

with INPUT_PATH.open(
    "r",
    encoding="utf-8",
    newline="",
) as file:
    rows = list(csv.DictReader(file))

if len(rows) != 699:
    raise SystemExit(
        f"Expected 699 calibration rows, found {len(rows)}"
    )

required_columns = {
    "review_sample_id",
    "sentence_text",
    "gold_health_label",
    "computed_character_count",
    "computed_token_count",
    "alphabetic_character_ratio",
    "punctuation_ratio",
    "very_short_token_ratio",
    "number_expression_density",
    "garbage_token_ratio",
    "merged_word_indicator_count",
}

missing_columns = sorted(
    required_columns - set(rows[0])
)

if missing_columns:
    raise SystemExit(
        f"Missing required columns: {missing_columns}"
    )


prediction_rows: list[dict[str, object]] = []
confusion: dict[str, Counter[str]] = defaultdict(Counter)

for row in rows:
    gold_label = str(
        row.get("gold_health_label", "")
    ).strip()

    if gold_label not in GOLD_ROUTE_MAP:
        raise SystemExit(
            f"Unexpected gold label: {gold_label!r}"
        )

    gold_route = GOLD_ROUTE_MAP[gold_label]
    decision = route_row(row)
    predicted_route = str(
        decision["predicted_route"]
    )

    confusion[gold_route][predicted_route] += 1

    prediction_rows.append(
        {
            "review_sample_id": row["review_sample_id"],
            "sentence_text": row["sentence_text"],
            "gold_health_label": gold_label,
            "gold_route": gold_route,
            "predicted_route": predicted_route,
            "route_correct": (
                predicted_route == gold_route
            ),
            "route_reasons": decision["route_reasons"],
            "duplicate_action": decision[
                "duplicate_action"
            ],
            "clean_short_utterance": decision[
                "clean_short_utterance"
            ],
            "foreign_token_count": decision[
                "foreign_token_count"
            ],
            "foreign_marker_count": decision[
                "foreign_marker_count"
            ],
            "foreign_marker_ratio": round(
                float(
                    decision["foreign_marker_ratio"]
                ),
                6,
            ),
            "foreign_markers": decision[
                "foreign_markers"
            ],
            "strong_foreign_evidence": decision[
                "strong_foreign_evidence"
            ],
            "possible_foreign_evidence": decision[
                "possible_foreign_evidence"
            ],
            "character_count": row[
                "computed_character_count"
            ],
            "token_count": row[
                "computed_token_count"
            ],
            "alphabetic_character_ratio": row[
                "alphabetic_character_ratio"
            ],
            "punctuation_ratio": row[
                "punctuation_ratio"
            ],
            "very_short_token_ratio": row[
                "very_short_token_ratio"
            ],
            "number_expression_density": row[
                "number_expression_density"
            ],
            "garbage_token_ratio": row[
                "garbage_token_ratio"
            ],
            "merged_word_indicator_count": row[
                "merged_word_indicator_count"
            ],
        }
    )


with PREDICTIONS_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    fieldnames = list(prediction_rows[0])

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(prediction_rows)


with CONFUSION_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    writer = csv.writer(file)

    writer.writerow(
        ["gold_route", *ROUTES, "total"]
    )

    for gold_route in ROUTES:
        values = [
            confusion[gold_route][predicted]
            for predicted in ROUTES
        ]

        writer.writerow(
            [
                gold_route,
                *values,
                sum(values),
            ]
        )


metric_rows: list[dict[str, object]] = []

for route in ROUTES:
    true_positive = confusion[route][route]

    false_positive = sum(
        confusion[other_route][route]
        for other_route in ROUTES
        if other_route != route
    )

    false_negative = sum(
        confusion[route][other_route]
        for other_route in ROUTES
        if other_route != route
    )

    precision, recall, f1 = precision_recall_f1(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )

    metric_rows.append(
        {
            "metric_scope": route,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    )


gold_usable_count = sum(
    1
    for row in prediction_rows
    if row["gold_route"] in USABLE_ROUTES
)

predicted_usable_count = sum(
    1
    for row in prediction_rows
    if row["predicted_route"] in USABLE_ROUTES
)

correct_usable_count = sum(
    1
    for row in prediction_rows
    if (
        row["gold_route"] in USABLE_ROUTES
        and row["predicted_route"] in USABLE_ROUTES
    )
)

gold_nonusable_count = sum(
    1
    for row in prediction_rows
    if row["gold_route"] in NON_USABLE_ROUTES
)

nonusable_leaked_to_usable = sum(
    1
    for row in prediction_rows
    if (
        row["gold_route"] in NON_USABLE_ROUTES
        and row["predicted_route"] in USABLE_ROUTES
    )
)

direct_predictions = [
    row
    for row in prediction_rows
    if row["predicted_route"] == "DIRECT_HEALTHY"
]

direct_correct = sum(
    1
    for row in direct_predictions
    if row["gold_route"] == "DIRECT_HEALTHY"
)

exact_correct = sum(
    1
    for row in prediction_rows
    if row["route_correct"]
)


summary_metrics = {
    "exact_route_accuracy": safe_divide(
        exact_correct,
        len(prediction_rows),
    ),
    "direct_healthy_precision": safe_divide(
        direct_correct,
        len(direct_predictions),
    ),
    "direct_healthy_coverage": safe_divide(
        len(direct_predictions),
        len(prediction_rows),
    ),
    "combined_usable_precision": safe_divide(
        correct_usable_count,
        predicted_usable_count,
    ),
    "combined_usable_recall": safe_divide(
        correct_usable_count,
        gold_usable_count,
    ),
    "nonusable_leakage_to_usable": safe_divide(
        nonusable_leaked_to_usable,
        gold_nonusable_count,
    ),
}


with METRICS_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(metric_rows[0]),
    )

    writer.writeheader()
    writer.writerows(metric_rows)


error_rows = [
    row
    for row in prediction_rows
    if not row["route_correct"]
]

error_priority = {
    ("REJECT", "DIRECT_HEALTHY"): 1,
    ("REJECT", "REPAIRABLE_USABLE"): 2,
    ("QUARANTINE", "DIRECT_HEALTHY"): 3,
    ("QUARANTINE", "REPAIRABLE_USABLE"): 4,
    ("DIRECT_HEALTHY", "REJECT"): 5,
    ("DIRECT_HEALTHY", "QUARANTINE"): 6,
    ("REPAIRABLE_USABLE", "REJECT"): 7,
    ("REPAIRABLE_USABLE", "QUARANTINE"): 8,
}

error_rows.sort(
    key=lambda row: (
        error_priority.get(
            (
                str(row["gold_route"]),
                str(row["predicted_route"]),
            ),
            99,
        ),
        str(row["review_sample_id"]),
    )
)


with ERRORS_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(error_rows[0])
        if error_rows
        else list(prediction_rows[0]),
    )

    writer.writeheader()

    if error_rows:
        writer.writerows(error_rows)


policy_specification = {
    "policy_name": "TEXT_HEALTH_ROUTING_POLICY_V1",
    "status": "CALIBRATION_CANDIDATE_NOT_FROZEN",
    "routes": ROUTES,
    "thresholds": POLICY_THRESHOLDS,
    "duplicate_policy": {
        "exact_duplicate": "KEEP_ONE_CANONICAL_COPY",
        "duplicate_status_is_not_a_language_health_label": True,
    },
    "priority_order": [
        "REJECT",
        "QUARANTINE",
        "REPAIRABLE_USABLE",
        "DIRECT_HEALTHY",
    ],
    "validation_labels_used": False,
}

with POLICY_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        policy_specification,
        file,
        ensure_ascii=False,
        indent=2,
    )


predicted_counts = Counter(
    str(row["predicted_route"])
    for row in prediction_rows
)

reason_counts = Counter()

for row in prediction_rows:
    for reason in str(row["route_reasons"]).split("|"):
        if reason:
            reason_counts[reason] += 1


summary_lines = [
    "# Text Health Routing Policy v1 — Calibration Summary",
    "",
    "## Status",
    "",
    "CALIBRATION_CANDIDATE_NOT_FROZEN",
    "",
    "## Predicted route distribution",
    "",
]

for route in ROUTES:
    count = predicted_counts[route]
    percentage = safe_divide(
        count,
        len(prediction_rows),
    ) * 100

    summary_lines.append(
        f"- {route}: {count} ({percentage:.2f}%)"
    )

summary_lines.extend(
    [
        "",
        "## Safety metrics",
        "",
        (
            "- Exact four-route accuracy: "
            f"{summary_metrics['exact_route_accuracy']:.2%}"
        ),
        (
            "- Direct healthy precision: "
            f"{summary_metrics['direct_healthy_precision']:.2%}"
        ),
        (
            "- Direct healthy coverage: "
            f"{summary_metrics['direct_healthy_coverage']:.2%}"
        ),
        (
            "- Combined usable precision: "
            f"{summary_metrics['combined_usable_precision']:.2%}"
        ),
        (
            "- Combined usable recall: "
            f"{summary_metrics['combined_usable_recall']:.2%}"
        ),
        (
            "- Non-usable leakage into usable lanes: "
            f"{summary_metrics['nonusable_leakage_to_usable']:.2%}"
        ),
        "",
        "## Most frequent routing reasons",
        "",
    ]
)

for reason, count in reason_counts.most_common(20):
    summary_lines.append(
        f"- {reason}: {count}"
    )

summary_lines.extend(
    [
        "",
        "## Data protection",
        "",
        "- Master CSV modified: NO",
        "- Validation rows used: NO",
        "- Validation labels exposed: NO",
        "- Full-corpus filtering applied: NO",
        "",
    ]
)

SUMMARY_PATH.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)


print("=" * 112)
print("TEXT HEALTH ROUTING POLICY V1 — CALIBRATION BASELINE")
print("=" * 112)

print("\nPREDICTED ROUTE DISTRIBUTION")
print("-" * 112)

for route in ROUTES:
    count = predicted_counts[route]
    percentage = safe_divide(
        count,
        len(prediction_rows),
    ) * 100

    print(
        f"{route:<26}: "
        f"{count:>3}/{len(prediction_rows)} "
        f"({percentage:>6.2f}%)"
    )


print("\nFOUR-ROUTE CONFUSION MATRIX")
print("-" * 112)

print(
    "{:<28}".format("GOLD / PREDICTED")
    + "".join(
        f"{route[:12]:>15}"
        for route in ROUTES
    )
)

for gold_route in ROUTES:
    print(
        f"{gold_route:<28}"
        + "".join(
            f"{confusion[gold_route][predicted]:>15}"
            for predicted in ROUTES
        )
    )


print("\nPER-ROUTE METRICS")
print("-" * 112)

for metric in metric_rows:
    print(
        f"{metric['metric_scope']:<26} | "
        f"precision={float(metric['precision']):>7.2%} | "
        f"recall={float(metric['recall']):>7.2%} | "
        f"f1={float(metric['f1']):>7.2%}"
    )


print("\nSAFETY-FOCUSED METRICS")
print("-" * 112)

print(
    "exact_route_accuracy:"
    f" {summary_metrics['exact_route_accuracy']:.2%}"
)

print(
    "direct_healthy_precision:"
    f" {summary_metrics['direct_healthy_precision']:.2%}"
)

print(
    "direct_healthy_coverage:"
    f" {summary_metrics['direct_healthy_coverage']:.2%}"
)

print(
    "combined_usable_precision:"
    f" {summary_metrics['combined_usable_precision']:.2%}"
)

print(
    "combined_usable_recall:"
    f" {summary_metrics['combined_usable_recall']:.2%}"
)

print(
    "nonusable_leakage_to_usable:"
    f" {summary_metrics['nonusable_leakage_to_usable']:.2%}"
)


print("\nTOP ROUTING REASONS")
print("-" * 112)

for reason, count in reason_counts.most_common(20):
    print(f"{reason:<55}: {count}")


print("\nOUTPUT FILES")
print("-" * 112)

for path in [
    PREDICTIONS_PATH,
    CONFUSION_PATH,
    METRICS_PATH,
    ERRORS_PATH,
    POLICY_PATH,
    SUMMARY_PATH,
]:
    print(path)


print("\nmaster_csv_modified: NO")
print("validation_rows_used: NO")
print("validation_labels_exposed: NO")
print("full_corpus_filtering_applied: NO")
print("TEXT HEALTH ROUTING POLICY V1 BASELINE: PASS")
print("STATUS: READY_TO_REVIEW_ROUTING_METRICS_AND_ERRORS")
