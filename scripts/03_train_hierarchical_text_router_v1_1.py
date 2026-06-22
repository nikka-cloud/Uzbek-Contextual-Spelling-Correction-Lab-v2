from __future__ import annotations

import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing Python package for router training: " + str(exc) + "\n"
        "Install scikit-learn/numpy first, then rerun this script."
    )


ROOT = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1"
)

INPUT_PATH = (
    ROOT
    / "text_health_features_v1"
    / "calibration_text_health_features_labeled_v1.csv"
)

VALIDATION_BLIND_PATH = (
    ROOT
    / "text_health_features_v1"
    / "validation_text_health_features_blind_v1.csv"
)

OUTPUT_DIR = ROOT / "hierarchical_text_router_v1_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OOF_PATH = OUTPUT_DIR / "calibration_hierarchical_oof_predictions_v1_1.csv"
FEATURES_PATH = OUTPUT_DIR / "selected_router_features_v1_1.csv"
STAGE1_SEARCH_PATH = OUTPUT_DIR / "stage1_usable_gate_threshold_search_v1_1.csv"
STAGE2A_SEARCH_PATH = OUTPUT_DIR / "stage2a_direct_healthy_threshold_search_v1_1.csv"
STAGE2B_SEARCH_PATH = OUTPUT_DIR / "stage2b_reject_threshold_search_v1_1.csv"
MODEL_COMPARISON_PATH = OUTPUT_DIR / "model_comparison_v1_1.csv"
CONFUSION_PATH = OUTPUT_DIR / "hierarchical_route_confusion_v1_1.csv"
SUMMARY_PATH = OUTPUT_DIR / "hierarchical_router_summary_v1_1.md"
POLICY_PATH = OUTPUT_DIR / "hierarchical_text_router_policy_v1_1.json"
MODELS_PATH = OUTPUT_DIR / "hierarchical_text_router_models_v1_1.pkl"

RANDOM_SEED = 42
N_SPLITS = 5

EXPECTED_ROWS = 699
EXPECTED_LABEL_COUNTS = {
    "HEALTHY": 239,
    "MINOR_DAMAGE_BUT_USABLE": 251,
    "FORMAT_OR_LIST": 116,
    "FRAGMENT": 31,
    "HEAVY_CORRUPTION": 39,
    "MIXED_OR_FOREIGN_TEXT": 23,
    "UNCERTAIN": 0,
}

LABEL_TO_ROUTE = {
    "HEALTHY": "DIRECT_HEALTHY",
    "MINOR_DAMAGE_BUT_USABLE": "REPAIRABLE_USABLE",
    "FORMAT_OR_LIST": "QUARANTINE",
    "FRAGMENT": "QUARANTINE",
    "HEAVY_CORRUPTION": "REJECT",
    "MIXED_OR_FOREIGN_TEXT": "REJECT",
    "UNCERTAIN": "QUARANTINE",
}

USABLE_ROUTES = {"DIRECT_HEALTHY", "REPAIRABLE_USABLE"}
NONUSABLE_ROUTES = {"QUARANTINE", "REJECT"}

EXCLUDE_EXACT = {
    "human_health_label",
    "human_health_notes",
    "gold_health_label",
    "gold_health_notes",
    "gold_label_hidden",
    "gold_reference_lane",
    "gold_route",
    "reference_label",
    "label",
    "notes",
    "sentence_text",
    "previous_sentence",
    "next_sentence",
    "previous_sentence_text",
    "next_sentence_text",
    "review_sample_id",
    "sample_role",
    "sample_group",
    "source_record_id",
    "sentence_id",
    "review_batch",
    "batch_number",
    "part",
    "part_number",
    "sentence_index_in_record",
    "sentence_start_char",
    "sentence_end_char",
    "sentence_start",
    "sentence_end",
    "sentence_offsets",
}

EXCLUDE_SUBSTRINGS = [
    "label",
    "note",
    "text",
    "sample_id",
    "sentence_id",
    "record_id",
    "route",
    "prediction",
    "probability",
    "gold",
]

THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)] + [0.97, 0.99]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise SystemExit(f"Input file has no rows: {path}")
    return rows, fieldnames


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Some rows may gain extra diagnostic keys later, for example selection_status.
    # Keep CSV writing robust by expanding the header instead of crashing.
    full_fieldnames = list(fieldnames)
    seen = set(full_fieldnames)

    for row in rows:
        for key in row.keys():
            if key not in seen:
                full_fieldnames.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=full_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_column(fieldnames: list[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    raise SystemExit(f"Could not resolve required column from candidates: {candidates}")


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None

    low = text.lower()

    if low in {"true", "yes", "y", "1", "pass"}:
        return 1.0
    if low in {"false", "no", "n", "0", "none", "blank", "n/a", "na"}:
        return 0.0

    if low.endswith("%"):
        try:
            return float(low[:-1]) / 100.0
        except ValueError:
            return None

    try:
        value_float = float(text)
    except ValueError:
        return None

    if math.isfinite(value_float):
        return value_float

    return None


def is_excluded_feature(column: str) -> bool:
    if column in EXCLUDE_EXACT:
        return True
    lowered = column.lower()
    return any(part in lowered for part in EXCLUDE_SUBSTRINGS)


def select_numeric_features(rows: list[dict[str, str]], fieldnames: list[str]) -> list[str]:
    selected = []

    for column in fieldnames:
        if is_excluded_feature(column):
            continue

        parsed_values = [parse_float(row.get(column)) for row in rows]
        non_missing = [value for value in parsed_values if value is not None]

        if not non_missing:
            continue

        valid_ratio = len(non_missing) / len(rows)
        unique_values = set(non_missing)

        if valid_ratio >= 0.98 and len(unique_values) > 1:
            selected.append(column)

    if not selected:
        raise SystemExit("No numeric/router feature columns were selected.")

    return selected


def build_matrix(rows: list[dict[str, str]], feature_columns: list[str]) -> np.ndarray:
    matrix = []

    for row in rows:
        values = []
        for column in feature_columns:
            parsed = parse_float(row.get(column))
            values.append(np.nan if parsed is None else float(parsed))
        matrix.append(values)

    return np.asarray(matrix, dtype=float)


def make_models() -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=10,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=0.05,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }


def positive_probability_from_oof(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    positive_label: str,
) -> np.ndarray:
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    probabilities = cross_val_predict(
        model,
        x,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=None,
    )

    fitted_once = clone(model).fit(x, y)
    classes = list(fitted_once.classes_)

    if positive_label not in classes:
        raise SystemExit(f"Positive label {positive_label!r} not found in {classes}")

    positive_index = classes.index(positive_label)
    return probabilities[:, positive_index]


def safe_div(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def binary_threshold_rows(
    *,
    model_name: str,
    stage_name: str,
    probabilities: np.ndarray,
    y_true: np.ndarray,
    positive_label: str,
    negative_label: str,
) -> list[dict[str, Any]]:
    rows = []

    total_positive = int(np.sum(y_true == positive_label))
    total_negative = int(np.sum(y_true == negative_label))

    for threshold in THRESHOLDS:
        pred_positive = probabilities >= threshold
        true_positive_mask = y_true == positive_label
        true_negative_mask = y_true == negative_label

        tp = int(np.sum(pred_positive & true_positive_mask))
        fp = int(np.sum(pred_positive & true_negative_mask))
        fn = int(np.sum((~pred_positive) & true_positive_mask))
        tn = int(np.sum((~pred_positive) & true_negative_mask))

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        negative_leakage_to_positive = safe_div(fp, total_negative)
        contamination_rate = safe_div(fp, tp + fp)
        positive_coverage = safe_div(tp + fp, len(y_true))

        rows.append(
            {
                "stage": stage_name,
                "model_name": model_name,
                "threshold": f"{threshold:.2f}",
                "positive_label": positive_label,
                "negative_label": negative_label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "positive_precision": f"{precision:.6f}",
                "positive_recall": f"{recall:.6f}",
                "positive_f1": f"{f1:.6f}",
                "negative_leakage_to_positive": f"{negative_leakage_to_positive:.6f}",
                "contamination_rate": f"{contamination_rate:.6f}",
                "positive_coverage": f"{positive_coverage:.6f}",
                "total_positive": total_positive,
                "total_negative": total_negative,
            }
        )

    return rows


def f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def choose_stage1(search_rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in search_rows
        if f(row, "positive_precision") >= 0.90
        and f(row, "negative_leakage_to_positive") <= 0.10
    ]

    if eligible:
        chosen = max(
            eligible,
            key=lambda row: (
                f(row, "positive_coverage"),
                f(row, "positive_recall"),
                f(row, "positive_precision"),
            ),
        )
        chosen["selection_status"] = "MEETS_INITIAL_SAFETY_TARGET"
        return chosen

    fallback = max(
        search_rows,
        key=lambda row: (
            f(row, "positive_precision"),
            -f(row, "negative_leakage_to_positive"),
            f(row, "positive_coverage"),
        ),
    )
    fallback["selection_status"] = "FALLBACK_BEST_AVAILABLE_DO_NOT_APPLY_FULL_CORPUS_YET"
    return fallback


def choose_precision_stage(
    search_rows: list[dict[str, Any]],
    *,
    min_precision: float,
    max_negative_leakage: float | None,
) -> dict[str, Any]:
    eligible = [
        row
        for row in search_rows
        if f(row, "positive_precision") >= min_precision
        and (
            max_negative_leakage is None
            or f(row, "negative_leakage_to_positive") <= max_negative_leakage
        )
    ]

    if eligible:
        chosen = max(
            eligible,
            key=lambda row: (
                f(row, "positive_coverage"),
                f(row, "positive_recall"),
                f(row, "positive_precision"),
            ),
        )
        chosen["selection_status"] = "MEETS_INITIAL_SAFETY_TARGET"
        return chosen

    fallback = max(
        search_rows,
        key=lambda row: (
            f(row, "positive_precision"),
            f(row, "positive_coverage"),
        ),
    )
    fallback["selection_status"] = "FALLBACK_BEST_AVAILABLE_DO_NOT_APPLY_FULL_CORPUS_YET"
    return fallback


def stage_model_oof(
    *,
    x: np.ndarray,
    y: np.ndarray,
    positive_label: str,
    stage_name: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    probabilities_by_model = {}
    all_search_rows = []

    for model_name, model in make_models().items():
        probabilities = positive_probability_from_oof(
            model,
            x,
            y,
            positive_label,
        )

        probabilities_by_model[model_name] = probabilities

        labels = sorted(set(y.tolist()))
        negative_candidates = [label for label in labels if label != positive_label]

        if len(negative_candidates) != 1:
            raise SystemExit(f"Expected binary labels for {stage_name}, got {labels}")

        all_search_rows.extend(
            binary_threshold_rows(
                model_name=model_name,
                stage_name=stage_name,
                probabilities=probabilities,
                y_true=y,
                positive_label=positive_label,
                negative_label=negative_candidates[0],
            )
        )

    return probabilities_by_model, all_search_rows


def route_confusion_rows(
    gold: list[str],
    pred: list[str],
    labels: list[str],
) -> list[dict[str, Any]]:
    matrix = confusion_matrix(gold, pred, labels=labels)
    rows = []

    for i, gold_label in enumerate(labels):
        row = {"gold_route": gold_label}
        for j, pred_label in enumerate(labels):
            row[f"pred_{pred_label}"] = int(matrix[i, j])
        rows.append(row)

    return rows


def final_safety_metrics(gold: list[str], pred: list[str]) -> dict[str, float]:
    gold_arr = np.asarray(gold, dtype=object)
    pred_arr = np.asarray(pred, dtype=object)

    pred_usable = np.isin(pred_arr, list(USABLE_ROUTES))
    gold_usable = np.isin(gold_arr, list(USABLE_ROUTES))

    pred_direct = pred_arr == "DIRECT_HEALTHY"
    gold_direct = gold_arr == "DIRECT_HEALTHY"

    pred_reject = pred_arr == "REJECT"
    gold_reject = gold_arr == "REJECT"

    gold_quarantine = gold_arr == "QUARANTINE"

    return {
        "exact_route_accuracy": float(np.mean(gold_arr == pred_arr)),
        "combined_usable_precision": safe_div(
            int(np.sum(pred_usable & gold_usable)),
            int(np.sum(pred_usable)),
        ),
        "combined_usable_recall": safe_div(
            int(np.sum(pred_usable & gold_usable)),
            int(np.sum(gold_usable)),
        ),
        "nonusable_leakage_to_usable": safe_div(
            int(np.sum(pred_usable & (~gold_usable))),
            int(np.sum(~gold_usable)),
        ),
        "direct_healthy_precision": safe_div(
            int(np.sum(pred_direct & gold_direct)),
            int(np.sum(pred_direct)),
        ),
        "direct_healthy_recall": safe_div(
            int(np.sum(pred_direct & gold_direct)),
            int(np.sum(gold_direct)),
        ),
        "reject_precision": safe_div(
            int(np.sum(pred_reject & gold_reject)),
            int(np.sum(pred_reject)),
        ),
        "reject_recall": safe_div(
            int(np.sum(pred_reject & gold_reject)),
            int(np.sum(gold_reject)),
        ),
        "quarantine_recall": safe_div(
            int(np.sum((pred_arr == "QUARANTINE") & gold_quarantine)),
            int(np.sum(gold_quarantine)),
        ),
    }


def model_level_summary(
    search_rows: list[dict[str, Any]],
    selected: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped = defaultdict(list)

    for row in search_rows:
        grouped[str(row["model_name"])].append(row)

    out = []

    for model_name, rows in sorted(grouped.items()):
        best_precision = max(f(row, "positive_precision") for row in rows)
        best_f1 = max(f(row, "positive_f1") for row in rows)
        best_coverage_at_90p = max(
            [
                f(row, "positive_coverage")
                for row in rows
                if f(row, "positive_precision") >= 0.90
            ]
            or [0.0]
        )

        out.append(
            {
                "stage": rows[0]["stage"],
                "model_name": model_name,
                "best_precision_any_threshold": f"{best_precision:.6f}",
                "best_f1_any_threshold": f"{best_f1:.6f}",
                "best_coverage_at_precision_90pct": f"{best_coverage_at_90p:.6f}",
                "selected_for_stage": "YES"
                if model_name == selected["model_name"]
                else "NO",
            }
        )

    return out


def main() -> None:
    rows, fieldnames = read_csv(INPUT_PATH)
    validation_rows, validation_fieldnames = read_csv(VALIDATION_BLIND_PATH)

    label_col = resolve_column(
        fieldnames,
        ["human_health_label", "reference_label", "gold_label"],
    )
    id_col = resolve_column(
        fieldnames,
        ["review_sample_id", "sample_id", "id"],
    )
    text_col = resolve_column(
        fieldnames,
        ["sentence_text", "text"],
    )

    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(f"Expected {EXPECTED_ROWS} calibration rows, found {len(rows)}")

    label_counts = Counter((row.get(label_col) or "").strip() for row in rows)

    for label, expected_count in EXPECTED_LABEL_COUNTS.items():
        found = label_counts.get(label, 0)
        if found != expected_count:
            raise SystemExit(
                f"Label count mismatch for {label}: expected {expected_count}, found {found}"
            )

    exposed_validation_label_cols = [
        col
        for col in validation_fieldnames
        if col in {"human_health_label", "reference_label", "gold_label"}
        or col.startswith("gold_")
    ]

    if exposed_validation_label_cols:
        raise SystemExit(
            "Validation file is not blind. Exposed label columns: "
            + str(exposed_validation_label_cols)
        )

    feature_columns = select_numeric_features(rows, fieldnames)
    x_all = build_matrix(rows, feature_columns)

    labels = np.asarray(
        [(row.get(label_col) or "").strip() for row in rows],
        dtype=object,
    )

    gold_routes = np.asarray(
        [LABEL_TO_ROUTE[label] for label in labels],
        dtype=object,
    )

    y_stage1 = np.asarray(
        [
            "USABLE" if route in USABLE_ROUTES else "NONUSABLE"
            for route in gold_routes
        ],
        dtype=object,
    )

    stage1_prob_by_model, stage1_search = stage_model_oof(
        x=x_all,
        y=y_stage1,
        positive_label="USABLE",
        stage_name="stage1_usable_gate",
    )
    selected_stage1 = choose_stage1(stage1_search)

    usable_indices = np.asarray([route in USABLE_ROUTES for route in gold_routes])
    x_usable = x_all[usable_indices]
    y_stage2a = np.asarray(gold_routes[usable_indices], dtype=object)
    usable_original_indices = np.where(usable_indices)[0]

    stage2a_prob_by_model, stage2a_search = stage_model_oof(
        x=x_usable,
        y=y_stage2a,
        positive_label="DIRECT_HEALTHY",
        stage_name="stage2a_direct_healthy_gate",
    )
    selected_stage2a = choose_precision_stage(
        stage2a_search,
        min_precision=0.90,
        max_negative_leakage=0.10,
    )

    nonusable_indices = np.asarray([route in NONUSABLE_ROUTES for route in gold_routes])
    x_nonusable = x_all[nonusable_indices]
    y_stage2b = np.asarray(gold_routes[nonusable_indices], dtype=object)
    nonusable_original_indices = np.where(nonusable_indices)[0]

    stage2b_prob_by_model, stage2b_search = stage_model_oof(
        x=x_nonusable,
        y=y_stage2b,
        positive_label="REJECT",
        stage_name="stage2b_reject_gate",
    )
    selected_stage2b = choose_precision_stage(
        stage2b_search,
        min_precision=0.85,
        max_negative_leakage=0.10,
    )

    stage1_model = selected_stage1["model_name"]
    stage2a_model = selected_stage2a["model_name"]
    stage2b_model = selected_stage2b["model_name"]

    stage1_threshold = float(selected_stage1["threshold"])
    stage2a_threshold = float(selected_stage2a["threshold"])
    stage2b_threshold = float(selected_stage2b["threshold"])

    p_usable = stage1_prob_by_model[stage1_model]

    p_direct_all = np.full(len(rows), np.nan, dtype=float)
    p_reject_all = np.full(len(rows), np.nan, dtype=float)

    p_direct_all[usable_original_indices] = stage2a_prob_by_model[stage2a_model]
    p_reject_all[nonusable_original_indices] = stage2b_prob_by_model[stage2b_model]

    predicted_routes = []

    for idx in range(len(rows)):
        if p_usable[idx] >= stage1_threshold:
            if not np.isnan(p_direct_all[idx]) and p_direct_all[idx] >= stage2a_threshold:
                predicted_routes.append("DIRECT_HEALTHY")
            else:
                predicted_routes.append("REPAIRABLE_USABLE")
        else:
            if not np.isnan(p_reject_all[idx]) and p_reject_all[idx] >= stage2b_threshold:
                predicted_routes.append("REJECT")
            else:
                predicted_routes.append("QUARANTINE")

    safety = final_safety_metrics(gold_routes.tolist(), predicted_routes)
    route_labels = [
        "DIRECT_HEALTHY",
        "REPAIRABLE_USABLE",
        "QUARANTINE",
        "REJECT",
    ]

    final_models = {
        "stage1_usable_gate": clone(make_models()[stage1_model]).fit(x_all, y_stage1),
        "stage2a_direct_healthy_gate": clone(make_models()[stage2a_model]).fit(
            x_usable,
            y_stage2a,
        ),
        "stage2b_reject_gate": clone(make_models()[stage2b_model]).fit(
            x_nonusable,
            y_stage2b,
        ),
    }

    with MODELS_PATH.open("wb") as f:
        pickle.dump(
            {
                "feature_columns": feature_columns,
                "selected_models": {
                    "stage1_usable_gate": stage1_model,
                    "stage2a_direct_healthy_gate": stage2a_model,
                    "stage2b_reject_gate": stage2b_model,
                },
                "thresholds": {
                    "stage1_usable_gate": stage1_threshold,
                    "stage2a_direct_healthy_gate": stage2a_threshold,
                    "stage2b_reject_gate": stage2b_threshold,
                },
                "models": final_models,
            },
            f,
        )

    oof_rows = []

    for idx, row in enumerate(rows):
        oof_rows.append(
            {
                "review_sample_id": row.get(id_col, ""),
                "gold_health_label": labels[idx],
                "gold_route": gold_routes[idx],
                "predicted_route": predicted_routes[idx],
                "p_stage1_usable": f"{p_usable[idx]:.6f}",
                "p_stage2a_direct_healthy": ""
                if np.isnan(p_direct_all[idx])
                else f"{p_direct_all[idx]:.6f}",
                "p_stage2b_reject": ""
                if np.isnan(p_reject_all[idx])
                else f"{p_reject_all[idx]:.6f}",
                "sentence_text": row.get(text_col, ""),
            }
        )

    write_csv(
        OOF_PATH,
        oof_rows,
        [
            "review_sample_id",
            "gold_health_label",
            "gold_route",
            "predicted_route",
            "p_stage1_usable",
            "p_stage2a_direct_healthy",
            "p_stage2b_reject",
            "sentence_text",
        ],
    )

    write_csv(STAGE1_SEARCH_PATH, stage1_search, list(stage1_search[0].keys()))
    write_csv(STAGE2A_SEARCH_PATH, stage2a_search, list(stage2a_search[0].keys()))
    write_csv(STAGE2B_SEARCH_PATH, stage2b_search, list(stage2b_search[0].keys()))

    model_summary = []
    model_summary.extend(model_level_summary(stage1_search, selected_stage1))
    model_summary.extend(model_level_summary(stage2a_search, selected_stage2a))
    model_summary.extend(model_level_summary(stage2b_search, selected_stage2b))

    write_csv(MODEL_COMPARISON_PATH, model_summary, list(model_summary[0].keys()))

    confusion_rows = route_confusion_rows(
        gold_routes.tolist(),
        predicted_routes,
        route_labels,
    )
    write_csv(CONFUSION_PATH, confusion_rows, list(confusion_rows[0].keys()))

    feature_rows = [
        {"feature_index": i + 1, "feature_name": col}
        for i, col in enumerate(feature_columns)
    ]
    write_csv(FEATURES_PATH, feature_rows, ["feature_index", "feature_name"])

    policy = {
        "policy_name": "hierarchical_text_router_v1_1",
        "status": "calibration_trained_not_validation_approved",
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "input_path": str(INPUT_PATH),
        "validation_blind_path": str(VALIDATION_BLIND_PATH),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "selected_stage1": selected_stage1,
        "selected_stage2a": selected_stage2a,
        "selected_stage2b": selected_stage2b,
        "safety_metrics_oof_calibration": safety,
    }

    POLICY_PATH.write_text(
        json.dumps(policy, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    selected_ok = all(
        item.get("selection_status") == "MEETS_INITIAL_SAFETY_TARGET"
        for item in [selected_stage1, selected_stage2a, selected_stage2b]
    )

    summary_lines = [
        "# Hierarchical Text Router v1.1 — Calibration Training Summary",
        "",
        "## Selected stages",
        "",
        f"- Stage 1 usable gate: `{stage1_model}` at threshold `{stage1_threshold:.2f}` — {selected_stage1['selection_status']}",
        f"- Stage 2A direct-healthy gate: `{stage2a_model}` at threshold `{stage2a_threshold:.2f}` — {selected_stage2a['selection_status']}",
        f"- Stage 2B reject gate: `{stage2b_model}` at threshold `{stage2b_threshold:.2f}` — {selected_stage2b['selection_status']}",
        "",
        "## Calibration out-of-fold safety metrics",
        "",
    ]

    for key, value in safety.items():
        summary_lines.append(f"- {key}: {value:.4%}")

    summary_lines.extend(
        [
            "",
            "## Decision",
            "",
            "Do not apply this router to the full corpus until the reserved validation split is evaluated.",
            "Validation labels remained blind in this run.",
        ]
    )

    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("=" * 112)
    print("HIERARCHICAL TEXT ROUTER V1.1 — CALIBRATION TRAINING")
    print("=" * 112)

    print("\nINPUT CHECKS")
    print("-" * 112)
    print(f"calibration_rows: {len(rows)}")
    print(f"validation_rows_blind: {len(validation_rows)}")
    print(f"feature_count: {len(feature_columns)}")
    print("validation_labels_exposed: NO")
    print("master_csv_modified: NO")
    print("full_corpus_filtering_applied: NO")

    print("\nSELECTED STAGES")
    print("-" * 112)

    for name, selected in [
        ("stage1_usable_gate", selected_stage1),
        ("stage2a_direct_healthy_gate", selected_stage2a),
        ("stage2b_reject_gate", selected_stage2b),
    ]:
        print(
            f"{name:<32} model={selected['model_name']:<24} "
            f"threshold={float(selected['threshold']):.2f} "
            f"precision={float(selected['positive_precision']):.2%} "
            f"recall={float(selected['positive_recall']):.2%} "
            f"neg_leak={float(selected['negative_leakage_to_positive']):.2%} "
            f"status={selected['selection_status']}"
        )

    print("\nHIERARCHICAL OOF SAFETY METRICS — CALIBRATION")
    print("-" * 112)

    for key, value in safety.items():
        print(f"{key:<36}: {value:.2%}")

    print("\nPREDICTED ROUTE DISTRIBUTION")
    print("-" * 112)

    pred_counts = Counter(predicted_routes)

    for route in route_labels:
        count = pred_counts.get(route, 0)
        pct = safe_div(count, len(rows))
        print(f"{route:<28}: {count:>3}/{len(rows)} ({pct:>6.2%})")

    print("\nOUTPUT FILES")
    print("-" * 112)

    for path in [
        OOF_PATH,
        STAGE1_SEARCH_PATH,
        STAGE2A_SEARCH_PATH,
        STAGE2B_SEARCH_PATH,
        MODEL_COMPARISON_PATH,
        CONFUSION_PATH,
        FEATURES_PATH,
        POLICY_PATH,
        MODELS_PATH,
        SUMMARY_PATH,
    ]:
        print(path)

    print("\nSAFETY DECISION")
    print("-" * 112)

    if (
        selected_ok
        and safety["combined_usable_precision"] >= 0.90
        and safety["nonusable_leakage_to_usable"] <= 0.10
    ):
        print("CALIBRATION_ROUTER_STATUS: STRONG_ENOUGH_FOR_RESERVED_VALIDATION_CHECK")
        print("STATUS: READY_TO_RUN_FROZEN_ROUTER_ON_BLIND_VALIDATION_V1_1")
    else:
        print("CALIBRATION_ROUTER_STATUS: NOT_SAFE_ENOUGH_YET")
        print("STATUS: READY_TO_AUDIT_HIERARCHICAL_ROUTER_ERRORS_V1_1")


if __name__ == "__main__":
    main()
