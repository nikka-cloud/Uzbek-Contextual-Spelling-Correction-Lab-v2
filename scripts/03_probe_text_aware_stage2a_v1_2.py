from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1"
)

INPUT_PATH = (
    ROOT
    / "text_health_features_v1"
    / "calibration_text_health_features_labeled_v1.csv"
)

OUTPUT_DIR = ROOT / "text_aware_router_v1_2_probe"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_PATH = OUTPUT_DIR / "stage2a_text_aware_threshold_search_v1_2.csv"
OOF_PATH = OUTPUT_DIR / "stage2a_text_aware_oof_predictions_v1_2.csv"

RANDOM_SEED = 42
N_SPLITS = 5

POSITIVE = "DIRECT_HEALTHY"
NEGATIVE = "REPAIRABLE_USABLE"

THRESHOLDS = [
    round(x, 2)
    for x in np.arange(0.50, 0.96, 0.05)
] + [0.97, 0.99]


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column: str):
        self.column = column

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return [str(row.get(self.column, "") or "") for row in x]


class NumericSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        matrix = []
        for row in x:
            values = []
            for col in self.columns:
                value = parse_float(row.get(col))
                values.append(np.nan if value is None else value)
            matrix.append(values)
        return np.asarray(matrix, dtype=float)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise SystemExit(f"No rows found: {path}")
    return rows, fields


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    all_fields = list(fields)
    seen = set(all_fields)
    for row in rows:
        for key in row:
            if key not in seen:
                all_fields.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve(fields: list[str], candidates: list[str]) -> str:
    for col in candidates:
        if col in fields:
            return col
    raise SystemExit(f"Could not resolve column from: {candidates}\nAvailable: {fields}")


def parse_float(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    low = text.lower()

    if low in {"true", "yes", "pass"}:
        return 1.0

    if low in {"false", "no", "none", "blank", "n/a", "na"}:
        return 0.0

    if low.endswith("%"):
        try:
            return float(low[:-1]) / 100.0
        except ValueError:
            return None

    try:
        out = float(text)
    except ValueError:
        return None

    return out if math.isfinite(out) else None


def excluded(col: str) -> bool:
    low = col.lower()

    exact = {
        "human_health_label",
        "human_health_notes",
        "gold_health_label",
        "gold_health_notes",
        "gold_label_hidden",
        "gold_reference_lane",
        "sentence_text",
        "review_sample_id",
        "sentence_id",
        "source_record_id",
    }

    bad_parts = [
        "label",
        "note",
        "text",
        "sample_id",
        "sentence_id",
        "record_id",
        "gold",
        "route",
    ]

    return col in exact or any(part in low for part in bad_parts)


def select_numeric(rows: list[dict[str, str]], fields: list[str]) -> list[str]:
    selected = []

    for col in fields:
        if excluded(col):
            continue

        values = [parse_float(row.get(col)) for row in rows]
        non_missing = [v for v in values if v is not None]

        if len(non_missing) / len(rows) >= 0.98 and len(set(non_missing)) > 1:
            selected.append(col)

    return selected


def make_model(mode: str, text_col: str, numeric_cols: list[str], c_value: float):
    branches = []

    if "char" in mode:
        branches.append(
            (
                "char",
                Pipeline(
                    [
                        ("text", TextSelector(text_col)),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 6),
                                min_df=2,
                                max_features=12000,
                                lowercase=True,
                            ),
                        ),
                    ]
                ),
            )
        )

    if "word" in mode:
        branches.append(
            (
                "word",
                Pipeline(
                    [
                        ("text", TextSelector(text_col)),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                min_df=2,
                                max_features=8000,
                                lowercase=True,
                                token_pattern=r"(?u)\b[\w'`‘’ʼ-]{2,}\b",
                            ),
                        ),
                    ]
                ),
            )
        )

    if "numeric" in mode:
        branches.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("select", NumericSelector(numeric_cols)),
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
            )
        )

    return Pipeline(
        [
            ("features", FeatureUnion(branches)),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    max_iter=3000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def safe_div(a: int | float, b: int | float) -> float:
    return float(a) / float(b) if b else 0.0


def threshold_rows(model_name: str, prob: np.ndarray, y: np.ndarray):
    rows = []
    total_pos = int(np.sum(y == POSITIVE))
    total_neg = int(np.sum(y == NEGATIVE))

    for thr in THRESHOLDS:
        pred_pos = prob >= thr
        true_pos = y == POSITIVE
        true_neg = y == NEGATIVE

        tp = int(np.sum(pred_pos & true_pos))
        fp = int(np.sum(pred_pos & true_neg))
        fn = int(np.sum((~pred_pos) & true_pos))
        tn = int(np.sum((~pred_pos) & true_neg))

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        leak = safe_div(fp, total_neg)
        coverage = safe_div(tp + fp, len(y))

        rows.append(
            {
                "model_name": model_name,
                "threshold": f"{thr:.2f}",
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "positive_precision": f"{precision:.6f}",
                "positive_recall": f"{recall:.6f}",
                "positive_f1": f"{f1:.6f}",
                "repairable_leakage_to_direct": f"{leak:.6f}",
                "direct_healthy_coverage": f"{coverage:.6f}",
                "total_direct_healthy": total_pos,
                "total_repairable": total_neg,
            }
        )

    return rows


def f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def main() -> None:
    print("=" * 120, flush=True)
    print("STAGE 2A TEXT-AWARE ROUTER PROBE V1.2", flush=True)
    print("=" * 120, flush=True)

    rows, fields = read_csv(INPUT_PATH)

    id_col = resolve(fields, ["review_sample_id", "sample_id", "id"])
    text_col = resolve(fields, ["sentence_text", "text"])
    label_col = resolve(fields, ["human_health_label", "reference_label", "gold_label"])

    usable_rows = []
    for row in rows:
        label = (row.get(label_col) or "").strip()
        if label == "HEALTHY":
            new = dict(row)
            new["_route"] = POSITIVE
            usable_rows.append(new)
        elif label == "MINOR_DAMAGE_BUT_USABLE":
            new = dict(row)
            new["_route"] = NEGATIVE
            usable_rows.append(new)

    counts = Counter(row["_route"] for row in usable_rows)
    numeric_cols = select_numeric(usable_rows, fields)

    print("\nINPUT", flush=True)
    print("-" * 120, flush=True)
    print("usable_rows:", len(usable_rows), flush=True)
    print("direct_healthy_gold:", counts[POSITIVE], flush=True)
    print("repairable_gold:", counts[NEGATIVE], flush=True)
    print("numeric_feature_count:", len(numeric_cols), flush=True)

    if counts[POSITIVE] != 239 or counts[NEGATIVE] != 251:
        raise SystemExit(f"Unexpected counts: {counts}")

    specs = [
        ("char_word_numeric_c0_5", "char_word_numeric", 0.5),
        ("char_word_numeric_c1_0", "char_word_numeric", 1.0),
        ("char_word_numeric_c2_0", "char_word_numeric", 2.0),
        ("char_numeric_c1_0", "char_numeric", 1.0),
        ("word_numeric_c1_0", "word_numeric", 1.0),
        ("numeric_only_c1_0", "numeric", 1.0),
    ]

    x = usable_rows
    y = np.asarray([row["_route"] for row in usable_rows], dtype=object)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    all_rows = []
    probs_by_model = {}

    print("\nTRAINING MODELS", flush=True)
    print("-" * 120, flush=True)

    for model_name, mode, c_value in specs:
        print("running:", model_name, flush=True)

        model = make_model(mode, text_col, numeric_cols, c_value)

        proba = cross_val_predict(
            model,
            x,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=None,
        )

        fitted = clone(model).fit(x, y)
        classes = list(fitted.classes_)
        pos_idx = classes.index(POSITIVE)

        pos_prob = proba[:, pos_idx]
        probs_by_model[model_name] = pos_prob
        all_rows.extend(threshold_rows(model_name, pos_prob, y))

    write_csv(
        SEARCH_PATH,
        all_rows,
        [
            "model_name",
            "threshold",
            "tp",
            "fp",
            "fn",
            "tn",
            "positive_precision",
            "positive_recall",
            "positive_f1",
            "repairable_leakage_to_direct",
            "direct_healthy_coverage",
            "total_direct_healthy",
            "total_repairable",
        ],
    )

    candidates_90 = [r for r in all_rows if f(r, "positive_precision") >= 0.90]
    candidates_85 = [r for r in all_rows if f(r, "positive_precision") >= 0.85]
    candidates_80 = [r for r in all_rows if f(r, "positive_precision") >= 0.80]

    def rank(row):
        return (
            f(row, "direct_healthy_coverage"),
            f(row, "positive_recall"),
            f(row, "positive_precision"),
        )

    best_90 = max(candidates_90, key=rank) if candidates_90 else None
    best_85 = max(candidates_85, key=rank) if candidates_85 else None
    best_80 = max(candidates_80, key=rank) if candidates_80 else None

    selected = best_85 or best_80 or best_90
    if selected:
        model_name = selected["model_name"]
        threshold = float(selected["threshold"])
        probs = probs_by_model[model_name]

        oof_rows = []
        for i, row in enumerate(usable_rows):
            pred = POSITIVE if probs[i] >= threshold else NEGATIVE
            oof_rows.append(
                {
                    "review_sample_id": row.get(id_col, ""),
                    "gold_route": row["_route"],
                    "predicted_route": pred,
                    "p_direct_healthy": f"{probs[i]:.6f}",
                    "sentence_text": row.get(text_col, ""),
                }
            )

        write_csv(
            OOF_PATH,
            oof_rows,
            [
                "review_sample_id",
                "gold_route",
                "predicted_route",
                "p_direct_healthy",
                "sentence_text",
            ],
        )

    print("\nBEST THRESHOLD CANDIDATES", flush=True)
    print("-" * 120, flush=True)

    for name, row in [
        ("precision >= 90%", best_90),
        ("precision >= 85%", best_85),
        ("precision >= 80%", best_80),
    ]:
        if row is None:
            print(f"{name:<18}: NONE", flush=True)
            continue

        print(
            f"{name:<18}: "
            f"model={row['model_name']:<24} "
            f"thr={float(row['threshold']):.2f} "
            f"precision={f(row,'positive_precision'):>7.2%} "
            f"recall={f(row,'positive_recall'):>7.2%} "
            f"coverage={f(row,'direct_healthy_coverage'):>7.2%} "
            f"repairable_leak={f(row,'repairable_leakage_to_direct'):>7.2%} "
            f"tp={row['tp']} fp={row['fp']}",
            flush=True,
        )

    print("\nTOP DIRECT_HEALTHY OPTIONS AT PRECISION >= 80%", flush=True)
    print("-" * 120, flush=True)

    for row in sorted(candidates_80, key=rank, reverse=True)[:20]:
        print(
            f"{row['model_name']:<24} "
            f"thr={float(row['threshold']):.2f} "
            f"precision={f(row,'positive_precision'):>7.2%} "
            f"recall={f(row,'positive_recall'):>7.2%} "
            f"coverage={f(row,'direct_healthy_coverage'):>7.2%} "
            f"leak={f(row,'repairable_leakage_to_direct'):>7.2%} "
            f"tp={row['tp']} fp={row['fp']}",
            flush=True,
        )

    print("\nOUTPUT FILES", flush=True)
    print("-" * 120, flush=True)
    print(SEARCH_PATH, flush=True)
    print(OOF_PATH, flush=True)

    print("\nSAFETY", flush=True)
    print("-" * 120, flush=True)
    print("calibration_only: YES", flush=True)
    print("validation_labels_used: NO", flush=True)
    print("full_corpus_filtering_applied: NO", flush=True)
    print("master_csv_modified: NO", flush=True)

    if best_90 and f(best_90, "direct_healthy_coverage") >= 0.10:
        print("\nPROBE_DECISION: TEXT_FEATURES_HELP_STAGE2A", flush=True)
        print("STATUS: READY_TO_BUILD_FULL_TEXT_AWARE_ROUTER_V1_2", flush=True)
    elif best_85 and f(best_85, "direct_healthy_coverage") >= 0.15:
        print("\nPROBE_DECISION: TEXT_FEATURES_PARTIALLY_HELP_STAGE2A", flush=True)
        print("STATUS: READY_TO_REVIEW_STAGE2A_TEXT_ERRORS_BEFORE_FULL_V1_2", flush=True)
    else:
        print("\nPROBE_DECISION: TEXT_FEATURES_NOT_ENOUGH_YET", flush=True)
        print("STATUS: READY_TO_AUDIT_DIRECT_HEALTHY_VS_REPAIRABLE_LABEL_BOUNDARY", flush=True)


if __name__ == "__main__":
    main()
