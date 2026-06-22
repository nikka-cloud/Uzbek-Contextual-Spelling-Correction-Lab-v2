from __future__ import annotations

import hashlib
import importlib.util
import math
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1"
)

VALIDATION_FEATURES = (
    ROOT
    / "text_health_features_v1"
    / "validation_text_health_features_blind_v1.csv"
)

MASTER = ROOT / "global_sentence_health_review_master_v1.csv"

ROUTER_MODELS = (
    ROOT
    / "hierarchical_text_router_v1_1"
    / "hierarchical_text_router_models_v1_1.pkl"
)

FROZEN_DECISION = (
    ROOT
    / "strict_clean_seed_extractor_v1_2"
    / "frozen_decision_v1_2"
    / "frozen_decision_v1_2.md"
)

V1_2_SCRIPT = Path("scripts/03_build_strict_clean_seed_extractor_v1_2.py")

OUT_DIR = ROOT / "strict_clean_seed_extractor_v1_2" / "validation_eval_v1_2"


LABEL_COLS = [
    "human_health_label",
    "gold_health_label",
]

JOIN_KEYS = [
    "review_sample_id",
    "review_row_number",
    "sentence_id",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_v1_2_module():
    spec = importlib.util.spec_from_file_location("strict_seed_v1_2", V1_2_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {V1_2_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pick_label_col(df: pd.DataFrame) -> str:
    for col in LABEL_COLS:
        if col in df.columns:
            return col
    raise ValueError(f"No label column found. Tried: {LABEL_COLS}")


def choose_join_key(validation: pd.DataFrame, master: pd.DataFrame) -> str:
    for key in JOIN_KEYS:
        if key not in validation.columns or key not in master.columns:
            continue

        val_unique = validation[key].astype(str).nunique(dropna=False) == len(validation)
        master_unique = master[key].astype(str).nunique(dropna=False) == len(master)

        if val_unique and master_unique:
            return key

    raise RuntimeError(
        "Could not find safe unique join key between validation features and master labels."
    )


def get_positive_class_probability(model, X) -> list[float]:
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Stage1 model does not support predict_proba().")

    proba = model.predict_proba(X)

    if not hasattr(model, "classes_"):
        if proba.shape[1] != 2:
            raise RuntimeError("No model.classes_ and probability shape is not binary.")
        return proba[:, 1].tolist()

    classes = list(model.classes_)
    normalized = [str(c).upper() for c in classes]

    positive_candidates = [
        "USABLE",
        "1",
        "TRUE",
        "DIRECT_HEALTHY",
        "REPAIRABLE_USABLE",
    ]

    pos_idx = None

    for cand in positive_candidates:
        if cand in normalized:
            pos_idx = normalized.index(cand)
            break

    if pos_idx is None:
        if len(classes) == 2:
            # Common sklearn binary fallback: second column is positive class.
            pos_idx = 1
        else:
            raise RuntimeError(f"Could not identify positive class from classes_: {classes}")

    return proba[:, pos_idx].tolist()


def pct(a: int, b: int) -> str:
    if b == 0:
        return "NA"
    return f"{100 * a / b:.2f}%"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    required = [
        VALIDATION_FEATURES,
        MASTER,
        ROUTER_MODELS,
        FROZEN_DECISION,
        V1_2_SCRIPT,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    frozen_text = FROZEN_DECISION.read_text(encoding="utf-8")

    if "`PASS`: v1.2 is frozen as validation-ready." not in frozen_text:
        raise RuntimeError("Frozen decision does not show PASS. Stop.")

    validation = pd.read_csv(VALIDATION_FEATURES, dtype=str).fillna("")
    master = pd.read_csv(MASTER, dtype=str).fillna("")

    if len(validation) != 301:
        raise RuntimeError(f"Expected 301 validation rows, got {len(validation)}")

    leaked_label_cols = [
        c for c in LABEL_COLS + ["gold_reference_lane", "gold_health_notes", "gold_label_hidden"]
        if c in validation.columns
    ]

    if leaked_label_cols:
        raise RuntimeError(
            f"Validation feature file already contains label/answer columns: {leaked_label_cols}"
        )

    with open(ROUTER_MODELS, "rb") as f:
        router_bundle = pickle.load(f)

    feature_columns = router_bundle["feature_columns"]
    stage1_model = router_bundle["models"]["stage1_usable_gate"]

    missing_features = [c for c in feature_columns if c not in validation.columns]
    if missing_features:
        raise RuntimeError(f"Validation is missing router feature columns: {missing_features}")

    X_val = validation[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    validation = validation.copy()
    validation["p_stage1_usable"] = get_positive_class_probability(stage1_model, X_val)

    # Load the exact frozen v1.2 rule code.
    v1_2 = load_v1_2_module()

    profile = None
    for p in v1_2.PROFILES:
        if p["name"] == "v1_2_ultra_plus_token_blockers":
            profile = p
            break

    if profile is None:
        raise RuntimeError("Could not find frozen profile v1_2_ultra_plus_token_blockers")

    text_col = v1_2.pick_col(validation, v1_2.TEXT_COLS)

    validation["v1_2_reasons"] = validation.apply(
        lambda row: "|".join(
            v1_2.strict_reasons(
                text=row[text_col],
                p_usable=float(row["p_stage1_usable"]),
                profile=profile,
            )
        ),
        axis=1,
    )

    validation["strict_clean_seed_selected_v1_2"] = validation["v1_2_reasons"].eq("")

    # Save blind predictions BEFORE label join.
    blind_pred_path = OUT_DIR / "validation_predictions_blind_v1_2.csv"
    validation.to_csv(blind_pred_path, index=False)

    join_key = choose_join_key(validation, master)
    label_col = pick_label_col(master)

    label_cols_to_join = [
        join_key,
        label_col,
    ]

    for extra in ["gold_reference_lane", "gold_health_notes", "gold_label_hidden"]:
        if extra in master.columns and extra not in label_cols_to_join:
            label_cols_to_join.append(extra)

    labeled = validation.merge(
        master[label_cols_to_join],
        on=join_key,
        how="left",
        validate="one_to_one",
    )

    missing_labels = int(labeled[label_col].eq("").sum())
    if missing_labels:
        raise RuntimeError(f"Missing labels after join: {missing_labels}")

    labeled["is_gold_healthy"] = labeled[label_col].eq("HEALTHY")

    selected = labeled[labeled["strict_clean_seed_selected_v1_2"]].copy()
    false_clean = selected[~selected["is_gold_healthy"]].copy()
    missed_healthy = labeled[
        labeled["is_gold_healthy"] & ~labeled["strict_clean_seed_selected_v1_2"]
    ].copy()

    selected_n = len(selected)
    selected_healthy = int(selected["is_gold_healthy"].sum())
    false_n = len(false_clean)
    total_healthy = int(labeled["is_gold_healthy"].sum())

    reason_counter = Counter()

    for reasons in labeled["v1_2_reasons"]:
        if not reasons:
            reason_counter["SELECTED"] += 1
        else:
            for reason in reasons.split("|"):
                reason_counter[reason] += 1

    labeled.to_csv(OUT_DIR / "validation_predictions_labeled_eval_v1_2.csv", index=False)
    selected.to_csv(OUT_DIR / "validation_selected_candidates_v1_2.csv", index=False)
    false_clean.to_csv(OUT_DIR / "validation_false_clean_v1_2.csv", index=False)
    missed_healthy.to_csv(OUT_DIR / "validation_missed_healthy_v1_2.csv", index=False)

    review = selected.copy()
    review.insert(0, "manual_review_label", "")
    review.insert(1, "manual_review_notes", "")

    review_cols = [
        "manual_review_label",
        "manual_review_notes",
        join_key,
        label_col,
        "p_stage1_usable",
        text_col,
        "v1_2_reasons",
    ]

    review[review_cols].to_csv(
        OUT_DIR / "manual_review_validation_selected_v1_2.csv",
        index=False,
    )

    lines = []
    lines.append("# Frozen v1.2 Strict Clean Seed Validation Evaluation")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- validation_rows: `301`")
    lines.append("- validation_features: `blind before prediction`")
    lines.append("- profile: `v1_2_ultra_plus_token_blockers`")
    lines.append("- full_corpus_touched: `NO`")
    lines.append("- correction_pairs_generated: `NO`")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- validation_features_sha256: `{sha256(VALIDATION_FEATURES)}`")
    lines.append(f"- router_models_sha256: `{sha256(ROUTER_MODELS)}`")
    lines.append(f"- frozen_decision_sha256: `{sha256(FROZEN_DECISION)}`")
    lines.append(f"- v1_2_script_sha256: `{sha256(V1_2_SCRIPT)}`")
    lines.append(f"- blind_prediction_file: `{blind_pred_path}`")
    lines.append("")
    lines.append("## Label join")
    lines.append("")
    lines.append(f"- join_key: `{join_key}`")
    lines.append(f"- label_col: `{label_col}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(f"- selected_candidates: `{selected_n}`")
    lines.append(f"- selected_gold_healthy: `{selected_healthy}`")
    lines.append(f"- false_clean_rows: `{false_n}`")
    lines.append(f"- total_gold_healthy_validation: `{total_healthy}`")
    lines.append(f"- clean_seed_precision: `{pct(selected_healthy, selected_n)}`")
    lines.append(f"- clean_seed_recall: `{pct(selected_healthy, total_healthy)}`")
    lines.append("")
    lines.append("## Selected label distribution")
    lines.append("")

    if selected_n:
        for label, count in selected[label_col].value_counts().items():
            lines.append(f"- {label}: `{count}`")
    else:
        lines.append("- NONE: `0`")

    lines.append("")
    lines.append("## Reason counts")
    lines.append("")

    for reason, count in reason_counter.most_common():
        lines.append(f"- {reason}: `{count}`")

    lines.append("")
    lines.append("## Decision")
    lines.append("")

    precision = selected_healthy / selected_n if selected_n else 0.0

    if selected_n == 0:
        lines.append("`FAIL`: extractor selected zero validation rows.")
    elif precision >= 0.95:
        lines.append("`PASS`: frozen v1.2 reached >=95% validation precision.")
        lines.append("")
        lines.append("Next step: manually review selected validation candidates, then decide whether to use this extractor on a larger unlabeled corpus sample.")
    else:
        lines.append("`FAIL`: frozen v1.2 did not reach >=95% validation precision.")
        lines.append("")
        lines.append("Next step: inspect validation_false_clean_v1_2.csv and build stronger lexical quality gates.")

    lines.append("")
    lines.append("## False-clean examples")
    lines.append("")

    if false_n == 0:
        lines.append("None.")
    else:
        for _, row in false_clean.head(30).iterrows():
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.append(f"label: `{row[label_col]}`")
            lines.append("")
            if "gold_health_notes" in row.index:
                lines.append(f"notes: {row.get('gold_health_notes', '')}")
                lines.append("")
            lines.append(f"p_stage1_usable: `{row['p_stage1_usable']}`")
            lines.append("")
            lines.append(f"text: {row[text_col]}")

    summary = "\n".join(lines)
    (OUT_DIR / "validation_eval_summary_v1_2.md").write_text(summary, encoding="utf-8")

    print(summary)


if __name__ == "__main__":
    main()
