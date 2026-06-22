from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd


ROOT = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1"
)

CALIBRATION_LABELED = (
    ROOT
    / "text_health_features_v1"
    / "calibration_text_health_features_labeled_v1.csv"
)

OUT_DIR = ROOT / "strict_clean_seed_extractor_v1"


LABEL_HEALTHY = "HEALTHY"

NON_INPUT_LABEL_COLS = {
    "gold_health_label",
    "gold_health_notes",
    "gold_label_hidden",
    "gold_reference_lane",
    "human_health_label",
}


TEXT_COL_CANDIDATES = [
    "sentence_text",
    "text",
    "raw_text",
    "raw_sentence",
    "sentence",
    "original_sentence",
    "context_preview",
]

ID_COL_CANDIDATES = [
    "calibration_row_id",
    "review_row_id",
    "row_id",
    "sample_id",
    "global_sentence_id",
    "sentence_id",
    "sentence_hash",
    "text_hash",
    "source_record_id",
    "source_sentence_id",
]


def normalize_apostrophes(text: str) -> str:
    return (
        str(text)
        .replace("ʻ", "'")
        .replace("ʼ", "'")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("`", "'")
        .replace("´", "'")
    )


def find_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def safe_float(value) -> float:
    try:
        if value is None:
            return math.nan
        if isinstance(value, str) and not value.strip():
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def load_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def possible_stage1_usable_score_cols(cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        lc = c.lower()
        if "nonusable" in lc:
            continue
        if "usable" in lc and any(x in lc for x in ["prob", "score", "p_"]):
            out.append(c)
        elif lc in {"p_usable", "prob_usable", "usable_prob", "usable_score"}:
            out.append(c)

    def priority(c: str) -> tuple[int, int, str]:
        lc = c.lower()
        return (
            1 if "stage1" in lc else 0,
            1 if "oof" in lc else 0,
            c,
        )

    return sorted(out, key=priority, reverse=True)


def possible_pred_route_cols(cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        lc = c.lower()
        if "route" in lc and any(x in lc for x in ["pred", "predicted", "final"]):
            out.append(c)
        elif lc in {"predicted_route", "route_pred", "final_predicted_route"}:
            out.append(c)
    return out


def autodetect_router_oof(root: Path) -> Optional[Path]:
    scored: list[tuple[int, Path]] = []

    for path in root.rglob("*.csv"):
        name = path.name.lower()

        if "validation" in name:
            continue
        if "strict_clean_seed" in str(path).lower():
            continue
        if "global_sentence_health_review_master" in name:
            continue
        if "calibration_text_health_features_labeled" in name:
            continue

        if not any(x in name for x in ["oof", "router", "prediction", "pred"]):
            continue

        try:
            cols = load_csv_header(path)
        except Exception:
            continue

        usable_cols = possible_stage1_usable_score_cols(cols)
        route_cols = possible_pred_route_cols(cols)

        score = 0
        if usable_cols:
            score += 10
        if route_cols:
            score += 5
        if "oof" in name:
            score += 3
        if "hierarchical" in name or "router" in name:
            score += 2

        if score > 0:
            scored.append((score, path))

    if not scored:
        return None

    scored.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    return scored[0][1]


def merge_router_outputs(
    calibration_df: pd.DataFrame,
    router_path: Optional[Path],
) -> tuple[pd.DataFrame, Optional[str], Optional[str], Optional[Path], str]:
    if router_path is None:
        router_path = autodetect_router_oof(ROOT)

    if router_path is None or not router_path.exists():
        return calibration_df, None, None, None, "NO_ROUTER_OOF_FOUND"

    router_df = pd.read_csv(router_path, dtype=str).fillna("")
    router_cols = list(router_df.columns)

    usable_score_cols = possible_stage1_usable_score_cols(router_cols)
    route_cols = possible_pred_route_cols(router_cols)

    usable_score_col = usable_score_cols[0] if usable_score_cols else None
    pred_route_col = route_cols[0] if route_cols else None

    shared_keys = [
        c for c in ID_COL_CANDIDATES
        if c in calibration_df.columns and c in router_df.columns
    ]

    if shared_keys:
        key = shared_keys[0]
        keep_cols = [key]
        if usable_score_col:
            keep_cols.append(usable_score_col)
        if pred_route_col:
            keep_cols.append(pred_route_col)

        router_small = router_df[keep_cols].drop_duplicates(subset=[key])
        merged = calibration_df.merge(
            router_small,
            on=key,
            how="left",
            suffixes=("", "_router"),
        )
        return merged, usable_score_col, pred_route_col, router_path, f"MERGED_ON_{key}"

    if len(router_df) == len(calibration_df):
        merged = calibration_df.copy()
        if usable_score_col:
            merged[usable_score_col] = router_df[usable_score_col].values
        if pred_route_col:
            merged[pred_route_col] = router_df[pred_route_col].values
        return merged, usable_score_col, pred_route_col, router_path, "MERGED_BY_ROW_ORDER_SAME_LENGTH"

    return calibration_df, None, None, router_path, "ROUTER_FOUND_BUT_NOT_MERGED"


WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

URL_EMAIL_RE = re.compile(
    r"(https?://|www\.|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
    re.IGNORECASE,
)

LIST_START_RE = re.compile(
    r"^\s*(([-*•]+)|([0-9]+[.)])|([A-Za-z][.)])|([ivxlcdm]+[.)]))\s+",
    re.IGNORECASE,
)

REFERENCE_RE = re.compile(
    r"(\[[0-9,\s\-]+\]|\([0-9]{4}\)|§|№|\b\d+\s*[-–]\s*\d+\b)"
)

FORMULA_RE = re.compile(r"[=+*/<>]{1,}|\\[a-zA-Z]+|\{|\}")

MID_CAP_RE = re.compile(r"[a-zà-öø-ÿ][A-ZÀ-ÖØ-Þ]")

ABBREV_FRAGMENT_RE = re.compile(r"\b[A-Z]\.\s*(?:[A-Z]\.)?")

FORMAT_KEYWORDS_RE = re.compile(
    r"\b("
    r"jadval|rasm|formula|test|savol|javobni|belgilang|variant|"
    r"mavzu|reja|ilova|manba|adabiyot|sahifa|bet|bob|qism|"
    r"inv|fig|table|chapter|section"
    r")\b",
    re.IGNORECASE,
)

FOREIGN_TECH_TOKENS = {
    "popup",
    "popups",
    "popunder",
    "popunders",
    "browser",
    "cache",
    "account",
    "dashboard",
    "login",
    "support",
    "website",
    "online",
    "email",
    "mail",
    "click",
    "windows",
    "linux",
    "android",
    "iphone",
}

KNOWN_DEFECT_FRAGMENTS = [
    "stolstul",
    "savdosotiq",
    "yopyorug",
    "o'ksibo'ksib",
    "qipayotir",
    "faromush",
    "tashiluvchanli",
    "kermdiaga",
    "yiiz",
    "ismhoi",
    "tobe'ro",
    "dihed",
    "noyo lik",
    "alhamadoniy",
    "xonlikbuxorova",
]

SUFFIX_FRAGMENT_RE = re.compile(
    r"^(ning|ni|ga|da|dan|gi|gina|lik|ligi|lar|lari|siz|miz|dir|"
    r"chi|cha|kor|xona|bop|dek|day|dagi|dagi)$",
    re.IGNORECASE,
)


def text_stats(text: str) -> dict[str, float]:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return {
            "char_count": 0,
            "alpha_ratio": 0.0,
            "digit_ratio": 0.0,
            "punct_ratio": 0.0,
        }

    alpha = sum(c.isalpha() for c in chars)
    digit = sum(c.isdigit() for c in chars)
    punct = sum((not c.isalnum()) for c in chars)

    return {
        "char_count": len(chars),
        "alpha_ratio": alpha / len(chars),
        "digit_ratio": digit / len(chars),
        "punct_ratio": punct / len(chars),
    }


def token_vowel_ratio(tok: str) -> float:
    letters = [c for c in tok.lower() if c.isalpha()]
    if not letters:
        return 0.0
    vowels = set("aeiouo'ўüğöä")
    return sum(c in vowels for c in letters) / len(letters)


def is_foreign_like_token(tok: str) -> bool:
    t = normalize_apostrophes(tok).lower().strip(".,;:!?()[]{}\"")

    if len(t) < 4:
        return False

    if t in FOREIGN_TECH_TOKENS:
        return True

    if "w" in t:
        return True

    # Uzbek Latin uses "ch"; a naked c is suspicious in strict-clean mode.
    if "c" in t and "ch" not in t:
        return True

    # English-looking endings; strict seed should avoid these.
    if re.search(r"(tion|sion|ment|ing|able|less|ness|ware)$", t):
        return True

    return False


def has_broken_spacing_like_pattern(tokens: list[str]) -> bool:
    clean = [normalize_apostrophes(t).lower().strip(".,;:!?()[]{}\"") for t in tokens]

    for i in range(len(clean) - 1):
        a = clean[i]
        b = clean[i + 1]

        if not a or not b:
            continue

        # Example: "noyo likning" or "Tashiluvchanli gi"
        if len(a) >= 3 and SUFFIX_FRAGMENT_RE.match(b):
            return True

        if 2 <= len(a) <= 5 and re.match(
            r"^(lik|ning|dagi|dan|ga|da|ni|siz|miz|chi|cha)",
            b,
        ):
            return True

    return False


def has_ocr_like_token(tokens: list[str]) -> bool:
    for tok in tokens:
        t = normalize_apostrophes(tok).lower().strip(".,;:!?()[]{}\"")

        if len(t) < 8:
            continue

        vr = token_vowel_ratio(t)

        if len(t) >= 14 and (vr < 0.22 or vr > 0.72):
            return True

        if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", t):
            return True

        if re.search(r"([a-z])\1\1", t):
            return True

    return False


def defect_reasons(
    row: pd.Series,
    text_col: str,
    usable_score_col: Optional[str],
    pred_route_col: Optional[str],
    usable_threshold: float,
) -> list[str]:
    raw_text = str(row.get(text_col, ""))
    text = normalize_apostrophes(raw_text).strip()
    lowered = text.lower()
    tokens = WORD_RE.findall(text)
    clean_tokens = [
        normalize_apostrophes(t).lower().strip(".,;:!?()[]{}\"")
        for t in tokens
    ]

    reasons: list[str] = []
    stats = text_stats(text)
    word_count = len(tokens)

    # Router safety gate.
    if usable_score_col and usable_score_col in row.index:
        score = safe_float(row.get(usable_score_col))
        if math.isnan(score):
            reasons.append("missing_stage1_usable_score")
        elif score < usable_threshold:
            reasons.append(f"stage1_usable_score_below_{usable_threshold}")

    elif pred_route_col and pred_route_col in row.index:
        route = str(row.get(pred_route_col, "")).strip().upper()
        if route not in {"DIRECT_HEALTHY", "REPAIRABLE_USABLE"}:
            reasons.append("router_predicted_nonusable_or_unknown")
    else:
        reasons.append("router_signal_missing_rule_only_mode")

    # Basic sentence-shape gates.
    if word_count < 5:
        reasons.append("too_few_word_tokens")

    if word_count > 32:
        reasons.append("too_many_word_tokens")

    if stats["char_count"] < 25:
        reasons.append("too_short_chars")

    if stats["char_count"] > 220:
        reasons.append("too_long_chars")

    if text and text[-1] not in ".?!":
        reasons.append("missing_terminal_sentence_punctuation")

    if stats["alpha_ratio"] < 0.65:
        reasons.append("low_alpha_ratio")

    if stats["digit_ratio"] > 0.12:
        reasons.append("number_heavy")

    if stats["punct_ratio"] > 0.18:
        reasons.append("punctuation_heavy")

    # Format/list/reference/formula gates.
    if LIST_START_RE.search(text):
        reasons.append("list_or_enumeration_start")

    if URL_EMAIL_RE.search(text):
        reasons.append("url_or_email")

    if REFERENCE_RE.search(text):
        reasons.append("reference_or_legal_marker")

    if FORMULA_RE.search(text):
        reasons.append("formula_like_structure")

    if FORMAT_KEYWORDS_RE.search(text):
        reasons.append("format_or_exam_keyword")

    if text.count(";") >= 2:
        reasons.append("semicolon_list_like")

    if text.count(":") >= 2:
        reasons.append("colon_heavy_format_like")

    if ABBREV_FRAGMENT_RE.search(text):
        reasons.append("short_abbreviation_fragment")

    # Script/language gates.
    if CYRILLIC_RE.search(text):
        reasons.append("contains_cyrillic")

    if ARABIC_RE.search(text):
        reasons.append("contains_arabic_script")

    if any(is_foreign_like_token(t) for t in clean_tokens):
        reasons.append("foreign_or_technical_token")

    # Local defect gates.
    if MID_CAP_RE.search(text):
        reasons.append("mid_word_capitalization_or_merge")

    if has_broken_spacing_like_pattern(tokens):
        reasons.append("broken_spacing_inside_word_like")

    if has_ocr_like_token(tokens):
        reasons.append("ocr_like_token_shape")

    if re.search(r"\w\s+'\w|'\s+\w|\w'\s+\w|'{2,}", text):
        reasons.append("apostrophe_spacing_or_repetition_issue")

    for bad in KNOWN_DEFECT_FRAGMENTS:
        if bad in lowered:
            reasons.append("known_defect_fragment")
            break

    # Feature-file warning signals, if those columns exist.
    for col in row.index:
        lc = col.lower()
        value = row.get(col)

        if lc in NON_INPUT_LABEL_COLS:
            continue

        num = safe_float(value)

        if math.isnan(num):
            continue

        if "garbage" in lc and ("ratio" in lc or "rate" in lc) and num > 0:
            reasons.append(f"feature_warning_{col}")
            break

    for duplicate_col in [
        "exact_duplicate_count",
        "duplicate_count",
        "text_duplicate_count",
    ]:
        if duplicate_col in row.index:
            num = safe_float(row.get(duplicate_col))
            if not math.isnan(num) and num > 1:
                reasons.append("exact_duplicate_signal")
                break

    return sorted(set(reasons))


def pct(num: float, den: float) -> str:
    if den == 0:
        return "NA"
    return f"{(100 * num / den):.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration",
        type=Path,
        default=CALIBRATION_LABELED,
    )
    parser.add_argument(
        "--router-oof",
        type=Path,
        default=None,
        help="Optional explicit router OOF CSV path. If omitted, the script tries to auto-detect it.",
    )
    parser.add_argument(
        "--usable-threshold",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.calibration.exists():
        raise FileNotFoundError(f"Missing calibration file: {args.calibration}")

    df = pd.read_csv(args.calibration, dtype=str).fillna("")

    if len(df) != 699:
        print(f"WARNING: expected 699 calibration rows, found {len(df)}")

    label_col = find_first_existing_col(
        df,
        ["human_health_label", "gold_health_label"],
    )
    if not label_col:
        raise ValueError(
            "Could not find human/gold health label column in calibration file."
        )

    text_col = find_first_existing_col(df, TEXT_COL_CANDIDATES)
    if not text_col:
        raise ValueError(
            "Could not find sentence text column. "
            f"Tried: {TEXT_COL_CANDIDATES}"
        )

    # Guard against accidental validation use.
    for col in ["split", "review_split", "set_name", "gold_label_hidden"]:
        if col in df.columns:
            values = set(str(x).upper() for x in df[col].dropna().unique())
            if "VALIDATION" in values:
                raise RuntimeError(
                    f"Validation rows detected in {args.calibration}. Stop."
                )

    df, usable_score_col, pred_route_col, router_path, router_merge_status = merge_router_outputs(
        df,
        args.router_oof,
    )

    df["strict_defect_reasons"] = df.apply(
        lambda row: "|".join(
            defect_reasons(
                row=row,
                text_col=text_col,
                usable_score_col=usable_score_col,
                pred_route_col=pred_route_col,
                usable_threshold=args.usable_threshold,
            )
        ),
        axis=1,
    )

    df["strict_clean_seed_candidate"] = df["strict_defect_reasons"].eq("")
    df["is_gold_healthy"] = df[label_col].eq(LABEL_HEALTHY)

    selected = df[df["strict_clean_seed_candidate"]].copy()
    false_clean = selected[~selected["is_gold_healthy"]].copy()
    missed_healthy = df[
        df["is_gold_healthy"] & ~df["strict_clean_seed_candidate"]
    ].copy()

    selected_n = len(selected)
    healthy_total = int(df["is_gold_healthy"].sum())
    selected_healthy = int(selected["is_gold_healthy"].sum())
    false_clean_n = len(false_clean)

    precision = selected_healthy / selected_n if selected_n else 0.0
    recall = selected_healthy / healthy_total if healthy_total else 0.0

    reason_counter = Counter()
    for reasons in df["strict_defect_reasons"]:
        if not reasons:
            reason_counter["SELECTED_STRICT_CLEAN_SEED"] += 1
        else:
            for reason in reasons.split("|"):
                reason_counter[reason] += 1

    preferred_cols = []
    for c in ID_COL_CANDIDATES:
        if c in df.columns and c not in preferred_cols:
            preferred_cols.append(c)

    for c in [
        label_col,
        text_col,
        usable_score_col,
        pred_route_col,
        "strict_clean_seed_candidate",
        "strict_defect_reasons",
    ]:
        if c and c in df.columns and c not in preferred_cols:
            preferred_cols.append(c)

    remaining_cols = [c for c in df.columns if c not in preferred_cols]
    export_cols = preferred_cols + remaining_cols

    selected[export_cols].to_csv(
        OUT_DIR / "calibration_strict_clean_seed_candidates_v1.csv",
        index=False,
    )

    false_clean[export_cols].to_csv(
        OUT_DIR / "calibration_strict_clean_seed_false_clean_v1.csv",
        index=False,
    )

    missed_healthy[export_cols].to_csv(
        OUT_DIR / "calibration_strict_clean_seed_missed_healthy_v1.csv",
        index=False,
    )

    review_sample = selected.sample(
        n=min(args.sample_size, len(selected)),
        random_state=42,
    ).copy()

    review_sample.insert(0, "manual_review_label", "")
    review_sample.insert(1, "manual_review_notes", "")

    review_sample[["manual_review_label", "manual_review_notes"] + export_cols].to_csv(
        OUT_DIR / "manual_review_sample_strict_clean_seed_v1.csv",
        index=False,
    )

    summary_lines = []
    summary_lines.append("# Strict Clean Seed Extractor v1")
    summary_lines.append("")
    summary_lines.append("## Inputs")
    summary_lines.append("")
    summary_lines.append(f"- calibration_file: `{args.calibration}`")
    summary_lines.append(f"- calibration_rows: `{len(df)}`")
    summary_lines.append(f"- text_col: `{text_col}`")
    summary_lines.append(f"- label_col: `{label_col}`")
    summary_lines.append(f"- router_path: `{router_path}`")
    summary_lines.append(f"- router_merge_status: `{router_merge_status}`")
    summary_lines.append(f"- usable_score_col: `{usable_score_col}`")
    summary_lines.append(f"- pred_route_col: `{pred_route_col}`")
    summary_lines.append(f"- usable_threshold: `{args.usable_threshold}`")
    summary_lines.append("")
    summary_lines.append("## Calibration result")
    summary_lines.append("")
    summary_lines.append(f"- selected_candidates: `{selected_n}`")
    summary_lines.append(f"- selected_gold_healthy: `{selected_healthy}`")
    summary_lines.append(f"- false_clean_rows: `{false_clean_n}`")
    summary_lines.append(f"- total_gold_healthy: `{healthy_total}`")
    summary_lines.append(f"- clean_seed_precision: `{pct(selected_healthy, selected_n)}`")
    summary_lines.append(f"- clean_seed_recall: `{pct(selected_healthy, healthy_total)}`")
    summary_lines.append("")
    summary_lines.append("## Decision rule")
    summary_lines.append("")
    if selected_n == 0:
        summary_lines.append(
            "No strict clean seed selected. Rules are too strict or router score was missing."
        )
    elif precision >= 0.95:
        summary_lines.append(
            "Calibration precision is at least 95%. Next step: manually review the seed sample before touching validation."
        )
    else:
        summary_lines.append(
            "Calibration precision is below 95%. Do not use this as clean target source yet."
        )
    summary_lines.append("")
    summary_lines.append("## Defect reason counts")
    summary_lines.append("")
    for reason, count in reason_counter.most_common():
        summary_lines.append(f"- {reason}: `{count}`")

    summary_lines.append("")
    summary_lines.append("## Output files")
    summary_lines.append("")
    for name in [
        "calibration_strict_clean_seed_candidates_v1.csv",
        "calibration_strict_clean_seed_false_clean_v1.csv",
        "calibration_strict_clean_seed_missed_healthy_v1.csv",
        "manual_review_sample_strict_clean_seed_v1.csv",
    ]:
        summary_lines.append(f"- `{OUT_DIR / name}`")

    summary_text = "\n".join(summary_lines)

    (OUT_DIR / "calibration_strict_clean_seed_summary_v1.md").write_text(
        summary_text,
        encoding="utf-8",
    )

    print(summary_text)


if __name__ == "__main__":
    main()
