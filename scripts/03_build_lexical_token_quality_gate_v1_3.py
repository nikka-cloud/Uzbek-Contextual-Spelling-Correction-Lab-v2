from __future__ import annotations

import importlib.util
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1"
)

CALIBRATION = (
    ROOT
    / "text_health_features_v1"
    / "calibration_text_health_features_labeled_v1.csv"
)

CAL_ROUTER_OOF = (
    ROOT
    / "hierarchical_text_router_v1_1"
    / "calibration_hierarchical_oof_predictions_v1_1.csv"
)

VALIDATION_LABELED = (
    ROOT
    / "strict_clean_seed_extractor_v1_2"
    / "validation_eval_v1_2"
    / "validation_predictions_labeled_eval_v1_2.csv"
)

V1_2_SCRIPT = Path("scripts/03_build_strict_clean_seed_extractor_v1_2.py")

OUT_DIR = ROOT / "strict_clean_seed_extractor_v1_3_lexical_gate"


TEXT_COLS = [
    "sentence_text",
    "text",
    "raw_text",
    "raw_sentence",
    "sentence",
    "context_preview",
]

LABEL_COLS = [
    "human_health_label",
    "gold_health_label",
]


WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+")


COMMON_SAFE_TOKENS = {
    # high-frequency Uzbek function words / particles / auxiliaries
    "va", "ham", "bu", "shu", "u", "ular", "bilan", "uchun", "deb", "edi", "ekan",
    "emas", "bo'ldi", "bo'lgan", "bo'lishi", "bo'ladi", "bo'lsa", "qilib", "qiladi",
    "qilish", "qilgan", "qadar", "keyin", "oldin", "juda", "eng", "ko'p", "kam",
    "bir", "ikki", "uch", "to'rt", "besh", "olti", "yetti", "sakkiz", "to'qqiz",
    "o'n", "yuz", "ming", "yil", "oy", "kun", "holda", "tarzda", "orqali",
    "agar", "lekin", "chunki", "balki", "masalan", "ya'ni", "yoki",

    # common clean content words that should not be treated as suspicious by shape alone
    "inson", "hayot", "kitob", "maktab", "shahar", "davlat", "xalq", "suv", "yer",
    "ish", "faoliyat", "tizim", "axborot", "marketing", "bozor", "hujayra",
    "hujayralar", "norma", "oila", "oilasi", "tushuncha", "asari",
}


COMMON_CANONICAL_WORDS = {
    "yoki",
    "shaxs",
    "an'ana",
    "an'analar",
    "an'anaviy",
    "dadaning",
    "tibbiyot",
}


MONTH_WORDS = {
    "yanvar", "fevral", "mart", "aprel", "may", "iyun",
    "iyul", "avgust", "sentabr", "oktabr", "noyabr", "dekabr",
}


NUMBER_WORDS = {
    "bir", "ikki", "uch", "to'rt", "besh", "olti", "yetti", "sakkiz", "to'qqiz",
    "o'n", "yigirma", "o'ttiz", "qirq", "ellik", "oltmish", "yetmish", "sakson",
    "to'qson", "yuz", "ming", "million",
}


ORDINAL_WORDS = {
    "birinchi", "ikkinchi", "uchinchi", "to'rtinchi", "beshinchi", "oltinchi",
    "yettinchi", "sakkizinchi", "to'qqizinchi", "o'ninchi",
}


UZBEK_SUFFIXES = [
    "larining", "larimizning", "laringizning",
    "laridan", "lariga", "larni", "lari", "larda",
    "imizning", "ingizning", "ining",
    "imiz", "ingiz", "ning", "dan", "dagi", "dagi",
    "ga", "da", "ni", "lar", "lik", "siz", "chi", "cha",
    "roq", "dek", "day", "dir", "man", "san", "miz",
]


# General token-shape corruption patterns.
SUSPICIOUS_TOKEN_SHAPE_RE = re.compile(
    r"("
    r"[bcdfghjklmnpqrstvwxyz]{5,}|"
    r"[aeiouo']{4,}|"
    r"([a-z])\2\2|"
    r"iing$|"
    r"miing$|"
    r"lair$"
    r")",
    re.I,
)


MISSING_APOSTROPHE_FAMILY_RE = re.compile(
    r"\b(anana|ananalar|ananaviy|mablag|togliq|togri|goyat|olim|otgan|oladi)\b",
    re.I,
)


SUSPICIOUS_PHRASE_RE = re.compile(
    r"\b("
    r"yil\s+may|"
    r"bir\s+maygacha|"
    r"har\s+qanday\s+faoliyati|"
    r"beshinchi\s+hujayralar|"
    r"yangi\s+dorivorlar|"
    r"bozor\s+sohasini\s+har\s+qanday\s+faoliyati"
    r")\b",
    re.I,
)


def norm_ap(text: str) -> str:
    return (
        str(text)
        .replace("ʻ", "'")
        .replace("ʼ", "'")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("`", "'")
        .replace("´", "'")
    )


def pick_col(df: pd.DataFrame, cols: list[str]) -> str:
    for c in cols:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any of: {cols}")


def load_v1_2_module():
    spec = importlib.util.spec_from_file_location("strict_seed_v1_2", V1_2_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {V1_2_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tokens(text: str) -> list[str]:
    return [
        norm_ap(t).strip(".,;:!?()[]{}\"'").lower()
        for t in WORD_RE.findall(norm_ap(text))
        if norm_ap(t).strip(".,;:!?()[]{}\"'")
    ]


def edit_distance_leq1(a: str, b: str) -> bool:
    if a == b:
        return True

    if abs(len(a) - len(b)) > 1:
        return False

    if len(a) == len(b):
        mismatches = sum(x != y for x, y in zip(a, b))
        return mismatches <= 1

    if len(a) > len(b):
        a, b = b, a

    # len(b) == len(a) + 1
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            edits += 1
            j += 1
            if edits > 1:
                return False

    return True


def strip_known_suffix(tok: str, vocab: set[str]) -> str | None:
    if tok in vocab:
        return tok

    for suffix in sorted(UZBEK_SUFFIXES, key=len, reverse=True):
        if tok.endswith(suffix) and len(tok) > len(suffix) + 3:
            root = tok[: -len(suffix)]
            if root in vocab:
                return root

    return None


def build_lexicon(cal: pd.DataFrame, text_col: str, label_col: str) -> dict:
    healthy = cal[cal[label_col].eq("HEALTHY")].copy()

    token_counter = Counter()
    char_trigrams = Counter()

    for text in healthy[text_col].astype(str):
        for tok in tokens(text):
            token_counter[tok] += 1
            padded = f"^{tok}$"
            for i in range(max(0, len(padded) - 2)):
                char_trigrams[padded[i:i+3]] += 1

    healthy_vocab_any = set(token_counter)
    healthy_vocab_confident = {
        tok for tok, count in token_counter.items()
        if count >= 2 or tok in COMMON_SAFE_TOKENS or len(tok) <= 3
    }

    healthy_vocab_confident |= COMMON_SAFE_TOKENS
    healthy_vocab_confident |= COMMON_CANONICAL_WORDS

    return {
        "token_counter": token_counter,
        "healthy_vocab_any": healthy_vocab_any,
        "healthy_vocab_confident": healthy_vocab_confident,
        "healthy_char_trigrams": set(char_trigrams),
    }


def trigram_coverage(tok: str, trigram_set: set[str]) -> float:
    padded = f"^{tok}$"
    grams = [padded[i:i+3] for i in range(max(0, len(padded) - 2))]

    if not grams:
        return 1.0

    return sum(g in trigram_set for g in grams) / len(grams)


def lexical_reasons(text: str, lexicon: dict, profile: dict) -> list[str]:
    text_norm = norm_ap(text)
    toks = tokens(text_norm)

    vocab_any = lexicon["healthy_vocab_any"]
    vocab_conf = lexicon["healthy_vocab_confident"]
    trigram_set = lexicon["healthy_char_trigrams"]

    reasons = []

    if SUSPICIOUS_PHRASE_RE.search(text_norm):
        reasons.append("lexical_suspicious_phrase")

    for i, tok in enumerate(toks):
        if not tok:
            continue

        if tok in COMMON_SAFE_TOKENS:
            continue

        # Strictly avoid month/date expressions in gold clean seed for now.
        if profile["block_month_words"] and tok in MONTH_WORDS:
            reasons.append(f"lexical_month_word:{tok}")

        # Ordinal injected-word risk. This is conservative.
        if profile["block_ordinal_words"] and tok in ORDINAL_WORDS:
            reasons.append(f"lexical_ordinal_word:{tok}")

        if MISSING_APOSTROPHE_FAMILY_RE.search(tok):
            reasons.append(f"lexical_possible_missing_apostrophe:{tok}")

        if SUSPICIOUS_TOKEN_SHAPE_RE.search(tok):
            reasons.append(f"lexical_bad_token_shape:{tok}")

        # Suspicious near-miss to a known common canonical word.
        for canonical in COMMON_CANONICAL_WORDS:
            if tok != canonical and edit_distance_leq1(tok, canonical):
                reasons.append(f"lexical_near_miss:{tok}->{canonical}")
                break

        # Unknown-token gate.
        if profile["unknown_policy"] != "off":
            if len(tok) >= profile["unknown_min_len"]:
                known = (
                    tok in vocab_conf
                    or strip_known_suffix(tok, vocab_conf) is not None
                )

                if not known:
                    coverage = trigram_coverage(tok, trigram_set)

                    if profile["unknown_policy"] == "hard":
                        reasons.append(f"lexical_unknown_token:{tok}")

                    elif profile["unknown_policy"] == "shape_or_low_trigram":
                        if coverage < profile["min_trigram_coverage"]:
                            reasons.append(
                                f"lexical_unknown_low_trigram:{tok}:{coverage:.2f}"
                            )

                    elif profile["unknown_policy"] == "very_hard":
                        if tok not in vocab_any and strip_known_suffix(tok, vocab_any) is None:
                            reasons.append(f"lexical_unknown_token_any_vocab:{tok}")

    return sorted(set(reasons))


PROFILES = [
    {
        "name": "v1_3_shape_phrase_only",
        "unknown_policy": "off",
        "unknown_min_len": 6,
        "min_trigram_coverage": 0.40,
        "block_month_words": True,
        "block_ordinal_words": True,
    },
    {
        "name": "v1_3_unknown_low_trigram",
        "unknown_policy": "shape_or_low_trigram",
        "unknown_min_len": 6,
        "min_trigram_coverage": 0.45,
        "block_month_words": True,
        "block_ordinal_words": True,
    },
    {
        "name": "v1_3_unknown_hard_len7",
        "unknown_policy": "hard",
        "unknown_min_len": 7,
        "min_trigram_coverage": 0.45,
        "block_month_words": True,
        "block_ordinal_words": True,
    },
    {
        "name": "v1_3_unknown_hard_len6",
        "unknown_policy": "hard",
        "unknown_min_len": 6,
        "min_trigram_coverage": 0.45,
        "block_month_words": True,
        "block_ordinal_words": True,
    },
]


def pct(a: int, b: int) -> str:
    if b == 0:
        return "NA"
    return f"{100 * a / b:.2f}%"


def prepare_datasets() -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    cal = pd.read_csv(CALIBRATION, dtype=str).fillna("")
    cal_oof = pd.read_csv(CAL_ROUTER_OOF, dtype=str).fillna("")
    val = pd.read_csv(VALIDATION_LABELED, dtype=str).fillna("")

    if len(cal) != len(cal_oof):
        raise RuntimeError(f"Calibration/router row mismatch: {len(cal)} vs {len(cal_oof)}")

    text_col = pick_col(cal, TEXT_COLS)
    label_col = pick_col(cal, LABEL_COLS)

    if text_col not in val.columns:
        raise RuntimeError(f"Validation missing text col: {text_col}")
    if label_col not in val.columns:
        raise RuntimeError(f"Validation missing label col: {label_col}")

    cal = cal.copy()
    cal["p_stage1_usable"] = cal_oof["p_stage1_usable"].astype(float)
    cal["dev_split"] = "calibration"

    val = val.copy()
    val["p_stage1_usable"] = val["p_stage1_usable"].astype(float)
    val["dev_split"] = "opened_validation"

    return cal, val, text_col, label_col


def evaluate_split(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    selected_col: str,
) -> dict:
    selected = df[df[selected_col]].copy()
    false_clean = selected[~selected[label_col].eq("HEALTHY")].copy()

    selected_n = len(selected)
    healthy_selected = int(selected[label_col].eq("HEALTHY").sum())
    false_n = len(false_clean)
    total_healthy = int(df[label_col].eq("HEALTHY").sum())

    return {
        "rows": len(df),
        "selected": selected_n,
        "healthy_selected": healthy_selected,
        "false_clean": false_n,
        "total_healthy": total_healthy,
        "precision": healthy_selected / selected_n if selected_n else 0.0,
        "recall": healthy_selected / total_healthy if total_healthy else 0.0,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v1_2 = load_v1_2_module()
    base_profile = next(
        p for p in v1_2.PROFILES
        if p["name"] == "v1_2_ultra_plus_token_blockers"
    )

    cal, val, text_col, label_col = prepare_datasets()
    lexicon = build_lexicon(cal, text_col, label_col)

    combined = pd.concat([cal, val], ignore_index=True)

    all_rows = []

    for profile in PROFILES:
        name = profile["name"]

        reason_col = f"{name}_reasons"
        selected_col = f"{name}_selected"

        def row_reasons(row) -> str:
            base_reasons = v1_2.strict_reasons(
                text=row[text_col],
                p_usable=float(row["p_stage1_usable"]),
                profile=base_profile,
            )

            lex_reasons = lexical_reasons(
                text=row[text_col],
                lexicon=lexicon,
                profile=profile,
            )

            return "|".join(sorted(set(base_reasons + lex_reasons)))

        combined[reason_col] = combined.apply(row_reasons, axis=1)
        combined[selected_col] = combined[reason_col].eq("")

        profile_dir = OUT_DIR / name
        profile_dir.mkdir(parents=True, exist_ok=True)

        combined.to_csv(profile_dir / "combined_dev_predictions.csv", index=False)

        selected = combined[combined[selected_col]].copy()
        false_clean = selected[~selected[label_col].eq("HEALTHY")].copy()

        selected.to_csv(profile_dir / "selected_candidates.csv", index=False)
        false_clean.to_csv(profile_dir / "false_clean.csv", index=False)

        reason_counter = Counter()
        for reasons in combined[reason_col]:
            if not reasons:
                reason_counter["SELECTED"] += 1
            else:
                for r in reasons.split("|"):
                    reason_counter[r] += 1

        split_results = {}

        for split in ["calibration", "opened_validation"]:
            split_df = combined[combined["dev_split"].eq(split)].copy()
            split_results[split] = evaluate_split(
                df=split_df,
                text_col=text_col,
                label_col=label_col,
                selected_col=selected_col,
            )

        combined_result = evaluate_split(
            df=combined,
            text_col=text_col,
            label_col=label_col,
            selected_col=selected_col,
        )

        with (profile_dir / "summary.md").open("w", encoding="utf-8") as f:
            f.write(f"# {name}\n\n")
            f.write("## Metrics\n\n")
            f.write("| split | rows | selected | healthy_selected | false_clean | precision | recall |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")

            for split, res in split_results.items():
                f.write(
                    f"| {split} | {res['rows']} | {res['selected']} | "
                    f"{res['healthy_selected']} | {res['false_clean']} | "
                    f"{pct(res['healthy_selected'], res['selected'])} | "
                    f"{pct(res['healthy_selected'], res['total_healthy'])} |\n"
                )

            f.write(
                f"| combined_dev | {combined_result['rows']} | {combined_result['selected']} | "
                f"{combined_result['healthy_selected']} | {combined_result['false_clean']} | "
                f"{pct(combined_result['healthy_selected'], combined_result['selected'])} | "
                f"{pct(combined_result['healthy_selected'], combined_result['total_healthy'])} |\n"
            )

            f.write("\n## Profile\n\n")
            for k, v in profile.items():
                f.write(f"- {k}: `{v}`\n")

            f.write("\n## Lexicon stats\n\n")
            f.write(f"- healthy_vocab_any: `{len(lexicon['healthy_vocab_any'])}`\n")
            f.write(f"- healthy_vocab_confident: `{len(lexicon['healthy_vocab_confident'])}`\n")
            f.write(f"- healthy_char_trigrams: `{len(lexicon['healthy_char_trigrams'])}`\n")

            f.write("\n## Reason counts\n\n")
            for reason, count in reason_counter.most_common(80):
                f.write(f"- {reason}: `{count}`\n")

            f.write("\n## False-clean examples\n\n")
            if len(false_clean) == 0:
                f.write("None.\n")
            else:
                for _, row in false_clean.head(50).iterrows():
                    f.write("\n---\n\n")
                    f.write(f"split: `{row['dev_split']}`\n\n")
                    f.write(f"label: `{row[label_col]}`\n\n")
                    f.write(f"p_stage1_usable: `{row['p_stage1_usable']}`\n\n")
                    f.write(f"reasons: `{row[reason_col]}`\n\n")
                    f.write(f"text: {row[text_col]}\n")

        for split, res in split_results.items():
            all_rows.append(
                {
                    "profile": name,
                    "split": split,
                    **res,
                }
            )

        all_rows.append(
            {
                "profile": name,
                "split": "combined_dev",
                **combined_result,
            }
        )

    comparison = pd.DataFrame(all_rows)
    comparison.to_csv(OUT_DIR / "profile_comparison_v1_3.csv", index=False)

    lines = []
    lines.append("# Lexical Token Quality Gate v1.3 Profile Comparison")
    lines.append("")
    lines.append("Important: `opened_validation` is no longer blind. Treat this as development evidence only.")
    lines.append("")
    lines.append("| profile | split | selected | healthy | false_clean | precision | recall |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for row in all_rows:
        lines.append(
            f"| {row['profile']} | {row['split']} | {row['selected']} | "
            f"{row['healthy_selected']} | {row['false_clean']} | "
            f"{pct(row['healthy_selected'], row['selected'])} | "
            f"{pct(row['healthy_selected'], row['total_healthy'])} |"
        )

    lines.append("")
    lines.append("## Decision guide")
    lines.append("")
    lines.append("- If a profile reaches high precision on both calibration and opened validation, it becomes a v1.3 candidate.")
    lines.append("- It still cannot be called validated because validation was already opened.")
    lines.append("- Final proof requires a new blind review sample.")

    summary = "\n".join(lines)
    (OUT_DIR / "profile_comparison_v1_3.md").write_text(summary, encoding="utf-8")

    print(summary)


if __name__ == "__main__":
    main()
