from __future__ import annotations

import math
import re
from collections import Counter
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

ROUTER_OOF = (
    ROOT
    / "hierarchical_text_router_v1_1"
    / "calibration_hierarchical_oof_predictions_v1_1.csv"
)

OUT_DIR = ROOT / "strict_clean_seed_extractor_v1_2"


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

ID_COLS = [
    "calibration_row_id",
    "review_row_id",
    "row_id",
    "sample_id",
    "global_sentence_id",
    "sentence_id",
    "sentence_hash",
    "source_record_id",
]


WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
MID_CAP_RE = re.compile(r"[a-zà-öø-ÿ][A-ZÀ-ÖØ-Þ]")
LIST_START_RE = re.compile(r"^\s*(([-*•]+)|([0-9]+[.)])|([A-Za-z][.)])|([ivxlcdm]+[.)]))\s+", re.I)
URL_RE = re.compile(r"(https?://|www\.|@|\.com|\.uz|\.ru)", re.I)


EXAM_OR_FORMAT_RE = re.compile(
    r"\b("
    r"test|savol|javob|belgilang|variant|jadval|rasm|formula|mavzu|reja|"
    r"bayt|inv|stenogramma|diagramma|statistik|xromosom|orqa miya|qayerda hosil"
    r")\b",
    re.I,
)


TECH_DOMAIN_RE = re.compile(
    r"\b("
    r"litsenziya|shartnoma|mualliflik|huquq|huquqlari|modda|paragraf|"
    r"klinik|sinusit|rinit|konyuktivit|parranda|galaktoza|fruktoza|"
    r"kompyuter|kompyug|dastur|numizmatik|xronologik|sillogizm|deduktiv|"
    r"multisektoral|akademik litsey|emlash|inshoot|loyihalashtirish"
    r")\b",
    re.I,
)


FOREIGN_OR_ODD_RE = re.compile(
    r"\b("
    r"popup|popups|popunder|popunders|browser|cache|account|dashboard|login|support|"
    r"website|online|email|windows|linux|android|iphone|shalke|gotssi|qomenskiy"
    r")\b",
    re.I,
)


# Built from calibration false-clean examples.
# Purpose: block known local corruption from becoming clean target text.
KNOWN_LOCAL_CORRUPTION_RE = re.compile(
    r"\b("
    r"faromush|qipayotirsiz|otaonaning|qalasiz|sidxarta|kpskarib|"
    r"esal|xamjixatlikka|dalolatlair|ranga|"
    r"yig'imterimida|ko'tarilaveradiqaynar|ananaviy|mablag|kalloriyalilik|"
    r"hayhuy|ko'psonli|depti|tumtayib|apirshapir|ahyonahyonda|bittayarimta|"
    r"qechki|tomonidai|shirinmibilolmay|holatikasal|hayota|"
    r"quydida|birikki|sxolastiq|toychi|bonuga|turlituman|"
    r"tevarakatrofimizga|inshootlarni|yebichishni|mahsus|xuquqiy|faolyat|"
    r"javopdan|utkazinglar|dedida|nuni|io'pusini|arbun"
    r")\b",
    re.I,
)


# Broken word-spacing patterns that sentence-shape filters missed.
BROKEN_SPACING_RE = re.compile(
    r"\b("
    r"muno\s+sabat\w*|"
    r"to\s+qima\w*|"
    r"sho\s+roi|"
    r"noyo\s+lik\w*|"
    r"hayhuy\s+lab|"
    r"ota\s+ona\w*"
    r")\b",
    re.I,
)


SUSPICIOUS_PHRASE_RE = re.compile(
    r"\b("
    r"tilla\s+ranga\s+kiradi|"
    r"rang[a]\s+kiradi|"
    r"muskul\s+tolalar.*kpskarib|"
    r"o'zlikni\s+anglash\s+piyozi|"
    r"xamjixatlikka\s+dalolatlair"
    r")\b",
    re.I,
)


SUSPICIOUS_ENDING_RE = re.compile(
    r"(tirida|veradiqaynar|bilolmay|atrofimizga|yebichishni|g'erlardan|lair)$",
    re.I,
)


SPACED_SUFFIX_RE = re.compile(
    r"\b([A-Za-zÀ-ÖØ-öø-ÿ']{3,})\s+"
    r"(ning|ni|ga|da|dan|gi|lik|ligi|lar|lari|siz|miz|chi|cha|qimalarga|roi)\b",
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


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise ValueError(f"Could not find any of these columns: {candidates}")


def safe_float(x) -> float:
    try:
        if x is None:
            return math.nan
        if isinstance(x, str) and not x.strip():
            return math.nan
        return float(x)
    except Exception:
        return math.nan


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
    punct = sum(not c.isalnum() for c in chars)

    return {
        "char_count": len(chars),
        "alpha_ratio": alpha / len(chars),
        "digit_ratio": digit / len(chars),
        "punct_ratio": punct / len(chars),
    }


def repeated_ngram_flags(tokens: list[str]) -> list[str]:
    clean = [
        norm_ap(t).strip(".,;:!?()[]{}\"'").lower()
        for t in tokens
        if len(norm_ap(t).strip(".,;:!?()[]{}\"'")) >= 2
    ]

    flags = []

    for n in [2, 3]:
        seen = Counter()
        for i in range(0, len(clean) - n + 1):
            gram = tuple(clean[i:i + n])
            seen[gram] += 1

        for gram, count in seen.items():
            if count >= 2:
                flags.append(f"repeated_{n}gram")
                break

    return flags


def token_shape_flags(tokens: list[str]) -> list[str]:
    flags = []

    for tok in tokens:
        t = norm_ap(tok).strip(".,;:!?()[]{}\"'").lower()

        if not t:
            continue

        if len(t) >= 14:
            flags.append("very_long_token_ge_14")

        if len(t) >= 12 and "'" in t:
            flags.append("very_long_apostrophe_token_ge_12")

        if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", t):
            flags.append("long_consonant_cluster")

        if re.search(r"([a-z])\1\1", t):
            flags.append("triple_repeated_char")

        if "w" in t:
            flags.append("contains_w")

        if "c" in t and "ch" not in t:
            flags.append("suspicious_c_without_ch")

        if SUSPICIOUS_ENDING_RE.search(t):
            flags.append("suspicious_token_ending")

    return sorted(set(flags))


PROFILES = [
    {
        "name": "v1_2_ultra_plus_token_blockers",
        "min_p_usable": 0.85,
        "min_words": 6,
        "max_words": 16,
        "min_chars": 30,
        "max_chars": 120,
        "min_alpha_ratio": 0.78,
        "max_digit_ratio": 0.00,
        "max_punct_ratio": 0.08,
        "max_commas": 0,
        "require_terminal_period": True,
        "block_technical_domain": True,
        "block_capital_after_comma": True,
        "block_inserted_clause_markers": True,
        "block_repeated_ngrams": True,
    },
    {
        "name": "v1_2_ultra_plus_token_blockers_p90",
        "min_p_usable": 0.90,
        "min_words": 6,
        "max_words": 16,
        "min_chars": 30,
        "max_chars": 120,
        "min_alpha_ratio": 0.78,
        "max_digit_ratio": 0.00,
        "max_punct_ratio": 0.08,
        "max_commas": 0,
        "require_terminal_period": True,
        "block_technical_domain": True,
        "block_capital_after_comma": True,
        "block_inserted_clause_markers": True,
        "block_repeated_ngrams": True,
    },
    {
        "name": "v1_2_ultra_plus_token_blockers_p95",
        "min_p_usable": 0.95,
        "min_words": 6,
        "max_words": 16,
        "min_chars": 30,
        "max_chars": 120,
        "min_alpha_ratio": 0.78,
        "max_digit_ratio": 0.00,
        "max_punct_ratio": 0.08,
        "max_commas": 0,
        "require_terminal_period": True,
        "block_technical_domain": True,
        "block_capital_after_comma": True,
        "block_inserted_clause_markers": True,
        "block_repeated_ngrams": True,
    },
]


def strict_reasons(text: str, p_usable: float, profile: dict) -> list[str]:
    text = norm_ap(text).strip()
    lowered = text.lower()
    tokens = WORD_RE.findall(text)
    stats = text_stats(text)
    word_count = len(tokens)

    reasons = []

    if math.isnan(p_usable):
        reasons.append("missing_p_stage1_usable")
    elif p_usable < profile["min_p_usable"]:
        reasons.append(f"p_stage1_usable_below_{profile['min_p_usable']}")

    if word_count < profile["min_words"]:
        reasons.append("too_few_words")

    if word_count > profile["max_words"]:
        reasons.append("too_many_words")

    if stats["char_count"] < profile["min_chars"]:
        reasons.append("too_short_chars")

    if stats["char_count"] > profile["max_chars"]:
        reasons.append("too_long_chars")

    if stats["alpha_ratio"] < profile["min_alpha_ratio"]:
        reasons.append("low_alpha_ratio")

    if stats["digit_ratio"] > profile["max_digit_ratio"]:
        reasons.append("digit_heavy")

    if stats["punct_ratio"] > profile["max_punct_ratio"]:
        reasons.append("punctuation_heavy")

    if profile["require_terminal_period"] and not text.endswith("."):
        reasons.append("not_plain_period_sentence")

    if "?" in text:
        reasons.append("question_sentence_excluded")

    if "!" in text:
        reasons.append("exclamation_or_dialogue_excluded")

    if ":" in text:
        reasons.append("colon_excluded")

    if ";" in text:
        reasons.append("semicolon_excluded")

    if text.count(",") > profile["max_commas"]:
        reasons.append("too_many_commas")

    if LIST_START_RE.search(text):
        reasons.append("list_start")

    if URL_RE.search(text):
        reasons.append("url_or_web_marker")

    if CYRILLIC_RE.search(text):
        reasons.append("contains_cyrillic")

    if ARABIC_RE.search(text):
        reasons.append("contains_arabic")

    if MID_CAP_RE.search(text):
        reasons.append("mid_word_capital_or_merge")

    if SPACED_SUFFIX_RE.search(text):
        reasons.append("spaced_suffix_or_broken_word")

    if EXAM_OR_FORMAT_RE.search(text):
        reasons.append("exam_or_format_material")

    if profile["block_technical_domain"] and TECH_DOMAIN_RE.search(text):
        reasons.append("technical_or_domain_specific_material")

    if FOREIGN_OR_ODD_RE.search(text):
        reasons.append("foreign_or_odd_named_token")

    if KNOWN_LOCAL_CORRUPTION_RE.search(text):
        reasons.append("known_local_corruption_token")

    if BROKEN_SPACING_RE.search(text):
        reasons.append("broken_spacing_phrase")

    if SUSPICIOUS_PHRASE_RE.search(text):
        reasons.append("suspicious_corruption_phrase")

    if re.match(r"^[a-z]\s+[A-ZÀ-ÖØ-Þ]", text):
        reasons.append("stray_lowercase_initial_token")

    if re.match(r"^[A-ZÀ-ÖØ-Þ]{1,2}\s+[a-zà-öø-ÿ]", text):
        reasons.append("stray_uppercase_initial_token")

    if re.search(r",\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]", text):
        if profile["block_capital_after_comma"]:
            reasons.append("capital_after_comma_possible_boundary_damage")

    if re.search(r"\b(masalan|ya'ni|balki|chunki|agar)\b", lowered):
        if profile["block_inserted_clause_markers"]:
            reasons.append("inserted_clause_marker_excluded")

    if profile["block_repeated_ngrams"]:
        reasons.extend(repeated_ngram_flags(tokens))

    reasons.extend(token_shape_flags(tokens))

    return sorted(set(reasons))


def pct(a: int, b: int) -> str:
    if b == 0:
        return "NA"
    return f"{100 * a / b:.2f}%"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CALIBRATION, dtype=str).fillna("")
    router = pd.read_csv(ROUTER_OOF, dtype=str).fillna("")

    if len(df) != 699:
        raise RuntimeError(f"Expected 699 calibration rows, got {len(df)}")

    if len(router) != len(df):
        raise RuntimeError(f"Router OOF rows do not match calibration rows: {len(router)} vs {len(df)}")

    text_col = pick_col(df, TEXT_COLS)
    label_col = pick_col(df, LABEL_COLS)

    if "p_stage1_usable" not in router.columns:
        raise RuntimeError("Missing p_stage1_usable in router OOF file")

    df = df.copy()
    df["p_stage1_usable"] = router["p_stage1_usable"].values

    if "predicted_route" in router.columns:
        df["predicted_route"] = router["predicted_route"].values

    df["is_healthy"] = df[label_col].eq("HEALTHY")

    all_summaries = []

    for profile in PROFILES:
        name = profile["name"]
        reason_col = f"{name}_reasons"
        selected_col = f"{name}_selected"

        df[reason_col] = df.apply(
            lambda row: "|".join(
                strict_reasons(
                    text=row[text_col],
                    p_usable=safe_float(row["p_stage1_usable"]),
                    profile=profile,
                )
            ),
            axis=1,
        )

        df[selected_col] = df[reason_col].eq("")

        selected = df[df[selected_col]].copy()
        false_clean = selected[~selected["is_healthy"]].copy()
        missed_healthy = df[df["is_healthy"] & ~df[selected_col]].copy()

        selected_n = len(selected)
        selected_healthy = int(selected["is_healthy"].sum())
        false_n = len(false_clean)
        total_healthy = int(df["is_healthy"].sum())

        reason_counter = Counter()

        for reasons in df[reason_col]:
            if not reasons:
                reason_counter["SELECTED"] += 1
            else:
                for r in reasons.split("|"):
                    reason_counter[r] += 1

        profile_dir = OUT_DIR / name
        profile_dir.mkdir(parents=True, exist_ok=True)

        export_cols = []

        for c in ID_COLS:
            if c in df.columns and c not in export_cols:
                export_cols.append(c)

        for c in [
            label_col,
            text_col,
            "p_stage1_usable",
            "predicted_route",
            selected_col,
            reason_col,
        ]:
            if c in df.columns and c not in export_cols:
                export_cols.append(c)

        for c in df.columns:
            if c not in export_cols:
                export_cols.append(c)

        selected[export_cols].to_csv(profile_dir / "selected_candidates.csv", index=False)
        false_clean[export_cols].to_csv(profile_dir / "false_clean.csv", index=False)
        missed_healthy[export_cols].to_csv(profile_dir / "missed_healthy.csv", index=False)

        review = selected.sample(n=min(100, len(selected)), random_state=42).copy()
        review.insert(0, "manual_review_label", "")
        review.insert(1, "manual_review_notes", "")
        review[["manual_review_label", "manual_review_notes"] + export_cols].to_csv(
            profile_dir / "manual_review_sample.csv",
            index=False,
        )

        with (profile_dir / "summary.md").open("w", encoding="utf-8") as f:
            f.write(f"# {name}\n\n")
            f.write("## Metrics\n\n")
            f.write(f"- selected_candidates: `{selected_n}`\n")
            f.write(f"- selected_gold_healthy: `{selected_healthy}`\n")
            f.write(f"- false_clean_rows: `{false_n}`\n")
            f.write(f"- total_gold_healthy: `{total_healthy}`\n")
            f.write(f"- precision: `{pct(selected_healthy, selected_n)}`\n")
            f.write(f"- recall: `{pct(selected_healthy, total_healthy)}`\n\n")

            f.write("## Profile\n\n")
            for k, v in profile.items():
                f.write(f"- {k}: `{v}`\n")

            f.write("\n## Reason counts\n\n")
            for reason, count in reason_counter.most_common():
                f.write(f"- {reason}: `{count}`\n")

            f.write("\n## False-clean examples\n\n")
            for _, row in false_clean.head(50).iterrows():
                f.write("\n---\n\n")
                f.write(f"label: {row[label_col]}\n\n")
                f.write(f"p_stage1_usable: {row['p_stage1_usable']}\n\n")
                f.write(f"text: {row[text_col]}\n\n")

        all_summaries.append(
            {
                "profile": name,
                "selected_candidates": selected_n,
                "selected_gold_healthy": selected_healthy,
                "false_clean_rows": false_n,
                "total_gold_healthy": total_healthy,
            }
        )

    lines = []
    lines.append("# Strict Clean Seed Extractor v1.2 Profile Comparison")
    lines.append("")
    lines.append("| profile | selected | healthy | false_clean | precision | recall |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for row in all_summaries:
        selected_n = int(row["selected_candidates"])
        healthy_n = int(row["selected_gold_healthy"])
        false_n = int(row["false_clean_rows"])
        total_healthy = int(row["total_gold_healthy"])

        lines.append(
            f"| {row['profile']} | {selected_n} | {healthy_n} | {false_n} | "
            f"{pct(healthy_n, selected_n)} | {pct(healthy_n, total_healthy)} |"
        )

    best = sorted(
        all_summaries,
        key=lambda r: (
            int(r["selected_gold_healthy"]) / int(r["selected_candidates"])
            if int(r["selected_candidates"]) else 0.0,
            int(r["selected_candidates"]),
        ),
        reverse=True,
    )[0]

    best_precision = (
        int(best["selected_gold_healthy"]) / int(best["selected_candidates"])
        if int(best["selected_candidates"]) else 0.0
    )

    lines.append("")
    lines.append("## Current decision")
    lines.append("")

    if best_precision >= 0.95 and int(best["selected_candidates"]) >= 30:
        lines.append(
            f"Best profile `{best['profile']}` reaches >=95% calibration precision with enough candidates. "
            "Next step: manually review its sample before touching validation."
        )
    elif best_precision >= 0.95:
        lines.append(
            f"Best profile `{best['profile']}` reaches >=95% calibration precision but candidate count is tiny. "
            "Treat as proof-of-concept only."
        )
    else:
        lines.append(
            "No v1.2 profile reaches 95% calibration precision. Rule-only extraction is still insufficient."
        )

    summary_text = "\n".join(lines)

    (OUT_DIR / "profile_comparison_v1_2.md").write_text(summary_text, encoding="utf-8")
    pd.DataFrame(all_summaries).to_csv(OUT_DIR / "profile_comparison_v1_2.csv", index=False)

    print(summary_text)


if __name__ == "__main__":
    main()
