from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


BASE = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1/"
    "strict_clean_seed_extractor_v1"
)

CANDIDATES = BASE / "calibration_strict_clean_seed_candidates_v1.csv"
FALSE_CLEAN = BASE / "calibration_strict_clean_seed_false_clean_v1.csv"
MISSED_HEALTHY = BASE / "calibration_strict_clean_seed_missed_healthy_v1.csv"

OUT = BASE / "strict_clean_seed_v1_error_audit"


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

NOTE_COLS = [
    "gold_health_notes",
    "human_health_notes",
    "review_notes",
    "notes",
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


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


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


WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
BAD_BIGRAM_RE = re.compile(r"[bcdfghjklmnpqrstvwxyz]{5,}", re.I)
REPEATED_RE = re.compile(r"([a-zA-Z])\1\1")
MID_CAP_RE = re.compile(r"[a-zà-öø-ÿ][A-ZÀ-ÖØ-Þ]")
SPACED_SUFFIX_RE = re.compile(
    r"\b([A-Za-zÀ-ÖØ-öø-ÿ']{3,})\s+"
    r"(ning|ni|ga|da|dan|gi|lik|ligi|lar|lari|siz|miz|chi|cha)\b",
    re.I,
)
EXAM_FORMAT_RE = re.compile(
    r"\b(test|savol|javob|belgilang|variant|jadval|rasm|formula|mavzu|reja|inv)\b",
    re.I,
)
FOREIGN_RE = re.compile(
    r"\b(popups?|popunders?|browser|cache|account|dashboard|login|support|"
    r"website|online|email|windows|linux|android|iphone)\b",
    re.I,
)


def diagnostic_flags(text: str) -> list[str]:
    text = norm_ap(text).strip()
    lowered = text.lower()
    tokens = WORD_RE.findall(text)

    flags = []

    if len(tokens) < 5:
        flags.append("few_tokens")

    if len(tokens) > 32:
        flags.append("many_tokens")

    if len(text) < 25:
        flags.append("short_sentence")

    if len(text) > 220:
        flags.append("long_sentence")

    if text and text[-1] not in ".?!":
        flags.append("no_terminal_punctuation")

    if CYRILLIC_RE.search(text):
        flags.append("contains_cyrillic")

    if MID_CAP_RE.search(text):
        flags.append("mid_word_capital_or_merge")

    if SPACED_SUFFIX_RE.search(text):
        flags.append("spaced_suffix_damage")

    if EXAM_FORMAT_RE.search(text):
        flags.append("exam_or_format_keyword")

    if FOREIGN_RE.search(text):
        flags.append("foreign_tech_token")

    if re.search(r"\d", text):
        flags.append("contains_digit")

    if text.count(",") >= 4:
        flags.append("many_commas")

    if text.count(";") >= 1:
        flags.append("semicolon")

    if text.count(":") >= 1:
        flags.append("colon")

    for tok in tokens:
        t = norm_ap(tok).strip(".,;:!?()[]{}\"").lower()

        if len(t) >= 14:
            flags.append("very_long_token")
            break

        if BAD_BIGRAM_RE.search(t):
            flags.append("consonant_cluster")
            break

        if REPEATED_RE.search(t):
            flags.append("triple_repeated_char")
            break

        if "w" in t:
            flags.append("contains_w")
            break

        if "c" in t and "ch" not in t:
            flags.append("suspicious_c_without_ch")
            break

    known_bad = [
        "stolstul",
        "savdosotiq",
        "yopyorug",
        "o'ksibo'ksib",
        "faromush",
        "qipayotir",
        "noyo lik",
        "tashiluvchanli",
        "kermdiaga",
        "ismhoi",
        "tobe'ro",
        "dihed",
    ]

    if any(x in lowered for x in known_bad):
        flags.append("known_bad_fragment")

    return sorted(set(flags))


def token_counter(df: pd.DataFrame, text_col: str) -> Counter:
    c = Counter()
    for text in df[text_col].astype(str):
        for tok in WORD_RE.findall(norm_ap(text).lower()):
            tok = tok.strip(".,;:!?()[]{}\"'")
            if len(tok) >= 3:
                c[tok] += 1
    return c


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(CANDIDATES, dtype=str).fillna("")
    false_clean = pd.read_csv(FALSE_CLEAN, dtype=str).fillna("")
    missed_healthy = pd.read_csv(MISSED_HEALTHY, dtype=str).fillna("")

    text_col = pick_col(candidates, TEXT_COLS)
    label_col = pick_col(candidates, LABEL_COLS)
    note_col = pick_col(candidates, NOTE_COLS)

    if not text_col:
        raise ValueError("Could not find text column.")
    if not label_col:
        raise ValueError("Could not find label column.")

    print("\n" + "=" * 120)
    print("STRICT CLEAN SEED V1 ERROR AUDIT")
    print("=" * 120)

    print("\nSelected candidate label distribution:")
    print(candidates[label_col].value_counts(dropna=False).to_string())

    print("\nFalse-clean label distribution:")
    print(false_clean[label_col].value_counts(dropna=False).to_string())

    if "p_stage1_usable" in candidates.columns:
        print("\nStage1 usable score by selected label:")
        tmp = candidates.copy()
        tmp["p_stage1_usable"] = pd.to_numeric(tmp["p_stage1_usable"], errors="coerce")
        print(
            tmp.groupby(label_col)["p_stage1_usable"]
            .describe()[["count", "mean", "min", "25%", "50%", "75%", "max"]]
            .to_string()
        )

    false_clean["new_diagnostic_flags"] = false_clean[text_col].map(
        lambda x: "|".join(diagnostic_flags(x))
    )

    missed_healthy["new_diagnostic_flags"] = missed_healthy[text_col].map(
        lambda x: "|".join(diagnostic_flags(x))
    )

    flag_counter = Counter()
    for flags in false_clean["new_diagnostic_flags"]:
        if not flags:
            flag_counter["NO_NEW_FLAG"] += 1
        else:
            for f in flags.split("|"):
                flag_counter[f] += 1

    print("\nNew diagnostic flags among false-clean rows:")
    for k, v in flag_counter.most_common():
        print(f"{k}: {v}")

    healthy_selected = candidates[candidates[label_col].eq("HEALTHY")].copy()

    false_tokens = token_counter(false_clean, text_col)
    healthy_tokens = token_counter(healthy_selected, text_col)

    suspicious_tokens = []
    for tok, fc in false_tokens.items():
        hc = healthy_tokens.get(tok, 0)

        if fc >= 2 and hc == 0:
            suspicious_tokens.append((tok, fc, hc))
        elif fc >= 2 and fc >= hc * 3:
            suspicious_tokens.append((tok, fc, hc))

    suspicious_tokens = sorted(
        suspicious_tokens,
        key=lambda x: (x[1], -x[2], x[0]),
        reverse=True,
    )

    print("\nTop tokens overrepresented in false-clean rows:")
    for tok, fc, hc in suspicious_tokens[:80]:
        print(f"{tok}\tfalse_clean={fc}\tselected_healthy={hc}")

    export_cols = []

    for c in ID_COLS:
        if c in false_clean.columns and c not in export_cols:
            export_cols.append(c)

    for c in [
        label_col,
        note_col,
        "p_stage1_usable",
        "predicted_route",
        text_col,
        "new_diagnostic_flags",
    ]:
        if c and c in false_clean.columns and c not in export_cols:
            export_cols.append(c)

    for c in false_clean.columns:
        if c not in export_cols:
            export_cols.append(c)

    false_clean.insert(0, "manual_defect_type", "")
    false_clean.insert(1, "manual_blocker_rule_needed", "")
    false_clean.insert(2, "manual_comment", "")

    false_clean.to_csv(
        OUT / "false_clean_priority_review_v1.csv",
        index=False,
    )

    missed_healthy.to_csv(
        OUT / "missed_healthy_with_diagnostics_v1.csv",
        index=False,
    )

    with (OUT / "false_clean_examples_printable_v1.txt").open("w", encoding="utf-8") as f:
        for i, row in false_clean.iterrows():
            f.write("\n" + "=" * 120 + "\n")
            f.write(f"row_number: {i}\n")
            f.write(f"label: {row.get(label_col, '')}\n")
            if note_col:
                f.write(f"notes: {row.get(note_col, '')}\n")
            if "p_stage1_usable" in row.index:
                f.write(f"p_stage1_usable: {row.get('p_stage1_usable', '')}\n")
            f.write(f"new_diagnostic_flags: {row.get('new_diagnostic_flags', '')}\n")
            f.write(f"text: {row.get(text_col, '')}\n")

    print("\nOutput files:")
    print(OUT / "false_clean_priority_review_v1.csv")
    print(OUT / "missed_healthy_with_diagnostics_v1.csv")
    print(OUT / "false_clean_examples_printable_v1.txt")

    print("\nFirst 40 false-clean examples:")
    for i, row in false_clean.head(40).iterrows():
        print("\n" + "=" * 120)
        print("label:", row.get(label_col, ""))
        if note_col:
            print("notes:", row.get(note_col, ""))
        print("p_stage1_usable:", row.get("p_stage1_usable", ""))
        print("flags:", row.get("new_diagnostic_flags", ""))
        print("text:", row.get(text_col, ""))


if __name__ == "__main__":
    main()
