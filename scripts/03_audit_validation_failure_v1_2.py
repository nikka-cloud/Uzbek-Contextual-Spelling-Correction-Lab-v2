from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd


BASE = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1/"
    "strict_clean_seed_extractor_v1_2/"
    "validation_eval_v1_2"
)

LABELED = BASE / "validation_predictions_labeled_eval_v1_2.csv"
FALSE_CLEAN = BASE / "validation_false_clean_v1_2.csv"
SELECTED = BASE / "validation_selected_candidates_v1_2.csv"

OUT = BASE / "validation_failure_audit_v1_2"


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


WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+")


def pick_col(df: pd.DataFrame, cols: list[str]) -> str:
    for c in cols:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any of: {cols}")


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


def token_flags(text: str) -> list[str]:
    text = norm_ap(text)
    tokens = WORD_RE.findall(text)
    clean = [t.strip(".,;:!?()[]{}\"'").lower() for t in tokens]

    flags = []

    suspicious_exact = {
        "shaxt",
        "ananalar",
        "yeki",
        "dadamiing",
        "meditsinada",
        "dorivorlar",
    }

    for tok in clean:
        if tok in suspicious_exact:
            flags.append(f"suspicious_exact_token:{tok}")

        if len(tok) >= 8 and re.search(r"ii|aaa|ooo|uuu", tok):
            flags.append(f"repeated_vowel_pattern:{tok}")

        if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", tok):
            flags.append(f"long_consonant_cluster:{tok}")

        if tok.endswith("iing"):
            flags.append(f"suspicious_iing_ending:{tok}")

    if re.search(r"\byil\s+may\b", text, re.I):
        flags.append("broken_date_like_phrase:yil_may")

    if re.search(r"\bhar\s+qanday\s+faoliyati\b", text, re.I):
        flags.append("suspicious_grammar_phrase:har_qanday_faoliyati")

    if re.search(r"\bbeshinchi\s+hujayralar\b", text, re.I):
        flags.append("injected_ordinal_phrase:beshinchi_hujayralar")

    if re.search(r"\bbir\s+maygacha\b", text, re.I):
        flags.append("date_word_phrase:bir_maygacha")

    return sorted(set(flags))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    labeled = pd.read_csv(LABELED, dtype=str).fillna("")
    false_clean = pd.read_csv(FALSE_CLEAN, dtype=str).fillna("")
    selected = pd.read_csv(SELECTED, dtype=str).fillna("")

    text_col = pick_col(labeled, TEXT_COLS)
    label_col = pick_col(labeled, LABEL_COLS)
    note_col = next((c for c in NOTE_COLS if c in labeled.columns), None)

    selected = selected.copy()
    false_clean = false_clean.copy()

    selected["diagnostic_token_flags"] = selected[text_col].map(lambda x: "|".join(token_flags(x)))
    false_clean["diagnostic_token_flags"] = false_clean[text_col].map(lambda x: "|".join(token_flags(x)))

    print("\n" + "=" * 120)
    print("VALIDATION FAILURE AUDIT v1.2")
    print("=" * 120)

    print("\nSelected label distribution:")
    print(selected[label_col].value_counts(dropna=False).to_string())

    print("\nFalse clean rows:", len(false_clean))

    flag_counter = Counter()
    for flags in false_clean["diagnostic_token_flags"]:
        if not flags:
            flag_counter["NO_FLAG"] += 1
        else:
            for f in flags.split("|"):
                flag_counter[f] += 1

    print("\nDiagnostic flags on false-clean rows:")
    for k, v in flag_counter.most_common():
        print(f"{k}: {v}")

    print("\nAll selected validation rows:")
    for i, row in selected.reset_index(drop=True).iterrows():
        print("\n" + "=" * 120)
        print(f"ROW {i+1}")
        print("label:", row[label_col])
        if note_col:
            print("notes:", row.get(note_col, ""))
        print("p_stage1_usable:", row.get("p_stage1_usable", ""))
        print("flags:", row.get("diagnostic_token_flags", ""))
        print("text:", row[text_col])

    selected.to_csv(OUT / "validation_selected_with_failure_flags_v1_2.csv", index=False)
    false_clean.to_csv(OUT / "validation_false_clean_with_failure_flags_v1_2.csv", index=False)

    lines = []
    lines.append("# Validation Failure Audit v1.2")
    lines.append("")
    lines.append("## Main result")
    lines.append("")
    lines.append("- Frozen v1.2 failed validation.")
    lines.append("- The failure is mostly local token corruption, not broad sentence-shape corruption.")
    lines.append("")
    lines.append("## Selected label distribution")
    lines.append("")
    for label, count in selected[label_col].value_counts(dropna=False).items():
        lines.append(f"- {label}: `{count}`")
    lines.append("")
    lines.append("## False-clean diagnostic flags")
    lines.append("")
    for k, v in flag_counter.most_common():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Engineering decision")
    lines.append("")
    lines.append("Do not tune v1.2 and call it validated.")
    lines.append("Build v1.3 with lexical/token-quality scoring, then validate on a new blind sample.")

    (OUT / "validation_failure_audit_v1_2.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nWROTE:")
    print(OUT / "validation_selected_with_failure_flags_v1_2.csv")
    print(OUT / "validation_false_clean_with_failure_flags_v1_2.csv")
    print(OUT / "validation_failure_audit_v1_2.md")


if __name__ == "__main__":
    main()
