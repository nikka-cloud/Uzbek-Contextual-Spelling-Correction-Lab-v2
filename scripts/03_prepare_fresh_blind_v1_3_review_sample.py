from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd


BASE = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1/"
    "strict_clean_seed_extractor_v1_3_lexical_gate"
)

IN = (
    BASE
    / "fresh_pool_exclusion_audit_v1_3"
    / "fresh_pool_plain_shape_prefilter_candidates_v1_3.csv"
)

OUT = BASE / "fresh_blind_review_sample_v1_3"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 200
RANDOM_SEED = 20260618


UZ_SUFFIXES = {
    "ga", "ka", "qa",
    "da", "dan",
    "ni", "ning",
    "lar", "lari", "larni", "larning",
    "mi", "ku", "chi",
}

ORDINAL_BAD_TOKENS = {
    "birinchi", "ikkinchi", "uchinchi", "to'rtinchi", "tortinchi",
    "beshinchi", "oltinchi", "yettinchi", "sakkizinchi", "to'qqizinchi",
    "oninchi", "o'ninchi",
    "yigirmanchi", "ottizinchi", "o'ttizinchi",
    "qirqinchi", "ellikinchi", "oltmishinchi", "yetmishinchi",
    "saksoninchi", "toqsoninchi", "to'qsoninchi",
    "yuzinchi", "minginchi",
}

KNOWN_BAD_TOKENS = {
    "shaxt",
    "ananalar",
    "yeki",
    "dadamiing",
    "meditsinada",
    "dorivorlar",
    "quron",
    "mablag",
    "geofafiyani",
    "yeifatida",
    "karalmayottanligi",
    "kurinadi",
    "xjsqorida",
    "izolatsivasida",
    "demaknadonning",
    "yemakhayvonning",
}


def norm_token(tok: str) -> str:
    return (
        tok.lower()
        .replace("ʻ", "'")
        .replace("ʼ", "'")
        .replace("`", "'")
        .strip(".,!?;:\"()[]{}«»“”")
    )


def tokens(text: str) -> list[str]:
    return [norm_token(x) for x in text.split() if norm_token(x)]


def strict_block_reasons(text: str) -> list[str]:
    reasons = []
    t = str(text).strip()
    low = t.lower().replace("ʻ", "'").replace("ʼ", "'").replace("`", "'")
    toks = tokens(t)

    # 1) Exact known corruption tokens / OCR leftovers
    for tok in toks:
        if tok in KNOWN_BAD_TOKENS:
            reasons.append(f"known_bad_token:{tok}")
        if tok in ORDINAL_BAD_TOKENS:
            reasons.append(f"ordinal_or_ocr_token:{tok}")

    # 2) Number-word + ordinal/OCR phrase patterns
    if re.search(
        r"\b(bir|ikki|uch|to'rt|tort|besh|olti|yetti|sakkiz|to'qqiz|on|o'n|yigirma|qirq|ellik|oltmish|yetmish|sakson|to'qson)\s+(yuzinchi|minginchi)\b",
        low,
    ):
        reasons.append("number_word_plus_ocr_ordinal")

    # 3) Broken spaced suffix: "yarmarkasi ga", "Pravda ga", "Toklar ga"
    for i in range(len(toks) - 1):
        left, right = toks[i], toks[i + 1]
        if right in UZ_SUFFIXES and len(left) >= 3:
            reasons.append(f"spaced_suffix:{left}_{right}")

    # 4) Broken unit expression: "m s", "km soat", etc.
    if re.search(r"\b(m|km|sm|mm|kg|g)\s+(s|soat)\b", low):
        reasons.append("broken_unit_expression")

    # 5) Obvious quote/apostrophe missing in high-value common words
    if re.search(r"\bquron\b", low):
        reasons.append("possible_missing_apostrophe:quron")
    if re.search(r"\bmablag\b", low):
        reasons.append("possible_missing_apostrophe:mablag")

    # 6) Unwanted textbook/list-like sentence endings
    if re.search(r"\b(yuzinchi|minginchi|ellikinchi)\.$", low):
        reasons.append("ocr_ordinal_sentence_ending")

    # 7) Too many uppercase tokens in a short sentence can be news/header-ish or acronym-heavy
    uppercase_tokens = [x for x in str(text).split() if len(x) > 1 and x.isupper()]
    if len(uppercase_tokens) >= 2:
        reasons.append("too_many_uppercase_tokens")

    return sorted(set(reasons))


df = pd.read_csv(IN, dtype=str).fillna("")

required = {"sentence_id", "source_record_id", "part_number", "sentence_text"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns: {sorted(missing)}")

before = len(df)

# Deduplicate exact text to avoid repeated weather/template sentences.
df["sentence_text_norm_for_dedup"] = (
    df["sentence_text"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
)

df = df.drop_duplicates(subset=["sentence_text_norm_for_dedup"], keep="first").copy()
after_dedup = len(df)

df["strict_extra_block_reasons"] = df["sentence_text"].map(lambda x: "|".join(strict_block_reasons(x)))
df["fresh_v1_3_candidate"] = df["strict_extra_block_reasons"].eq("")

candidates = df[df["fresh_v1_3_candidate"]].copy()
blocked = df[~df["fresh_v1_3_candidate"]].copy()

# Balanced-ish sample across parts: shuffle first, then take up to 2 per part before filling.
candidates = candidates.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

sample_parts = []
for _, group in candidates.groupby("part_number", sort=False):
    sample_parts.append(group.head(2))

balanced = pd.concat(sample_parts, ignore_index=True) if sample_parts else candidates.head(0)
balanced = balanced.sample(frac=1, random_state=RANDOM_SEED).head(SAMPLE_SIZE)

if len(balanced) < SAMPLE_SIZE:
    remaining = candidates[~candidates["sentence_id"].isin(set(balanced["sentence_id"]))]
    fill = remaining.head(SAMPLE_SIZE - len(balanced))
    balanced = pd.concat([balanced, fill], ignore_index=True)

review = balanced.copy()
review.insert(0, "manual_clean_label", "")
review.insert(1, "manual_notes", "")

review_cols = [
    "manual_clean_label",
    "manual_notes",
    "sentence_id",
    "source_record_id",
    "part_number",
    "document_sample_id",
    "sentence_index_in_record",
    "sentence_character_count",
    "sentence_token_count",
    "sentence_text",
]

review = review[review_cols]

candidate_path = OUT / "fresh_v1_3_candidates_after_extra_blockers.csv"
blocked_path = OUT / "fresh_v1_3_blocked_by_extra_blockers.csv"
review_path = OUT / "manual_review_fresh_blind_200_v1_3.csv"
summary_path = OUT / "fresh_blind_review_sample_summary_v1_3.md"

candidates.to_csv(candidate_path, index=False)
blocked.to_csv(blocked_path, index=False)
review.to_csv(review_path, index=False)

reason_counts = Counter()
for s in blocked["strict_extra_block_reasons"].astype(str):
    for r in s.split("|"):
        r = r.strip()
        if r:
            reason_counts[r] += 1

lines = []
lines.append("# Fresh Blind Review Sample v1.3")
lines.append("")
lines.append("## Purpose")
lines.append("")
lines.append("Prepare a fresh blind manual-review sample from unseen sentence pool after excluding old reviewed source records.")
lines.append("")
lines.append("## Counts")
lines.append("")
lines.append(f"- plain_prefilter_candidates_input: `{before}`")
lines.append(f"- after_exact_text_dedup: `{after_dedup}`")
lines.append(f"- blocked_by_extra_token_rules: `{len(blocked)}`")
lines.append(f"- remaining_fresh_v1_3_candidates: `{len(candidates)}`")
lines.append(f"- manual_review_sample_rows: `{len(review)}`")
lines.append("")
lines.append("## Top extra blocker reasons")
lines.append("")
for k, v in reason_counts.most_common(50):
    lines.append(f"- {k}: `{v}`")
lines.append("")
lines.append("## Files")
lines.append("")
lines.append(f"- candidates: `{candidate_path}`")
lines.append(f"- blocked: `{blocked_path}`")
lines.append(f"- manual_review_sample: `{review_path}`")
lines.append("")
lines.append("## Manual labels to use")
lines.append("")
lines.append("- `CLEAN_TARGET`")
lines.append("- `NOT_CLEAN_TARGET`")
lines.append("")
lines.append("## Decision rule")
lines.append("")
lines.append("- If manual precision >= 95%, v1.3 passes fresh blind candidate validation.")
lines.append("- If manual precision < 95%, inspect false clean rows and revise gate.")

summary_path.write_text("\n".join(lines), encoding="utf-8")

print(summary_path.read_text(encoding="utf-8"))

print("\nFIRST 40 MANUAL REVIEW ROWS")
for i, row in review.head(40).iterrows():
    print("\n" + "=" * 120)
    print("ROW:", i + 1)
    print("part:", row["part_number"])
    print("sentence_id:", row["sentence_id"])
    print("text:", row["sentence_text"])
