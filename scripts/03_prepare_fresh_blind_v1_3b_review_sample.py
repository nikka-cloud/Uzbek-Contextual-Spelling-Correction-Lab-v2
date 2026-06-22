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

OUT = BASE / "fresh_blind_review_sample_v1_3b"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 200
RANDOM_SEED = 2026061802


BAD_EXACT_TOKENS = {
    # old known failures
    "shaxt", "ananalar", "yeki", "dadamiing", "meditsinada", "dorivorlar",
    "quron", "mablag",

    # fresh-preview failures
    "uulki",
    "utkaziladi",
    "qilmaymen",
    "narsanidim",
    "yeildirakli",
    "ifodala",
    "chiqqan",
    "sanatkorlar",
    "tanikli",
    "yoo'zlashtirilgan",
    "bulvon",
}

BAD_SUBSTRINGS = {
    "markazidauniversitet",
    "mazmung'yasini",
    "tilme'yornutq",
    "so'roqsurishtirish",
    "grekbaqtriya",
    "konsignatoreksporterdan",
    "cholkampir",
}

FOREIGN_OR_LATIN_TECH_TOKENS = {
    "cauda",
}

ARCHAIC_RISK_TOKENS = {
    "qilurda",
    "mavlo",
    "taolo",
    "birla",
    "yorliqamish",
    "erdi",
    "bilasun",
}

ORDINAL_BAD_TOKENS = {
    "birinchi", "ikkinchi", "uchinchi", "to'rtinchi", "tortinchi",
    "beshinchi", "oltinchi", "yettinchi", "sakkizinchi", "to'qqizinchi",
    "oninchi", "o'ninchi", "yigirmanchi", "ottizinchi", "o'ttizinchi",
    "qirqinchi", "ellikinchi", "oltmishinchi", "yetmishinchi",
    "saksoninchi", "toqsoninchi", "to'qsoninchi", "yuzinchi", "minginchi",
}

UZ_SUFFIXES = {
    "ga", "ka", "qa", "da", "dan", "ni", "ning",
    "lar", "lari", "larni", "larning", "mi", "ku", "chi",
}

NUMBER_WORDS = {
    "nol", "bir", "ikki", "uch", "to'rt", "tort", "besh", "olti", "yetti",
    "sakkiz", "to'qqiz", "toqiz", "o'n", "on", "yigirma", "o'ttiz", "ottiz",
    "qirq", "ellik", "oltmish", "yetmish", "sakson", "to'qson", "toqson",
    "yuz", "ming", "million", "milliard", "butun", "o'ndan", "yuzdan",
}


def norm_apostrophe(s: str) -> str:
    return str(s).replace("ʻ", "'").replace("ʼ", "'").replace("`", "'")


def norm_token(tok: str) -> str:
    return (
        norm_apostrophe(tok)
        .lower()
        .strip(".,!?;:\"()[]{}«»“”")
    )


def tokens(text: str) -> list[str]:
    return [norm_token(x) for x in str(text).split() if norm_token(x)]


def has_midword_case_merge(text: str) -> bool:
    # markazidaUniversitet, GrekBaqtriya, MuhammadSolih style.
    return bool(re.search(r"[a-zʻʼ'][A-ZА-Я]", str(text)))


def suspicious_repeated_vowel(tok: str) -> bool:
    # catches yoo'zlashtirilgan, dadamiing-style damage
    return bool(re.search(r"(aa|ee|ii|oo|uu|yy)", tok))


def mostly_number_fragment(toks: list[str]) -> bool:
    if not toks:
        return False
    n = sum(1 for t in toks if t in NUMBER_WORDS)
    return len(toks) >= 5 and n / len(toks) >= 0.55


def strict_reasons(text: str) -> list[str]:
    reasons = []
    raw = str(text).strip()
    low = norm_apostrophe(raw).lower()
    toks = tokens(raw)

    if not raw.endswith("."):
        reasons.append("not_plain_final_period")

    if raw and raw[0].islower():
        reasons.append("starts_lowercase")

    if has_midword_case_merge(raw):
        reasons.append("midword_case_merge")

    if re.search(r"\b[A-Z][a-z]+[A-Z][a-z]+", raw):
        reasons.append("camelcase_like_token")

    if re.search(r"\b[a-zA-Z']{18,}\b", low):
        reasons.append("very_long_alpha_token_ge18")

    if mostly_number_fragment(toks):
        reasons.append("mostly_number_word_fragment")

    # Single-letter broken tail: "ifodala i."
    if re.search(r"\b[a-zʻʼ']{4,}\s+[a-z]\.$", low):
        reasons.append("single_letter_sentence_tail")

    # Number/formula/image reference damage: "yettti butun ... rasm"
    if "rasm" in toks and sum(1 for t in toks if t in NUMBER_WORDS) >= 2:
        reasons.append("number_formula_rasm_fragment")

    # Broken unit expression: "m s"
    if re.search(r"\b(m|km|sm|mm|kg|g)\s+(s|soat)\b", low):
        reasons.append("broken_unit_expression")

    # Spaced suffix, but avoid overblocking normal numerals as much as possible
    for i in range(len(toks) - 1):
        left, right = toks[i], toks[i + 1]
        if right in UZ_SUFFIXES and len(left) >= 4 and left not in NUMBER_WORDS:
            reasons.append(f"spaced_suffix:{left}_{right}")

    for tok in toks:
        if tok in BAD_EXACT_TOKENS:
            reasons.append(f"bad_exact_token:{tok}")
        if tok in FOREIGN_OR_LATIN_TECH_TOKENS:
            reasons.append(f"foreign_or_latin_token:{tok}")
        if tok in ARCHAIC_RISK_TOKENS:
            reasons.append(f"archaic_risk_token:{tok}")
        if tok in ORDINAL_BAD_TOKENS:
            reasons.append(f"ordinal_or_ocr_token:{tok}")
        if suspicious_repeated_vowel(tok):
            reasons.append(f"repeated_vowel_token:{tok}")

    for bad in BAD_SUBSTRINGS:
        if bad in low:
            reasons.append(f"bad_substring:{bad}")

    # Missing apostrophe common forms
    if re.search(r"\b(quron|mablag|sanat|utkaz|tanikli)\b", low):
        reasons.append("possible_missing_apostrophe_or_wrong_letter")

    # Merged words without punctuation
    if re.search(r"\b(gapso'z|cholkampir|tilme'yor|so'roqsurishtirish)\b", low):
        reasons.append("merged_or_unhyphenated_damage")

    # Too many apostrophe-heavy malformed tokens
    malformed_apostrophe = [
        t for t in toks
        if "'" in t and not re.search(r"(o'|g'|ko'|bo'|to'|so'|qo'|yo'|ma'|ta'|a'|e')", t)
    ]
    if len(malformed_apostrophe) >= 1:
        reasons.append("malformed_apostrophe_token")

    return sorted(set(reasons))


df = pd.read_csv(IN, dtype=str).fillna("")

required = {"sentence_id", "source_record_id", "part_number", "sentence_text"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns: {sorted(missing)}")

before = len(df)

df["sentence_text_norm_for_dedup"] = (
    df["sentence_text"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
)

df = df.drop_duplicates(subset=["sentence_text_norm_for_dedup"], keep="first").copy()
after_dedup = len(df)

df["v1_3b_block_reasons"] = df["sentence_text"].map(lambda x: "|".join(strict_reasons(x)))
df["fresh_v1_3b_candidate"] = df["v1_3b_block_reasons"].eq("")

candidates = df[df["fresh_v1_3b_candidate"]].copy()
blocked = df[~df["fresh_v1_3b_candidate"]].copy()

# Avoid reusing the opened v1.3 review sample.
old_review_path = BASE / "fresh_blind_review_sample_v1_3" / "manual_review_fresh_blind_200_v1_3.csv"
if old_review_path.exists():
    old = pd.read_csv(old_review_path, dtype=str).fillna("")
    old_ids = set(old["sentence_id"].astype(str))
    candidates = candidates[~candidates["sentence_id"].astype(str).isin(old_ids)].copy()

# Shuffle and balance by part.
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

candidate_path = OUT / "fresh_v1_3b_candidates.csv"
blocked_path = OUT / "fresh_v1_3b_blocked.csv"
review_path = OUT / "manual_review_fresh_blind_200_v1_3b.csv"
summary_path = OUT / "fresh_blind_review_sample_summary_v1_3b.md"

candidates.to_csv(candidate_path, index=False)
blocked.to_csv(blocked_path, index=False)
review.to_csv(review_path, index=False)

reason_counts = Counter()
for s in blocked["v1_3b_block_reasons"].astype(str):
    for r in s.split("|"):
        r = r.strip()
        if r:
            reason_counts[r] += 1

lines = []
lines.append("# Fresh Blind Review Sample v1.3b")
lines.append("")
lines.append("## Status")
lines.append("")
lines.append("This replaces the opened v1.3 fresh sample, which was visibly contaminated during preview.")
lines.append("")
lines.append("## Counts")
lines.append("")
lines.append(f"- plain_prefilter_candidates_input: `{before}`")
lines.append(f"- after_exact_text_dedup: `{after_dedup}`")
lines.append(f"- blocked_by_v1_3b_rules: `{len(blocked)}`")
lines.append(f"- remaining_candidates_after_blockers_and_old_sample_exclusion: `{len(candidates)}`")
lines.append(f"- manual_review_sample_rows: `{len(review)}`")
lines.append("")
lines.append("## Top v1.3b blocker reasons")
lines.append("")
for k, v in reason_counts.most_common(80):
    lines.append(f"- {k}: `{v}`")
lines.append("")
lines.append("## Files")
lines.append("")
lines.append(f"- candidates: `{candidate_path}`")
lines.append(f"- blocked: `{blocked_path}`")
lines.append(f"- manual_review_sample: `{review_path}`")
lines.append("")
lines.append("## Manual labels")
lines.append("")
lines.append("- `CLEAN_TARGET`")
lines.append("- `NOT_CLEAN_TARGET`")
lines.append("")
lines.append("## Decision rule")
lines.append("")
lines.append("- Review the 200 rows only if the preview looks mostly clean.")
lines.append("- If reviewed precision >= 95%, v1.3b passes fresh blind candidate validation.")
lines.append("- If precision < 95%, do not use this extractor for corpus-wide clean target extraction.")

summary_path.write_text("\n".join(lines), encoding="utf-8")

print(summary_path.read_text(encoding="utf-8"))

print("\nFIRST 60 REVIEW ROWS")
for i, row in review.head(60).reset_index(drop=True).iterrows():
    print("\n" + "=" * 120)
    print("ROW:", i + 1)
    print("part:", row["part_number"])
    print("sentence_id:", row["sentence_id"])
    print("text:", row["sentence_text"])
