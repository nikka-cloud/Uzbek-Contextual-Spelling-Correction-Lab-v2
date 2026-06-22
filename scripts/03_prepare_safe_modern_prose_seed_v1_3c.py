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

OLD_OPENED = [
    BASE / "fresh_blind_review_sample_v1_3" / "manual_review_fresh_blind_200_v1_3.csv",
    BASE / "fresh_blind_review_sample_v1_3b" / "manual_review_fresh_blind_200_v1_3b.csv",
]

OUT = BASE / "safe_modern_prose_seed_v1_3c"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 200
RANDOM_SEED = 2026061803


COMMON_UZ_ANCHORS = {
    "bu", "shu", "ushbu", "bunda", "biroq", "ammo", "lekin",
    "uchun", "bilan", "haqida", "tomonidan", "bo'yicha", "sababli",
    "ham", "esa", "yana", "juda", "eng", "ko'p", "kam",
    "uning", "ular", "u", "o'z", "biz", "men",
    "deb", "deya", "qildi", "qilindi", "etdi", "berdi", "bo'ldi",
    "bo'lgan", "bo'lib", "mumkin", "kerak", "lozim",
    "yil", "oy", "kuni", "vaqt", "natijada",
}

BAD_EXACT = {
    "naytida", "oralikpi", "kassalanga", "net", "narsanidim",
    "qilmaymen", "yeildirakli", "ifodala", "sanatkorlar", "tanikli",
    "sotiboldi", "mablag", "quron", "xam", "jag", "utkaziladi",
    "shaxt", "ananalar", "yeki", "dadamiing", "meditsinada", "dorivorlar",
}

BAD_SUBSTRINGS = {
    "markazidauniversitet", "tilme'yornutq", "mazmung'yasini",
    "grekbaqtriya", "konsignatoreksporterdan", "kuchlanishbir",
    "ishlabchiqarish", "to'qimapo'kak", "bo'g'imchanoq",
    "buqahramon", "so'roqsurishtirish",
}

DOMAIN_RISK = {
    # textbook / biology / chemistry / medicine / grammar / legal-heavy / technical
    "biotsenoz", "meteorologik", "sferik", "aberratsiya", "linza",
    "jo'g'rofik", "arxeolog", "inkassatsiya", "inkubatsion",
    "kulminatsiya", "istiora", "istioraviy", "artikulyatsiya",
    "mototsikl", "individual", "sulola", "elektrolit", "ionlanish",
    "gipoteza", "ortodontik", "infeksionist", "metilen",
    "mikroprotsessor", "zveno", "integrirlovchi", "bikarbonat",
    "bufer", "html", "kambiy", "protoplast", "injenerligi",
    "olmosh", "tarozilar", "konsignatsiya", "sterjni", "qalay",
    "kristallaridan", "rift", "linzalarning", "bo'g'im",
}

FOREIGN_MARKERS = {
    "nas", "ved", "prekrasnie", "dve", "komnati", "patnuslarni",
    "kursiy", "ruf", "shtyox", "kastellyani",
}

ARCHAIC_LITERARY = {
    "g'ulom", "shotiga", "qilurda", "mavlo", "taolo", "birla",
    "yorliqamish", "erdi", "bilasun", "senlar", "kelgansanlar",
    "mayog'idir", "qorasuvga",
}

NUMBER_WORDS = {
    "nol", "bir", "ikki", "uch", "to'rt", "tort", "besh", "olti", "yetti",
    "sakkiz", "to'qqiz", "o'n", "yigirma", "o'ttiz", "qirq", "ellik",
    "oltmish", "yetmish", "sakson", "to'qson", "yuz", "ming", "million",
    "milliard", "butun", "o'ndan", "yuzdan",
}


def norm_apostrophe(s: str) -> str:
    return str(s).replace("ʻ", "'").replace("ʼ", "'").replace("`", "'")


def norm_token(tok: str) -> str:
    return norm_apostrophe(tok).lower().strip(".,!?;:\"()[]{}«»“”")


def toks(text: str) -> list[str]:
    return [norm_token(x) for x in str(text).split() if norm_token(x)]


def is_bad_repeated_vowel(tok: str) -> bool:
    safe = {
        "tabiiy", "badiiy", "muayyan", "murojaat", "tayyor", "tayyorlash",
        "tayyorlangan", "mudofaa", "manfaat", "manfaatli", "taalluqli",
        "inshoot", "sayyoh", "taassurot", "koordinata",
    }
    if tok in safe:
        return False
    return bool(re.search(r"(aaa|eee|iii|ooo|uuu|yyy|aa|ii|yy)", tok))


def reasons(text: str) -> list[str]:
    raw = str(text).strip()
    low = norm_apostrophe(raw).lower()
    tokens = toks(raw)
    out = []

    wc = len(tokens)
    cc = len(raw)

    if not raw.endswith("."):
        out.append("not_final_period")
    if raw and raw[0].islower():
        out.append("starts_lowercase")
    if "," in raw:
        out.append("comma_present")
    if ":" in raw or "?" in raw or "!" in raw:
        out.append("question_exclamation_colon")
    if wc < 6:
        out.append("too_few_words")
    if wc > 18:
        out.append("too_many_words")
    if cc < 35:
        out.append("too_short_chars")
    if cc > 150:
        out.append("too_long_chars")

    if re.search(r"[a-zʻʼ'][A-Z]", raw):
        out.append("midword_case_merge")

    if re.search(r"\b[a-zA-Zʻʼ']{18,}\b", raw):
        out.append("very_long_token")

    if re.search(r"\b[a-z]\s+[a-z]{1,3}\s+[a-z]{1,3}\b", low):
        out.append("spaced_letter_garbage")

    if re.search(r"\b[a-z]\b", low):
        single_letters = [t for t in tokens if len(t) == 1 and t not in {"u"}]
        if single_letters:
            out.append("single_letter_token")

    anchor_count = sum(1 for t in tokens if t in COMMON_UZ_ANCHORS)
    if anchor_count < 2:
        out.append("low_common_uzbek_anchor_count")

    number_count = sum(1 for t in tokens if t in NUMBER_WORDS)
    if number_count >= 3:
        out.append("number_word_heavy")

    for t in tokens:
        if t in BAD_EXACT:
            out.append(f"bad_exact:{t}")
        if t in DOMAIN_RISK:
            out.append(f"domain_risk:{t}")
        if t in FOREIGN_MARKERS:
            out.append(f"foreign_marker:{t}")
        if t in ARCHAIC_LITERARY:
            out.append(f"archaic_literary:{t}")
        if is_bad_repeated_vowel(t):
            out.append(f"bad_repeated_vowel:{t}")

    for sub in BAD_SUBSTRINGS:
        if sub in low:
            out.append(f"bad_substring:{sub}")

    if re.search(r"\b[A-Z]{2,}\b", raw):
        out.append("uppercase_acronym")

    if re.search(r"\b[A-Z][a-z]+[A-Z][a-z]+", raw):
        out.append("camelcase_like_token")

    return sorted(set(out))


df = pd.read_csv(IN, dtype=str).fillna("")

opened_ids = set()
for p in OLD_OPENED:
    if p.exists():
        old = pd.read_csv(p, dtype=str).fillna("")
        if "sentence_id" in old.columns:
            opened_ids.update(old["sentence_id"].astype(str))

df = df[~df["sentence_id"].astype(str).isin(opened_ids)].copy()

df["norm_text"] = (
    df["sentence_text"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
)
df = df.drop_duplicates(subset=["norm_text"], keep="first").copy()

df["v1_3c_reasons"] = df["sentence_text"].map(lambda x: "|".join(reasons(x)))
df["safe_modern_prose_seed_v1_3c"] = df["v1_3c_reasons"].eq("")

selected = df[df["safe_modern_prose_seed_v1_3c"]].copy()
blocked = df[~df["safe_modern_prose_seed_v1_3c"]].copy()

selected = selected.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# max 2 per part for preview diversity
pieces = []
for _, g in selected.groupby("part_number", sort=False):
    pieces.append(g.head(2))
sample = pd.concat(pieces, ignore_index=True) if pieces else selected.head(0)
sample = sample.sample(frac=1, random_state=RANDOM_SEED).head(SAMPLE_SIZE)

if len(sample) < SAMPLE_SIZE:
    rest = selected[~selected["sentence_id"].isin(set(sample["sentence_id"]))]
    sample = pd.concat([sample, rest.head(SAMPLE_SIZE - len(sample))], ignore_index=True)

review = sample.copy()
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

selected_path = OUT / "safe_modern_prose_candidates_v1_3c.csv"
blocked_path = OUT / "blocked_safe_modern_prose_v1_3c.csv"
review_path = OUT / "manual_review_safe_modern_prose_200_v1_3c.csv"
summary_path = OUT / "safe_modern_prose_seed_summary_v1_3c.md"

selected.to_csv(selected_path, index=False)
blocked.to_csv(blocked_path, index=False)
review.to_csv(review_path, index=False)

cnt = Counter()
for s in blocked["v1_3c_reasons"].astype(str):
    for r in s.split("|"):
        if r:
            cnt[r] += 1

lines = []
lines.append("# Safe Modern Prose Seed v1.3c")
lines.append("")
lines.append("## Status")
lines.append("")
lines.append("Development sieve after v1.3 and v1.3b previews failed.")
lines.append("")
lines.append("## Counts")
lines.append("")
lines.append(f"- input_after_opened_sample_exclusion_and_dedup: `{len(df)}`")
lines.append(f"- selected_safe_modern_prose_candidates: `{len(selected)}`")
lines.append(f"- blocked_rows: `{len(blocked)}`")
lines.append(f"- review_preview_rows: `{len(review)}`")
lines.append("")
lines.append("## Top blocker reasons")
lines.append("")
for k, v in cnt.most_common(80):
    lines.append(f"- {k}: `{v}`")
lines.append("")
lines.append("## Files")
lines.append("")
lines.append(f"- selected: `{selected_path}`")
lines.append(f"- blocked: `{blocked_path}`")
lines.append(f"- review_preview: `{review_path}`")
lines.append("")
lines.append("## Rule")
lines.append("")
lines.append("Do not label until the preview looks clean enough by visual inspection.")

summary_path.write_text("\n".join(lines), encoding="utf-8")

print(summary_path.read_text(encoding="utf-8"))

print("\nFIRST 80 REVIEW PREVIEW ROWS")
for i, row in review.head(80).reset_index(drop=True).iterrows():
    print("\n" + "=" * 120)
    print("ROW:", i + 1)
    print("part:", row["part_number"])
    print("sentence_id:", row["sentence_id"])
    print("text:", row["sentence_text"])
