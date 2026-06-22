from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


POOL = Path(
    "outputs/03_sentence_inventory/"
    "global_validation_sentence_pool_v1/"
    "global_validation_sentences_v1.jsonl"
)

OLD_MASTER = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1/"
    "global_sentence_health_review_master_v1.csv"
)

BASE = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1/"
    "strict_clean_seed_extractor_v1_3_lexical_gate"
)

OPENED_PREVIEWS = [
    BASE / "fresh_blind_review_sample_v1_3" / "manual_review_fresh_blind_200_v1_3.csv",
    BASE / "fresh_blind_review_sample_v1_3b" / "manual_review_fresh_blind_200_v1_3b.csv",
    BASE / "safe_modern_prose_seed_v1_3c" / "manual_review_safe_modern_prose_200_v1_3c.csv",
]

OUT = BASE / "document_safe_clean_seed_v1_4"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 200
RANDOM_SEED = 2026061804


COMMON_UZ_ANCHORS = {
    "bu", "shu", "ushbu", "bunda", "buning", "bunday",
    "uchun", "bilan", "haqida", "tomonidan", "bo'yicha",
    "ham", "esa", "yana", "juda", "eng", "ko'p", "kam",
    "uning", "ular", "u", "o'z", "biz", "men",
    "mumkin", "kerak", "lozim", "bo'ladi", "bo'lgan", "qildi",
    "ma'lum", "xabar", "berdi", "aytdi", "ta'kidladi",
    "sababli", "orqali", "davomida", "natijada", "shuningdek",
    "lekin", "ammo", "biroq", "chunki", "agar",
    "yil", "kuni", "bugun", "kecha", "hozir",
}

NUMBER_WORDS = {
    "nol", "bir", "ikki", "uch", "to'rt", "tort", "besh", "olti", "yetti", "sakkiz", "to'qqiz",
    "o'n", "on", "yigirma", "o'ttiz", "ottiz", "qirq", "ellik", "oltmish", "yetmish",
    "sakson", "to'qson", "toqson", "yuz", "ming", "million", "milliard",
    "yarim", "butun", "foiz",
}

ORDINAL_RISK = {
    "birinchi", "ikkinchi", "uchinchi", "to'rtinchi", "tortinchi", "beshinchi",
    "oltinchi", "yettinchi", "sakkizinchi", "to'qqizinchi", "toqqizinchi",
    "o'ninchi", "oninchi", "yigirmanchi", "o'ttizinchi", "ottizinchi",
    "ellikinchi", "yuzinchi", "minginchi",
}

BAD_EXACT_TOKENS = {
    # old validation failures
    "shaxt", "ananalar", "yeki", "dadamiing", "meditsinada", "dorivorlar",

    # repeated preview failures
    "uulki", "utkaziladi", "qilmaymen", "narsanidim", "naytida", "oralikpi",
    "kassalanga", "savoljavob", "chiqqan", "net", "devdim", "xam",
    "tabiy", "tomenidan", "mablag", "quron", "maksadga", "yuli",
    "alahsiymiz", "sotiboldi",

    # common OCR / corpus-damage exact fragments
    "okmomsafu", "yanm", "patnuslarni", "komnati", "prekrasnie", "ved",
}

BAD_SUBSTRINGS = {
    # merged/broken phrases seen in previews
    "markazidauniversitet",
    "gapso'z",
    "gapsoz",
    "tilme'yornutq",
    "so'roqsurishtirish",
    "grekbaqtriya",
    "konsignatoreksporter",
    "gajdumbek",
    "ishlabchiqarish",
    "to'qimapo'kak",
    "bo'g'imchanoq",
    "asbobtermometr",
    "qismioyog",
    "sharqiysharqiy",
    "qahramonlarning",
    "buqahramon",
    "kuchlanishbir",
    "qo'llabquvvatlab",
    "yuzmayuz",
    "teztez",
    "edii",
    "ediu",
    "lardavriy",
    "xattotrassom",
    "masalabola",
    "qim mat",
    "olti oma",
    "t kursiy",
    "havas chni",
    "za oyat",
}

FOREIGN_MARKERS = {
    "cauda", "nas", "ved", "prekrasnie", "komnati", "been", "real",
    "html", "display", "kursiy", "ruf", "shtyox",
}

ARCHAIC_LITERARY_MARKERS = {
    "birla", "erdi", "mavlo", "taolo", "g'ulom", "hazrat", "sohibkiron",
    "aqd", "yorliqamish", "arig", "shahvat",
}

TECHNICAL_DOMAIN_MARKERS = {
    "biotsenoz", "sferik", "aberratsiya", "linza", "elektrolit", "gipoteza",
    "katalizator", "reaksion", "parallelepiped", "tsellyuloza", "bikarbonat",
    "bufer", "protoplast", "injenerligi", "poligon", "atom", "zaryad",
    "o'tkazgich", "termometr", "kambiy", "po'kak", "konsignatsiya",
    "inkassatsiya", "tarozilar", "sterjnida", "plastinka", "qalay",
    "muskullari", "surg'ich", "ionlanishi", "melanj",
}

ALLOWED_SINGLE_LETTERS = {"u", "o"}


def norm_apostrophe(text: str) -> str:
    return (
        str(text)
        .replace("ʻ", "'")
        .replace("ʼ", "'")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("`", "'")
    )


def parse_listish(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s or s == "[]":
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in re.split(r"[|,;]", s) if x.strip()]


WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+|\d+")


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(norm_apostrophe(text).lower())]


def tokens(text: str) -> list[str]:
    return [w.lower() for w in TOKEN_RE.findall(norm_apostrophe(text).lower())]


def sentence_source_from_id(sentence_id: str) -> str:
    m = re.match(r"^s3v1-(\d+)-", str(sentence_id))
    return m.group(1) if m else ""


def repeated_segment_token(token: str) -> bool:
    t = token.lower().replace("'", "")
    if len(t) < 10:
        return False
    for size in range(3, len(t) // 2 + 1):
        if len(t) % size == 0:
            piece = t[:size]
            if piece * (len(t) // size) == t:
                return True
        if t[:size] == t[size: size * 2]:
            return True
    return False


def alpha_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    alpha = sum(c.isalpha() for c in chars)
    return alpha / len(chars)


def punctuation_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    punct = sum((not c.isalnum()) and (not c.isspace()) for c in chars)
    return punct / len(chars)


def count_upper_acronyms(text: str) -> int:
    return len(re.findall(r"\b[A-Z]{2,}\b", text))


def has_midword_case_merge(text: str) -> bool:
    return bool(re.search(r"[a-zа-яё][A-ZА-ЯЁ]", text))


def has_spaced_letter_garbage(text: str) -> bool:
    # catches patterns like: S ob ir ak a
    return bool(re.search(r"\b(?:[A-Za-z]\s+){3,}[A-Za-z]\b", text))


def has_malformed_apostrophe(text: str) -> bool:
    s = norm_apostrophe(text)
    if "''" in s:
        return True
    if re.search(r"\b'\w+|\w+'\b", s):
        return True
    return False


def sentence_risks(row: dict) -> tuple[list[str], list[str], dict]:
    text = str(row.get("sentence_text", ""))
    clean_text = norm_apostrophe(text)
    low = clean_text.lower()
    ws = words(clean_text)
    ts = tokens(clean_text)

    severe: list[str] = []
    mild: list[str] = []

    boundary_flags = parse_listish(row.get("boundary_flags"))
    parent_flags = parse_listish(row.get("parent_review_flags"))

    if boundary_flags:
        severe.append("has_boundary_flags")
    if parent_flags:
        severe.append("has_parent_flags")

    if not clean_text.strip():
        severe.append("empty_text")
    if clean_text.strip() and clean_text.strip()[0].islower():
        severe.append("starts_lowercase")

    if re.search(r"[\u0400-\u04FF]", clean_text):
        severe.append("cyrillic_script")

    if has_spaced_letter_garbage(clean_text):
        severe.append("spaced_letter_garbage")

    if has_midword_case_merge(clean_text):
        severe.append("midword_case_merge")

    if has_malformed_apostrophe(clean_text):
        severe.append("malformed_apostrophe")

    if alpha_ratio(clean_text) < 0.82:
        severe.append("low_alpha_ratio")

    if punctuation_ratio(clean_text) > 0.085:
        severe.append("punctuation_heavy")

    if re.search(r"[:;]", clean_text):
        mild.append("colon_or_semicolon")

    if clean_text.count(",") > 1:
        mild.append("too_many_commas")

    if count_upper_acronyms(clean_text) > 0:
        mild.append("uppercase_acronym")

    if any(x in low for x in BAD_SUBSTRINGS):
        for x in sorted(BAD_SUBSTRINGS):
            if x in low:
                severe.append(f"bad_substring:{x}")
                break

    for t in ts:
        if t in BAD_EXACT_TOKENS:
            severe.append(f"bad_exact:{t}")
        if t in FOREIGN_MARKERS:
            severe.append(f"foreign_marker:{t}")
        if t in ARCHAIC_LITERARY_MARKERS:
            mild.append(f"archaic_literary:{t}")
        if t in TECHNICAL_DOMAIN_MARKERS:
            mild.append(f"technical_domain:{t}")
        if t in ORDINAL_RISK:
            mild.append(f"ordinal_risk:{t}")
        if len(t) == 1 and t not in ALLOWED_SINGLE_LETTERS and t.isalpha():
            severe.append(f"single_letter_token:{t}")
        if len(t) >= 22 and t.isalpha():
            severe.append(f"very_long_token:{t[:30]}")
        if repeated_segment_token(t):
            severe.append(f"repeated_segment:{t[:30]}")
        if re.search(r"(.)\1\1", t):
            severe.append(f"triple_repeated_char:{t[:30]}")

    number_word_count = sum(1 for t in ts if t in NUMBER_WORDS)
    if number_word_count >= 2:
        mild.append("number_word_heavy")

    anchor_count = sum(1 for t in ts if t in COMMON_UZ_ANCHORS)

    # extra broad garbage shape: many unknown-looking single/small pieces
    single_letter_count = sum(1 for t in ts if len(t) == 1 and t.isalpha())
    if single_letter_count >= 2:
        severe.append("many_single_letter_tokens")

    meta = {
        "word_count": len(ws),
        "token_count": len(ts),
        "char_count": len(clean_text),
        "anchor_count": anchor_count,
        "number_word_count": number_word_count,
        "comma_count": clean_text.count(","),
        "upper_acronym_count": count_upper_acronyms(clean_text),
        "alpha_ratio_calc": round(alpha_ratio(clean_text), 5),
        "punctuation_ratio_calc": round(punctuation_ratio(clean_text), 5),
        "terminal_period": clean_text.rstrip().endswith("."),
    }

    severe = sorted(set(severe))
    mild = sorted(set(mild))
    return severe, mild, meta


def is_simple_modern_sentence(row: dict, severe: list[str], mild: list[str], meta: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if severe:
        reasons.append("has_severe_sentence_risk")
    if mild:
        reasons.append("has_mild_sentence_risk")

    if not meta["terminal_period"]:
        reasons.append("not_terminal_period")

    if meta["word_count"] < 7:
        reasons.append("too_few_words")
    if meta["word_count"] > 18:
        reasons.append("too_many_words")

    if meta["char_count"] < 45:
        reasons.append("too_short_chars")
    if meta["char_count"] > 170:
        reasons.append("too_long_chars")

    if meta["anchor_count"] < 3:
        reasons.append("low_common_uzbek_anchor_count")

    if meta["comma_count"] > 1:
        reasons.append("too_many_commas")

    if meta["number_word_count"] > 1:
        reasons.append("number_word_heavy")

    if meta["upper_acronym_count"] > 0:
        reasons.append("uppercase_acronym")

    return (len(reasons) == 0), reasons


def read_pool() -> pd.DataFrame:
    rows = []
    with POOL.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)

    df = pd.DataFrame(rows)
    df["source_record_id"] = df["source_record_id"].astype(str)
    df["sentence_id"] = df["sentence_id"].astype(str)
    df["sentence_text"] = df["sentence_text"].astype(str)
    return df


def read_excluded_sources() -> tuple[set[str], set[str], Counter]:
    excluded_sources: set[str] = set()
    excluded_sentence_ids: set[str] = set()
    counts = Counter()

    old = pd.read_csv(OLD_MASTER, dtype=str).fillna("")
    old_sources = set(old["source_record_id"].astype(str))
    old_sids = set(old["sentence_id"].astype(str))
    excluded_sources |= old_sources
    excluded_sentence_ids |= old_sids
    counts["old_reviewed_source_record_ids"] = len(old_sources)
    counts["old_reviewed_sentence_ids"] = len(old_sids)

    opened_source_count = 0
    opened_sid_count = 0

    for path in OPENED_PREVIEWS:
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if "sentence_id" in df.columns:
            sids = set(df["sentence_id"].astype(str))
            excluded_sentence_ids |= sids
            opened_sid_count += len(sids)

        if "source_record_id" in df.columns:
            srcs = set(df["source_record_id"].astype(str))
        elif "sentence_id" in df.columns:
            srcs = {
                sentence_source_from_id(x)
                for x in df["sentence_id"].astype(str)
                if sentence_source_from_id(x)
            }
        else:
            srcs = set()

        excluded_sources |= srcs
        opened_source_count += len(srcs)

    counts["opened_preview_source_record_ids_raw_sum"] = opened_source_count
    counts["opened_preview_sentence_ids_raw_sum"] = opened_sid_count
    counts["total_excluded_source_record_ids"] = len(excluded_sources)
    counts["total_excluded_sentence_ids"] = len(excluded_sentence_ids)

    return excluded_sources, excluded_sentence_ids, counts


def main() -> None:
    if not POOL.exists():
        raise FileNotFoundError(POOL)
    if not OLD_MASTER.exists():
        raise FileNotFoundError(OLD_MASTER)

    df = read_pool()
    excluded_sources, excluded_sentence_ids, exclusion_counts = read_excluded_sources()

    df["excluded_old_or_opened_source"] = df["source_record_id"].isin(excluded_sources)
    df["excluded_old_or_opened_sentence_id"] = df["sentence_id"].isin(excluded_sentence_ids)

    fresh = df[
        ~df["excluded_old_or_opened_source"]
        & ~df["excluded_old_or_opened_sentence_id"]
    ].copy()

    severe_col = []
    mild_col = []
    candidate_reason_col = []
    meta_rows = []
    candidate_flags = []

    sentence_severe_counter = Counter()
    sentence_mild_counter = Counter()
    candidate_block_counter = Counter()

    for row in fresh.to_dict("records"):
        severe, mild, meta = sentence_risks(row)
        is_candidate, candidate_reasons = is_simple_modern_sentence(row, severe, mild, meta)

        severe_col.append("|".join(severe))
        mild_col.append("|".join(mild))
        candidate_reason_col.append("|".join(candidate_reasons))
        meta_rows.append(meta)
        candidate_flags.append(is_candidate)

        sentence_severe_counter.update(severe)
        sentence_mild_counter.update(mild)
        candidate_block_counter.update(candidate_reasons)

    meta_df = pd.DataFrame(meta_rows)
    fresh = pd.concat([fresh.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)
    fresh["sentence_severe_reasons_v1_4"] = severe_col
    fresh["sentence_mild_reasons_v1_4"] = mild_col
    fresh["candidate_block_reasons_v1_4"] = candidate_reason_col
    fresh["simple_modern_sentence_candidate_v1_4"] = candidate_flags

    fresh["has_severe_sentence_risk"] = fresh["sentence_severe_reasons_v1_4"].ne("")
    fresh["has_mild_sentence_risk"] = fresh["sentence_mild_reasons_v1_4"].ne("")

    doc_rows = []
    doc_reject_counter = Counter()

    for source_record_id, g in fresh.groupby("source_record_id", sort=False):
        total_sentences = len(g)
        severe_count = int(g["has_severe_sentence_risk"].sum())
        mild_count = int(g["has_mild_sentence_risk"].sum())
        candidate_count = int(g["simple_modern_sentence_candidate_v1_4"].sum())

        boundary_count = sum(bool(parse_listish(x)) for x in g["boundary_flags"])
        parent_count = sum(bool(parse_listish(x)) for x in g["parent_review_flags"])

        all_severe = "|".join(g["sentence_severe_reasons_v1_4"].tolist())
        all_mild = "|".join(g["sentence_mild_reasons_v1_4"].tolist())

        foreign_count = all_severe.count("foreign_marker:")
        bad_exact_count = all_severe.count("bad_exact:")
        bad_substring_count = all_severe.count("bad_substring:")
        domain_count = all_mild.count("technical_domain:")
        archaic_count = all_mild.count("archaic_literary:")

        reasons = []

        if total_sentences < 2:
            reasons.append("doc_too_few_sentences")
        if boundary_count > 0:
            reasons.append("doc_has_boundary_flags")
        if parent_count > 0:
            reasons.append("doc_has_parent_flags")
        if severe_count > 0:
            reasons.append("doc_has_any_severe_sentence_risk")
        if foreign_count > 0:
            reasons.append("doc_has_foreign_marker")
        if bad_exact_count > 0:
            reasons.append("doc_has_bad_exact_token")
        if bad_substring_count > 0:
            reasons.append("doc_has_bad_substring")
        if domain_count > 0:
            reasons.append("doc_has_technical_domain_marker")
        if archaic_count > 0:
            reasons.append("doc_has_archaic_literary_marker")
        if candidate_count == 0:
            reasons.append("doc_has_zero_simple_candidates")
        if total_sentences > 0 and candidate_count / total_sentences < 0.25:
            reasons.append("doc_low_candidate_rate")

        doc_reject_counter.update(reasons)

        doc_rows.append(
            {
                "source_record_id": source_record_id,
                "part_number": g["part_number"].iloc[0],
                "document_sample_id": g["document_sample_id"].iloc[0],
                "total_sentences": total_sentences,
                "severe_sentence_count": severe_count,
                "mild_sentence_count": mild_count,
                "simple_candidate_sentence_count": candidate_count,
                "candidate_rate": round(candidate_count / total_sentences, 5) if total_sentences else 0,
                "boundary_flagged_sentence_count": boundary_count,
                "parent_flagged_sentence_count": parent_count,
                "foreign_marker_count": foreign_count,
                "bad_exact_count": bad_exact_count,
                "bad_substring_count": bad_substring_count,
                "technical_domain_marker_count": domain_count,
                "archaic_literary_marker_count": archaic_count,
                "document_safe_v1_4": len(reasons) == 0,
                "document_reject_reasons_v1_4": "|".join(sorted(set(reasons))),
                "preview_first_sentence": g["sentence_text"].iloc[0],
            }
        )

    doc_scores = pd.DataFrame(doc_rows)
    safe_docs = doc_scores[doc_scores["document_safe_v1_4"]].copy()
    rejected_docs = doc_scores[~doc_scores["document_safe_v1_4"]].copy()

    safe_source_ids = set(safe_docs["source_record_id"].astype(str))

    candidates = fresh[
        fresh["source_record_id"].isin(safe_source_ids)
        & fresh["simple_modern_sentence_candidate_v1_4"].astype(bool)
    ].copy()

    candidates["normalized_sentence_text_for_dedup"] = (
        candidates["sentence_text"]
        .map(norm_apostrophe)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    candidates = candidates.drop_duplicates(
        subset=["normalized_sentence_text_for_dedup"],
        keep="first",
    ).copy()

    # one sentence per source document for preview/manual review
    if len(candidates):
        one_per_doc = (
            candidates
            .sample(frac=1, random_state=RANDOM_SEED)
            .groupby("source_record_id", as_index=False, sort=False)
            .head(1)
        )
        review = one_per_doc.sample(
            n=min(SAMPLE_SIZE, len(one_per_doc)),
            random_state=RANDOM_SEED,
        ).copy()
    else:
        review = candidates.copy()

    review.insert(0, "manual_clean_label", "")
    review.insert(1, "manual_notes", "")

    keep_review_cols = [
        "manual_clean_label",
        "manual_notes",
        "source_record_id",
        "document_sample_id",
        "part_number",
        "sentence_id",
        "word_count",
        "char_count",
        "anchor_count",
        "sentence_text",
        "sentence_severe_reasons_v1_4",
        "sentence_mild_reasons_v1_4",
        "candidate_block_reasons_v1_4",
    ]
    review = review[[c for c in keep_review_cols if c in review.columns]].copy()

    fresh_path = OUT / "fresh_sentences_scored_v1_4.csv"
    doc_path = OUT / "document_scores_v1_4.csv"
    safe_doc_path = OUT / "safe_documents_v1_4.csv"
    rejected_doc_path = OUT / "rejected_documents_v1_4.csv"
    cand_path = OUT / "document_safe_sentence_candidates_v1_4.csv"
    review_path = OUT / "manual_review_document_safe_200_v1_4.csv"
    summary_path = OUT / "document_safe_clean_seed_summary_v1_4.md"

    fresh.to_csv(fresh_path, index=False)
    doc_scores.to_csv(doc_path, index=False)
    safe_docs.to_csv(safe_doc_path, index=False)
    rejected_docs.to_csv(rejected_doc_path, index=False)
    candidates.to_csv(cand_path, index=False)
    review.to_csv(review_path, index=False)

    part_counts = candidates["part_number"].astype(str).value_counts().head(30)

    lines = []
    lines.append("# Document-Safe Clean Seed Extractor v1.4")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("Development preview. This is not validated until the preview/manual review passes.")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- pool_total_sentences: `{len(df)}`")
    lines.append(f"- old_reviewed_source_record_ids: `{exclusion_counts['old_reviewed_source_record_ids']}`")
    lines.append(f"- opened_preview_source_record_ids_raw_sum: `{exclusion_counts['opened_preview_source_record_ids_raw_sum']}`")
    lines.append(f"- total_excluded_source_record_ids: `{exclusion_counts['total_excluded_source_record_ids']}`")
    lines.append(f"- fresh_sentences_after_exclusion: `{len(fresh)}`")
    lines.append(f"- fresh_source_documents_after_exclusion: `{fresh['source_record_id'].nunique()}`")
    lines.append(f"- safe_documents_v1_4: `{len(safe_docs)}`")
    lines.append(f"- rejected_documents_v1_4: `{len(rejected_docs)}`")
    lines.append(f"- document_safe_sentence_candidates_v1_4: `{len(candidates)}`")
    lines.append(f"- review_preview_rows: `{len(review)}`")
    lines.append("")
    lines.append("## Top document reject reasons")
    lines.append("")
    for k, v in doc_reject_counter.most_common(60):
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Top sentence severe risks")
    lines.append("")
    for k, v in sentence_severe_counter.most_common(60):
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Top sentence mild risks")
    lines.append("")
    for k, v in sentence_mild_counter.most_common(60):
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Top candidate block reasons")
    lines.append("")
    for k, v in candidate_block_counter.most_common(60):
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Candidate distribution by part")
    lines.append("")
    for part, count in part_counts.items():
        lines.append(f"- part {part}: `{count}`")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- fresh sentence scores: `{fresh_path}`")
    lines.append(f"- document scores: `{doc_path}`")
    lines.append(f"- safe documents: `{safe_doc_path}`")
    lines.append(f"- rejected documents: `{rejected_doc_path}`")
    lines.append(f"- candidates: `{cand_path}`")
    lines.append(f"- review preview: `{review_path}`")
    lines.append("")
    lines.append("## Decision rule")
    lines.append("")
    lines.append("- First inspect the preview.")
    lines.append("- Do not label if preview is visibly contaminated.")
    lines.append("- If preview looks clean, label `manual_clean_label` as `CLEAN_TARGET` or `NOT_CLEAN_TARGET`.")
    lines.append("- If manual precision >= 95%, v1.4 can be considered fresh-preview passed.")
    lines.append("- If manual precision < 95%, do not use v1.4 for corpus-wide clean target extraction.")

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))

    print("\nFIRST 100 DOCUMENT-SAFE REVIEW PREVIEW ROWS")
    for i, row in review.head(100).reset_index(drop=True).iterrows():
        print("\n" + "=" * 120)
        print("ROW:", i + 1)
        print("source_record_id:", row.get("source_record_id", ""))
        print("part:", row.get("part_number", ""))
        print("sentence_id:", row.get("sentence_id", ""))
        print("words:", row.get("word_count", ""))
        print("anchors:", row.get("anchor_count", ""))
        print("text:", row.get("sentence_text", ""))


if __name__ == "__main__":
    main()
