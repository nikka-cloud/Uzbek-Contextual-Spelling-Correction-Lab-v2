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

OUT = BASE / "document_safe_clean_seed_v1_4a"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 200
RANDOM_SEED = 2026061805


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
    "qilish", "etish", "olish", "berish", "ko'rish",
    "amalga", "oshirish", "asosida", "maqsadida",
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
    "shaxt", "ananalar", "yeki", "dadamiing", "meditsinada", "dorivorlar",
    "uulki", "utkaziladi", "qilmaymen", "narsanidim", "naytida", "oralikpi",
    "kassalanga", "chiqqan", "net", "devdim", "xam", "tabiy", "tomenidan",
    "mablag", "quron", "maksadga", "yuli", "alahsiymiz", "sotiboldi",
    "okmomsafu", "yanm", "patnuslarni", "komnati", "prekrasnie", "ved",
}

BAD_SUBSTRINGS = {
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
    "birla", "erdi", "mavlo", "taolo", "g'ulom", "aqd", "yorliqamish", "arig",
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


def alpha_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(c.isalpha() for c in chars) / len(chars)


def punctuation_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum((not c.isalnum()) and (not c.isspace()) for c in chars) / len(chars)


def has_valid_or_invalid_apostrophe_shape(text: str) -> tuple[bool, list[str]]:
    """
    Valid Uzbek apostrophe examples:
    o'z, bo'lgan, ma'lum, g'arb, ta'lim, ko'ra.

    Invalid examples:
    'word, word', o''z, o' z, o 'z.
    """
    s = norm_apostrophe(text)
    reasons = []

    if "''" in s:
        reasons.append("double_apostrophe")

    if re.search(r"(^|\s)'[A-Za-zÀ-ÖØ-öø-ÿ]", s):
        reasons.append("leading_apostrophe_token")

    if re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]'(\s|$)", s):
        reasons.append("trailing_apostrophe_token")

    if re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]\s+'\s*[A-Za-zÀ-ÖØ-öø-ÿ]", s):
        reasons.append("space_before_apostrophe")

    if re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]'\s+[A-Za-zÀ-ÖØ-öø-ÿ]", s):
        reasons.append("space_after_apostrophe")

    return len(reasons) > 0, reasons


def repeated_segment_token(token: str) -> bool:
    t = token.lower().replace("'", "")
    if len(t) < 11:
        return False

    safe_redup = {
        "birbiriga", "birbirini", "birbiridan", "birbirining",
        "qaytaqayta", "vaqtivaqti", "alohidaalohida",
    }
    if t in safe_redup:
        return False

    for size in range(3, len(t) // 2 + 1):
        if len(t) % size == 0:
            piece = t[:size]
            if piece * (len(t) // size) == t:
                return True
        if t[:size] == t[size: size * 2]:
            return True
    return False


def has_midword_case_merge(text: str) -> bool:
    # catches GrekBaqtriya, UniversitetBulvari, etc.
    # does not catch AQShda because that is upper-upper-lower, not lower-upper.
    return bool(re.search(r"[a-zа-яё][A-ZА-ЯЁ]", text))


def has_spaced_letter_garbage(text: str) -> bool:
    return bool(re.search(r"\b(?:[A-Za-z]\s+){3,}[A-Za-z]\b", text))


def count_upper_acronyms(text: str) -> int:
    return len(re.findall(r"\b[A-Z]{2,}\b", text))


def read_pool() -> pd.DataFrame:
    rows = []
    with POOL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    df["source_record_id"] = df["source_record_id"].astype(str)
    df["sentence_id"] = df["sentence_id"].astype(str)
    df["sentence_text"] = df["sentence_text"].astype(str)
    return df


def read_exclusions() -> tuple[set[str], set[str], Counter]:
    excluded_sources = set()
    excluded_sentence_ids = set()
    counts = Counter()

    old = pd.read_csv(OLD_MASTER, dtype=str).fillna("")
    old_sources = set(old["source_record_id"].astype(str))
    old_sids = set(old["sentence_id"].astype(str))

    excluded_sources |= old_sources
    excluded_sentence_ids |= old_sids

    counts["old_reviewed_source_record_ids"] = len(old_sources)
    counts["old_reviewed_sentence_ids"] = len(old_sids)

    opened_source_raw_sum = 0
    opened_sid_raw_sum = 0

    for path in OPENED_PREVIEWS:
        if not path.exists():
            continue

        df = pd.read_csv(path, dtype=str).fillna("")

        if "sentence_id" in df.columns:
            sids = set(df["sentence_id"].astype(str))
            excluded_sentence_ids |= sids
            opened_sid_raw_sum += len(sids)

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
        opened_source_raw_sum += len(srcs)

    counts["opened_preview_source_record_ids_raw_sum"] = opened_source_raw_sum
    counts["opened_preview_sentence_ids_raw_sum"] = opened_sid_raw_sum
    counts["total_excluded_source_record_ids"] = len(excluded_sources)
    counts["total_excluded_sentence_ids"] = len(excluded_sentence_ids)

    return excluded_sources, excluded_sentence_ids, counts


def analyze_sentence(row: dict) -> tuple[list[str], list[str], list[str], dict]:
    text = str(row.get("sentence_text", ""))
    clean = norm_apostrophe(text)
    low = clean.lower()
    ws = words(clean)
    ts = tokens(clean)

    doc_risks = []
    sent_risks = []
    mild_risks = []

    boundary_flags = parse_listish(row.get("boundary_flags"))
    parent_flags = parse_listish(row.get("parent_review_flags"))

    if parent_flags:
        doc_risks.append("parent_review_flags")

    if boundary_flags:
        sent_risks.append("boundary_flags")

    if re.search(r"[\u0400-\u04FF]", clean):
        doc_risks.append("cyrillic_script")

    bad_apostrophe, apostrophe_reasons = has_valid_or_invalid_apostrophe_shape(clean)
    if bad_apostrophe:
        doc_risks.extend([f"bad_apostrophe:{r}" for r in apostrophe_reasons])

    if has_spaced_letter_garbage(clean):
        doc_risks.append("spaced_letter_garbage")

    if has_midword_case_merge(clean):
        doc_risks.append("midword_case_merge")

    if alpha_ratio(clean) < 0.82:
        sent_risks.append("low_alpha_ratio")

    if punctuation_ratio(clean) > 0.09:
        sent_risks.append("punctuation_heavy")

    if clean.strip() and clean.strip()[0].islower():
        sent_risks.append("starts_lowercase")

    if re.search(r"[:;!?]", clean):
        sent_risks.append("non_plain_terminal_or_strong_punct")

    for bad in sorted(BAD_SUBSTRINGS):
        if bad in low:
            doc_risks.append(f"bad_substring:{bad}")

    for t in ts:
        if t in BAD_EXACT_TOKENS:
            doc_risks.append(f"bad_exact:{t}")

        if t in FOREIGN_MARKERS:
            doc_risks.append(f"foreign_marker:{t}")

        if t in ARCHAIC_LITERARY_MARKERS:
            doc_risks.append(f"archaic_literary:{t}")

        if t in TECHNICAL_DOMAIN_MARKERS:
            doc_risks.append(f"technical_domain:{t}")

        if t in ORDINAL_RISK:
            mild_risks.append(f"ordinal_risk:{t}")

        if len(t) == 1 and t not in ALLOWED_SINGLE_LETTERS and t.isalpha():
            sent_risks.append(f"single_letter_token:{t}")

        if len(t) >= 22 and t.isalpha():
            doc_risks.append(f"very_long_token:{t[:30]}")

        if repeated_segment_token(t):
            doc_risks.append(f"repeated_segment:{t[:30]}")

        if re.search(r"(.)\1\1", t):
            doc_risks.append(f"triple_repeated_char:{t[:30]}")

    number_word_count = sum(1 for t in ts if t in NUMBER_WORDS)
    anchor_count = sum(1 for t in ts if t in COMMON_UZ_ANCHORS)

    if number_word_count >= 2:
        mild_risks.append("number_word_heavy")

    if clean.count(",") > 1:
        mild_risks.append("too_many_commas")

    if count_upper_acronyms(clean) > 0:
        mild_risks.append("uppercase_acronym")

    meta = {
        "word_count": len(ws),
        "token_count": len(ts),
        "char_count": len(clean),
        "anchor_count": anchor_count,
        "number_word_count": number_word_count,
        "comma_count": clean.count(","),
        "upper_acronym_count": count_upper_acronyms(clean),
        "alpha_ratio_calc": round(alpha_ratio(clean), 5),
        "punctuation_ratio_calc": round(punctuation_ratio(clean), 5),
        "terminal_period": clean.rstrip().endswith("."),
    }

    return sorted(set(doc_risks)), sorted(set(sent_risks)), sorted(set(mild_risks)), meta


def candidate_decision(row: dict, doc_risks: list[str], sent_risks: list[str], mild_risks: list[str], meta: dict) -> tuple[bool, list[str]]:
    reasons = []

    if doc_risks:
        reasons.append("has_doc_level_risk")
    if sent_risks:
        reasons.append("has_sentence_risk")
    if mild_risks:
        reasons.append("has_mild_risk")

    if not meta["terminal_period"]:
        reasons.append("not_terminal_period")

    if meta["word_count"] < 7:
        reasons.append("too_few_words")
    if meta["word_count"] > 18:
        reasons.append("too_many_words")

    if meta["char_count"] < 45:
        reasons.append("too_short_chars")
    if meta["char_count"] > 165:
        reasons.append("too_long_chars")

    if meta["anchor_count"] < 3:
        reasons.append("low_common_uzbek_anchor_count")

    if meta["comma_count"] > 1:
        reasons.append("too_many_commas")

    if meta["number_word_count"] > 1:
        reasons.append("number_word_heavy")

    if meta["upper_acronym_count"] > 0:
        reasons.append("uppercase_acronym")

    return len(reasons) == 0, reasons


def main() -> None:
    df = read_pool()
    excluded_sources, excluded_sentence_ids, exclusion_counts = read_exclusions()

    df["excluded_source_record_id"] = df["source_record_id"].isin(excluded_sources)
    df["excluded_sentence_id"] = df["sentence_id"].isin(excluded_sentence_ids)

    fresh = df[
        ~df["excluded_source_record_id"]
        & ~df["excluded_sentence_id"]
    ].copy()

    doc_risk_col = []
    sent_risk_col = []
    mild_risk_col = []
    cand_reason_col = []
    candidate_flags = []
    meta_rows = []

    doc_risk_counter = Counter()
    sent_risk_counter = Counter()
    mild_risk_counter = Counter()
    candidate_block_counter = Counter()

    for row in fresh.to_dict("records"):
        doc_risks, sent_risks, mild_risks, meta = analyze_sentence(row)
        is_candidate, cand_reasons = candidate_decision(row, doc_risks, sent_risks, mild_risks, meta)

        doc_risk_col.append("|".join(doc_risks))
        sent_risk_col.append("|".join(sent_risks))
        mild_risk_col.append("|".join(mild_risks))
        cand_reason_col.append("|".join(cand_reasons))
        candidate_flags.append(is_candidate)
        meta_rows.append(meta)

        doc_risk_counter.update(doc_risks)
        sent_risk_counter.update(sent_risks)
        mild_risk_counter.update(mild_risks)
        candidate_block_counter.update(cand_reasons)

    fresh = pd.concat(
        [fresh.reset_index(drop=True), pd.DataFrame(meta_rows).reset_index(drop=True)],
        axis=1,
    )

    fresh["doc_level_risks_v1_4a"] = doc_risk_col
    fresh["sentence_risks_v1_4a"] = sent_risk_col
    fresh["mild_risks_v1_4a"] = mild_risk_col
    fresh["candidate_block_reasons_v1_4a"] = cand_reason_col
    fresh["simple_modern_candidate_v1_4a"] = candidate_flags

    fresh["has_doc_level_risk"] = fresh["doc_level_risks_v1_4a"].ne("")
    fresh["has_sentence_risk"] = fresh["sentence_risks_v1_4a"].ne("")
    fresh["has_mild_risk"] = fresh["mild_risks_v1_4a"].ne("")

    doc_rows = []
    doc_reject_counter = Counter()

    for source_record_id, g in fresh.groupby("source_record_id", sort=False):
        total_sentences = len(g)
        doc_risk_count = int(g["has_doc_level_risk"].sum())
        boundary_sentence_count = int(g["sentence_risks_v1_4a"].str.contains("boundary_flags", regex=False).sum())
        parent_flag_count = int(g["doc_level_risks_v1_4a"].str.contains("parent_review_flags", regex=False).sum())
        candidate_count = int(g["simple_modern_candidate_v1_4a"].sum())

        reasons = []

        # strict document safety: any real garbage/domain/foreign/OCR marker rejects the whole document
        if doc_risk_count > 0:
            reasons.append("doc_has_doc_level_risk")

        if parent_flag_count > 0:
            reasons.append("doc_has_parent_flags")

        # boundary flags are segmentation-risk; allow zero only for safe document seed
        if boundary_sentence_count > 0:
            reasons.append("doc_has_boundary_flags")

        if total_sentences < 2:
            reasons.append("doc_too_few_sentences")

        if candidate_count == 0:
            reasons.append("doc_has_zero_simple_candidates")

        if total_sentences > 0 and candidate_count / total_sentences < 0.15:
            reasons.append("doc_low_candidate_rate")

        doc_reject_counter.update(reasons)

        doc_rows.append(
            {
                "source_record_id": source_record_id,
                "part_number": g["part_number"].iloc[0],
                "document_sample_id": g["document_sample_id"].iloc[0],
                "total_sentences": total_sentences,
                "doc_level_risk_sentence_count": doc_risk_count,
                "boundary_sentence_count": boundary_sentence_count,
                "parent_flag_sentence_count": parent_flag_count,
                "simple_candidate_sentence_count": candidate_count,
                "candidate_rate": round(candidate_count / total_sentences, 5) if total_sentences else 0,
                "document_safe_v1_4a": len(reasons) == 0,
                "document_reject_reasons_v1_4a": "|".join(sorted(set(reasons))),
                "preview_first_sentence": g["sentence_text"].iloc[0],
            }
        )

    doc_scores = pd.DataFrame(doc_rows)
    safe_docs = doc_scores[doc_scores["document_safe_v1_4a"]].copy()
    rejected_docs = doc_scores[~doc_scores["document_safe_v1_4a"]].copy()

    safe_sources = set(safe_docs["source_record_id"].astype(str))

    candidates = fresh[
        fresh["source_record_id"].isin(safe_sources)
        & fresh["simple_modern_candidate_v1_4a"].astype(bool)
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

    review_cols = [
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
        "doc_level_risks_v1_4a",
        "sentence_risks_v1_4a",
        "mild_risks_v1_4a",
        "candidate_block_reasons_v1_4a",
    ]

    review = review[[c for c in review_cols if c in review.columns]].copy()

    fresh_path = OUT / "fresh_sentences_scored_v1_4a.csv"
    doc_path = OUT / "document_scores_v1_4a.csv"
    safe_doc_path = OUT / "safe_documents_v1_4a.csv"
    rejected_doc_path = OUT / "rejected_documents_v1_4a.csv"
    cand_path = OUT / "document_safe_sentence_candidates_v1_4a.csv"
    review_path = OUT / "manual_review_document_safe_200_v1_4a.csv"
    summary_path = OUT / "document_safe_clean_seed_summary_v1_4a.md"

    fresh.to_csv(fresh_path, index=False)
    doc_scores.to_csv(doc_path, index=False)
    safe_docs.to_csv(safe_doc_path, index=False)
    rejected_docs.to_csv(rejected_doc_path, index=False)
    candidates.to_csv(cand_path, index=False)
    review.to_csv(review_path, index=False)

    part_counts = candidates["part_number"].astype(str).value_counts().head(30) if len(candidates) else pd.Series(dtype=int)

    lines = []
    lines.append("# Document-Safe Clean Seed Extractor v1.4a")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("Development preview. v1.4a fixes the over-broad apostrophe rejection from v1.4.")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- pool_total_sentences: `{len(df)}`")
    lines.append(f"- old_reviewed_source_record_ids: `{exclusion_counts['old_reviewed_source_record_ids']}`")
    lines.append(f"- opened_preview_source_record_ids_raw_sum: `{exclusion_counts['opened_preview_source_record_ids_raw_sum']}`")
    lines.append(f"- total_excluded_source_record_ids: `{exclusion_counts['total_excluded_source_record_ids']}`")
    lines.append(f"- fresh_sentences_after_exclusion: `{len(fresh)}`")
    lines.append(f"- fresh_source_documents_after_exclusion: `{fresh['source_record_id'].nunique()}`")
    lines.append(f"- safe_documents_v1_4a: `{len(safe_docs)}`")
    lines.append(f"- rejected_documents_v1_4a: `{len(rejected_docs)}`")
    lines.append(f"- document_safe_sentence_candidates_v1_4a: `{len(candidates)}`")
    lines.append(f"- review_preview_rows: `{len(review)}`")
    lines.append("")
    lines.append("## Top document reject reasons")
    lines.append("")
    for k, v in doc_reject_counter.most_common(60):
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Top document-level sentence risks")
    lines.append("")
    for k, v in doc_risk_counter.most_common(60):
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Top sentence-only risks")
    lines.append("")
    for k, v in sent_risk_counter.most_common(60):
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Top mild risks")
    lines.append("")
    for k, v in mild_risk_counter.most_common(60):
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
    lines.append("- If manual precision >= 95%, v1.4a passes fresh candidate validation.")
    lines.append("- If manual precision < 95%, do not use v1.4a for corpus-wide clean target extraction.")

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
