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

IN_SCORED = BASE / "document_safe_clean_seed_v1_4b" / "fresh_sentences_scored_v1_4b.csv"

OPENED_PREVIEWS = [
    BASE / "document_safe_clean_seed_v1_4a" / "manual_review_document_safe_200_v1_4a.csv",
    BASE / "document_safe_clean_seed_v1_4b" / "manual_review_document_safe_200_v1_4b.csv",
]

OUT = BASE / "news_style_clean_seed_v1_4c"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 200
RANDOM_SEED = 2026061807


NEWS_ANCHORS = {
    "bu", "shu", "ushbu", "mazkur",
    "haqda", "xabar", "berdi", "ma'lum", "qildi",
    "ta'kidladi", "bildirdi", "aytdi",
    "toshkent", "o'zbekiston", "respublika",
    "vazirligi", "hokimligi", "agentligi", "xizmati",
    "matbuot", "rasmiy", "vakil",
    "joriy", "yil", "bugun", "kecha",
    "amalga", "oshirish", "uchun", "bilan", "bo'yicha",
    "tomonidan", "davomida", "natijada", "sababli",
}

BAD_EXACT_TOKENS = {
    "zavslanadi", "o'k", "usulitaktikasini", "otaonasi", "bordikeldini",
    "loyihasmeta", "jumlarning", "qo'shqanoti", "mrernet", "yaratatishda",
    "iazoratchisi", "yukoridagi", "orkali", "mantiliy", "kursatilgan",
    "tarkimondan", "taqazo", "o'zo'zimiz", "yozinqishin", "sharqiyjanubiy",
    "io'q", "sa'yharakatlarini", "qo'llabquvvatlashda", "uzilkesil",
    "havotomchi", "muxim", "guslubiy", "ko'ymokchi", "konsevtratordagi",
    "keraqli", "xujjatlar", "vidyabu", "rur", "apiltakil", "xam",
    "tabiy", "mablag", "quron", "yeki", "ananalar", "shaxt",
    "dadamiing", "utkaziladi", "tomenidan", "qilmaymen", "naytida",
    "oralikpi", "sotiboldi", "devdim", "net",
}

BAD_SUBSTRINGS = {
    "otaona",
    "loyihasmeta",
    "qo'shqanot",
    "sa'yharakat",
    "qo'llabquvvat",
    "uzilkesil",
    "havotomchi",
    "sharqiyjanub",
    "bordikeldi",
    "o'zo'z",
    "teztez",
    "yuzmayuz",
    "ishlabchiqarish",
    "qahramonlarning",
    "markazidauniversitet",
    "gapso'z",
    "so'roqsurishtirish",
    "asbobtermometr",
    "qismioyog",
    "tarixisiz",
    "tomirpomiri",
    "tushun gan",
    "noziklik lar",
    "obraz lar",
    "bola siga",
}

DOMAIN_OR_GENRE_RISK = {
    # literature / archaic / religious
    "sovxoz", "beku", "amir", "furqat", "majnun", "ishq", "tangri",
    "taolo", "shariat", "hadis", "tasavvuf", "muroqaba", "muhavvil",
    "doston", "g'azal", "muxammas", "fuzalo", "muhib",

    # medical / science / technical
    "xlamidioz", "anevrizma", "parametr", "grafik", "interval",
    "arteriya", "autizm", "kompressor", "fatseta", "yustirovka",
    "xokkey", "frontal", "kush", "ranjirovka", "trenirovka",
    "xlamidioz", "kasalliklar", "jismoniy yuklama",

    # foreign/specific names likely from fiction or noisy translation
    "panchon", "dans", "kitt", "buffalo", "komron",
}


WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+|\d+")


def norm(text: str) -> str:
    return (
        str(text)
        .replace("ʻ", "'")
        .replace("ʼ", "'")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("`", "'")
    )


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(norm(text).lower())]


def tokens(text: str) -> list[str]:
    return [w.lower() for w in TOKEN_RE.findall(norm(text).lower())]


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def local_blockers(text: str) -> list[str]:
    clean = norm(text)
    low = clean.lower()
    toks = tokens(clean)
    reasons = []

    for t in toks:
        if t in BAD_EXACT_TOKENS:
            reasons.append(f"bad_exact:{t}")

        if t in DOMAIN_OR_GENRE_RISK:
            reasons.append(f"domain_or_genre_risk:{t}")

        if len(t) >= 17 and t.isalpha():
            reasons.append(f"very_long_alpha_token:{t[:30]}")

        if re.search(r"(.)\1\1", t):
            reasons.append(f"triple_repeated_char:{t[:30]}")

    for sub in BAD_SUBSTRINGS:
        if sub in low:
            reasons.append(f"bad_substring:{sub}")

    if re.search(r"[a-z][A-Z]", clean):
        reasons.append("midword_case_merge")

    if re.search(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b", clean):
        reasons.append("spaced_letter_garbage")

    if re.search(
        r"\b[a-z']{4,}\s+(larning|larni|larini|larining|lariga|lardan|ganligi|ganligingiz|ganligini|moqda|dagi|siga)\b",
        low,
    ):
        reasons.append("split_suffix_like_corruption")

    if "," in clean:
        reasons.append("comma_present")

    if "?" in clean or "!" in clean or ":" in clean or ";" in clean:
        reasons.append("strong_or_non_plain_punctuation")

    return sorted(set(reasons))


def news_score(text: str) -> int:
    ws = set(words(text))
    return len(ws & NEWS_ANCHORS)


def main() -> None:
    if not IN_SCORED.exists():
        raise FileNotFoundError(IN_SCORED)

    df = pd.read_csv(IN_SCORED, dtype=str).fillna("")
    df["source_record_id"] = df["source_record_id"].astype(str)

    opened_sources = set()
    for path in OPENED_PREVIEWS:
        if path.exists():
            opened = pd.read_csv(path, dtype=str).fillna("")
            if "source_record_id" in opened.columns:
                opened_sources.update(opened["source_record_id"].astype(str))

    df = df[~df["source_record_id"].isin(opened_sources)].copy()

    for col in ["word_count", "char_count", "anchor_count", "comma_count", "number_word_count", "upper_acronym_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "terminal_period" in df.columns:
        df["terminal_period_bool"] = as_bool(df["terminal_period"])
    else:
        df["terminal_period_bool"] = df["sentence_text"].astype(str).str.rstrip().str.endswith(".")

    block_counter = Counter()
    candidate_rows = []

    for row in df.to_dict("records"):
        text = str(row.get("sentence_text", ""))
        blockers = local_blockers(text)

        wc = int(row.get("word_count", 0) or 0)
        cc = int(row.get("char_count", 0) or 0)
        anchors = int(row.get("anchor_count", 0) or 0)
        comma_count = int(row.get("comma_count", 0) or 0)
        num_words = int(row.get("number_word_count", 0) or 0)
        acronyms = int(row.get("upper_acronym_count", 0) or 0)
        nscore = news_score(text)

        reasons = []

        if str(row.get("doc_level_risks_v1_4a", "")).strip():
            reasons.append("has_doc_level_risk")

        if str(row.get("sentence_risks_v1_4a", "")).strip():
            reasons.append("has_sentence_risk")

        if str(row.get("mild_risks_v1_4a", "")).strip():
            reasons.append("has_mild_risk")

        if blockers:
            reasons.append("has_local_blocker")

        if not row.get("terminal_period_bool", False):
            reasons.append("not_terminal_period")

        if wc < 8:
            reasons.append("too_few_words")
        if wc > 13:
            reasons.append("too_many_words")

        if cc < 55:
            reasons.append("too_short_chars")
        if cc > 120:
            reasons.append("too_long_chars")

        if anchors < 3:
            reasons.append("low_common_anchor_count")

        if nscore < 2:
            reasons.append("low_news_anchor_score")

        if comma_count > 0:
            reasons.append("comma_present")

        if num_words > 0:
            reasons.append("number_word_present")

        if acronyms > 0:
            reasons.append("uppercase_acronym_present")

        block_counter.update(reasons)
        block_counter.update(blockers)

        row["news_anchor_score_v1_4c"] = nscore
        row["local_blockers_v1_4c"] = "|".join(blockers)
        row["candidate_block_reasons_v1_4c"] = "|".join(sorted(set(reasons)))
        row["news_style_candidate_v1_4c"] = len(reasons) == 0

        if len(reasons) == 0:
            candidate_rows.append(row)

    scored = pd.DataFrame(df.to_dict("records"))
    # Above scored does not include row modifications, so rebuild from loop output:
    all_rows = []
    for row in df.to_dict("records"):
        text = str(row.get("sentence_text", ""))
        blockers = local_blockers(text)

        wc = int(row.get("word_count", 0) or 0)
        cc = int(row.get("char_count", 0) or 0)
        anchors = int(row.get("anchor_count", 0) or 0)
        comma_count = int(row.get("comma_count", 0) or 0)
        num_words = int(row.get("number_word_count", 0) or 0)
        acronyms = int(row.get("upper_acronym_count", 0) or 0)
        nscore = news_score(text)

        reasons = []
        if str(row.get("doc_level_risks_v1_4a", "")).strip():
            reasons.append("has_doc_level_risk")
        if str(row.get("sentence_risks_v1_4a", "")).strip():
            reasons.append("has_sentence_risk")
        if str(row.get("mild_risks_v1_4a", "")).strip():
            reasons.append("has_mild_risk")
        if blockers:
            reasons.append("has_local_blocker")
        if not row.get("terminal_period_bool", False):
            reasons.append("not_terminal_period")
        if wc < 8:
            reasons.append("too_few_words")
        if wc > 13:
            reasons.append("too_many_words")
        if cc < 55:
            reasons.append("too_short_chars")
        if cc > 120:
            reasons.append("too_long_chars")
        if anchors < 3:
            reasons.append("low_common_anchor_count")
        if nscore < 2:
            reasons.append("low_news_anchor_score")
        if comma_count > 0:
            reasons.append("comma_present")
        if num_words > 0:
            reasons.append("number_word_present")
        if acronyms > 0:
            reasons.append("uppercase_acronym_present")

        row["news_anchor_score_v1_4c"] = nscore
        row["local_blockers_v1_4c"] = "|".join(blockers)
        row["candidate_block_reasons_v1_4c"] = "|".join(sorted(set(reasons)))
        row["news_style_candidate_v1_4c"] = len(reasons) == 0
        all_rows.append(row)

    scored = pd.DataFrame(all_rows)
    candidates = scored[scored["news_style_candidate_v1_4c"]].copy()

    if len(candidates):
        candidates["normalized_sentence_text_for_dedup"] = (
            candidates["sentence_text"]
            .map(norm)
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        candidates = candidates.drop_duplicates(
            subset=["normalized_sentence_text_for_dedup"],
            keep="first",
        ).copy()

        review = (
            candidates
            .sample(frac=1, random_state=RANDOM_SEED)
            .groupby("source_record_id", as_index=False, sort=False)
            .head(1)
            .sample(n=min(SAMPLE_SIZE, candidates["source_record_id"].nunique()), random_state=RANDOM_SEED)
            .copy()
        )
    else:
        review = candidates.copy()

    review.insert(0, "manual_clean_label", "")
    review.insert(1, "manual_notes", "")

    scored_path = OUT / "fresh_sentences_scored_v1_4c.csv"
    cand_path = OUT / "news_style_sentence_candidates_v1_4c.csv"
    review_path = OUT / "manual_review_news_style_200_v1_4c.csv"
    summary_path = OUT / "news_style_clean_seed_summary_v1_4c.md"

    scored.to_csv(scored_path, index=False)
    candidates.to_csv(cand_path, index=False)

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
        "news_anchor_score_v1_4c",
        "sentence_text",
        "local_blockers_v1_4c",
        "candidate_block_reasons_v1_4c",
    ]
    review[[c for c in review_cols if c in review.columns]].to_csv(review_path, index=False)

    candidate_block_counts = Counter()
    local_block_counts = Counter()

    for val in scored["candidate_block_reasons_v1_4c"].fillna(""):
        for r in str(val).split("|"):
            if r:
                candidate_block_counts[r] += 1

    for val in scored["local_blockers_v1_4c"].fillna(""):
        for r in str(val).split("|"):
            if r:
                local_block_counts[r] += 1

    part_counts = (
        candidates["part_number"].astype(str).value_counts().head(30)
        if len(candidates)
        else pd.Series(dtype=int)
    )

    lines = []
    lines.append("# News-Style Clean Seed v1.4c")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("Development preview after v1.4b failed. This version narrows the target to simple modern news/reporting style.")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- input_sentences_from_v1_4b_scored: `{len(pd.read_csv(IN_SCORED, dtype=str))}`")
    lines.append(f"- opened_preview_source_record_ids_excluded: `{len(opened_sources)}`")
    lines.append(f"- remaining_sentences_after_opened_exclusion: `{len(scored)}`")
    lines.append(f"- news_style_candidates_v1_4c: `{len(candidates)}`")
    lines.append(f"- review_preview_rows: `{len(review)}`")
    lines.append("")
    lines.append("## Top candidate block reasons")
    lines.append("")
    for k, v in candidate_block_counts.most_common(80):
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Top local blockers")
    lines.append("")
    for k, v in local_block_counts.most_common(80):
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Candidate distribution by part")
    lines.append("")
    for part, count in part_counts.items():
        lines.append(f"- part {part}: `{count}`")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- scored: `{scored_path}`")
    lines.append(f"- candidates: `{cand_path}`")
    lines.append(f"- review preview: `{review_path}`")
    lines.append("")
    lines.append("## Decision rule")
    lines.append("")
    lines.append("- Inspect preview first.")
    lines.append("- Do not label if visibly contaminated.")
    lines.append("- If preview looks clean, label `manual_clean_label` as `CLEAN_TARGET` or `NOT_CLEAN_TARGET`.")
    lines.append("- If manual precision >= 95%, v1.4c passes as a narrow clean seed source.")
    lines.append("- If precision < 95%, stop heuristic extraction and move to manually curated seed expansion.")

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))

    print("\nFIRST 100 NEWS-STYLE REVIEW PREVIEW ROWS")
    for i, row in review.head(100).reset_index(drop=True).iterrows():
        print("\n" + "=" * 120)
        print("ROW:", i + 1)
        print("source_record_id:", row.get("source_record_id", ""))
        print("part:", row.get("part_number", ""))
        print("sentence_id:", row.get("sentence_id", ""))
        print("words:", row.get("word_count", ""))
        print("anchors:", row.get("anchor_count", ""))
        print("news_score:", row.get("news_anchor_score_v1_4c", ""))
        print("text:", row.get("sentence_text", ""))


if __name__ == "__main__":
    main()
