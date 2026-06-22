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

IN_SCORED = BASE / "document_safe_clean_seed_v1_4a" / "fresh_sentences_scored_v1_4a.csv"

OPENED_V1_4A = BASE / "document_safe_clean_seed_v1_4a" / "manual_review_document_safe_200_v1_4a.csv"

OUT = BASE / "document_safe_clean_seed_v1_4b"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 200
RANDOM_SEED = 2026061806


BAD_EXACT_TOKENS = {
    # old observed failures
    "shaxt", "ananalar", "yeki", "dadamiing", "meditsinada", "dorivorlar",
    "uulki", "utkaziladi", "qilmaymen", "narsanidim", "naytida", "oralikpi",
    "kassalanga", "chiqqan", "net", "devdim", "xam", "tabiy", "tomenidan",
    "mablag", "quron", "maksadga", "yuli", "alahsiymiz", "sotiboldi",

    # v1.4a preview failures
    "keraqli", "xujjatlar", "xujjat", "vidyabu", "rur", "apiltakil",
    "qo'yechkilar", "tarlari", "qo'rg'oinping", "donrasi", "tomirpomiri",
    "pto'rolar", "tarixisizlarga", "qulosiga", "xarorat", "pereferik",
    "feysbukda", "ko'rub", "sevinmakdamiz", "g'ovurg'uvuridan",
    "chervyakli", "reduktorlar", "sinxronik", "diaxronik", "nirvana",
    "azobuqubatlariga", "azobuqubat", "domulla", "qo'yechki", "lishaynik",
    "impichment", "tramp", "nitroglitserinni", "radioaloqa", "epidemiologik",
    "rezeksiyalar", "limfangektaziyalarda", "kriptovalyuta", "bitkoin",
    "laytkoin", "pto'rolar", "deviy", "fuzalolar", "muhiblaru",
    "qo'rg'oinping", "qoloqroq",
}

BAD_SUBSTRINGS = {
    # merged / spacing corruption
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
    "otaonalar",
    "yangiyangi",
    "kechakunduz",
    "biryo'la",
    "tomirpomiri",
    "ko'pda",
    "obraz larning",
    "noziklik larini",
    "tushun ganligingiz",
    "xos, noziklik",
    "qo'rg'oin",
    "tarixisiz",
    "qim mat",
    "ediu",
    "edilaru",
    "u albinadan",
    "deviy, nirvana",
    "qon ko'pira",
    "tushayotganimizni his",
}

DOMAIN_RISK_SUBSTRINGS = {
    "stomatolog", "radioaloqa", "epidemiolog", "nitroglitserin",
    "chervyak", "reduktor", "sinxronik", "diaxronik", "kripto",
    "bitkoin", "laytkoin", "limfang", "rezeksiya", "planetar",
    "yadro quroli", "ballistik", "impichment", "kampaniya",
    "yustinian", "institutsiya", "protsedura", "konsignatsiya",
    "inkassatsiya", "elektromagnit", "amplituda", "elektrolit",
    "parallelepiped", "protoplast", "bikarbonat", "tsellyuloza",
}

FOREIGN_RISK_SUBSTRINGS = {
    "u nas", "ved ", "prekrasnie", "komnati", "cauda", "kursiy ruf",
    "html", "display", "been", " real ",
}


WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+|\d+")


def norm_apostrophe(text: str) -> str:
    return (
        str(text)
        .replace("ʻ", "'")
        .replace("ʼ", "'")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("`", "'")
    )


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(norm_apostrophe(text).lower())]


def tokens(text: str) -> list[str]:
    return [w.lower() for w in TOKEN_RE.findall(norm_apostrophe(text).lower())]


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def extra_candidate_blockers(text: str) -> list[str]:
    clean = norm_apostrophe(text)
    low = clean.lower()
    toks = tokens(clean)

    reasons = []

    for t in toks:
        if t in BAD_EXACT_TOKENS:
            reasons.append(f"bad_exact:{t}")

        # suspicious very short unknown-ish garbage tokens
        if len(t) <= 3 and t not in {
            "u", "bu", "shu", "ham", "esa", "men", "sen", "biz", "ular",
            "eng", "kam", "ko'p", "bir", "ikki", "uch", "besh", "olti",
            "yil", "kun", "oy", "ha", "yo'q", "bor", "edi", "deb",
            "va", "yo", "har", "ana", "shu", "uni", "ega", "o'z",
        }:
            if t.isalpha():
                reasons.append(f"short_suspicious_token:{t}")

        if len(t) >= 18 and t.isalpha():
            reasons.append(f"very_long_alpha_token:{t[:30]}")

        if re.search(r"(.)\1\1", t):
            reasons.append(f"triple_repeated_char:{t[:30]}")

    for sub in BAD_SUBSTRINGS:
        if sub in low:
            reasons.append(f"bad_substring:{sub}")

    for sub in DOMAIN_RISK_SUBSTRINGS:
        if sub in low:
            reasons.append(f"domain_risk:{sub}")

    for sub in FOREIGN_RISK_SUBSTRINGS:
        if sub in f" {low} ":
            reasons.append(f"foreign_risk:{sub.strip()}")

    # split Uzbek suffix corruption: obraz larning, noziklik larini, tushun ganligingiz
    if re.search(
        r"\b[a-z']{4,}\s+(larning|larni|larini|larining|lariga|lardan|ganligi|ganligingiz|ganligini|moqda|dagi)\b",
        low,
    ):
        reasons.append("split_suffix_like_corruption")

    # lower+Upper merge: Vidyabu style is already lower text check impossible after lower,
    # so check original.
    if re.search(r"[a-z][A-Z]", clean):
        reasons.append("midword_case_merge")

    # bad separated letters/garbage
    if re.search(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b", clean):
        reasons.append("spaced_letter_garbage")

    # require plain modern sentence: no comma in this strict seed
    if "," in clean:
        reasons.append("comma_present_strict_seed")

    # no quote/speech fragments
    if re.search(r"\b(deya|deb aytdi|deb tushuntirgan|deb javob berdi)\b", low):
        reasons.append("reported_speech_fragment")

    return sorted(set(reasons))


def main() -> None:
    if not IN_SCORED.exists():
        raise FileNotFoundError(IN_SCORED)

    df = pd.read_csv(IN_SCORED, dtype=str).fillna("")

    opened_sources = set()
    if OPENED_V1_4A.exists():
        opened = pd.read_csv(OPENED_V1_4A, dtype=str).fillna("")
        if "source_record_id" in opened.columns:
            opened_sources = set(opened["source_record_id"].astype(str))

    df["source_record_id"] = df["source_record_id"].astype(str)
    df = df[~df["source_record_id"].isin(opened_sources)].copy()

    numeric_cols = [
        "word_count",
        "char_count",
        "anchor_count",
        "comma_count",
        "number_word_count",
        "upper_acronym_count",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "terminal_period" in df.columns:
        df["terminal_period_bool"] = as_bool(df["terminal_period"])
    else:
        df["terminal_period_bool"] = df["sentence_text"].astype(str).str.rstrip().str.endswith(".")

    # Existing v1.4a risks.
    df["has_doc_level_risk_v1_4a"] = df["doc_level_risks_v1_4a"].astype(str).str.strip().ne("")
    df["has_sentence_risk_v1_4a"] = df["sentence_risks_v1_4a"].astype(str).str.strip().ne("")
    df["has_mild_risk_v1_4a"] = df["mild_risks_v1_4a"].astype(str).str.strip().ne("")

    # Re-score documents: only real doc-level risks reject whole document.
    doc_scores = []
    doc_reject_counter = Counter()

    for source_record_id, g in df.groupby("source_record_id", sort=False):
        total_sentences = len(g)
        doc_level_risk_count = int(g["has_doc_level_risk_v1_4a"].sum())

        reasons = []
        if doc_level_risk_count > 0:
            reasons.append("doc_has_doc_level_risk")
        if total_sentences < 2:
            reasons.append("doc_too_few_sentences")

        doc_reject_counter.update(reasons)

        doc_scores.append(
            {
                "source_record_id": source_record_id,
                "part_number": g["part_number"].iloc[0],
                "document_sample_id": g["document_sample_id"].iloc[0],
                "total_sentences": total_sentences,
                "doc_level_risk_sentence_count": doc_level_risk_count,
                "document_safe_v1_4b": len(reasons) == 0,
                "document_reject_reasons_v1_4b": "|".join(sorted(set(reasons))),
                "preview_first_sentence": g["sentence_text"].iloc[0],
            }
        )

    doc_scores = pd.DataFrame(doc_scores)
    safe_sources = set(
        doc_scores.loc[doc_scores["document_safe_v1_4b"], "source_record_id"].astype(str)
    )

    extra_reason_col = []
    candidate_reason_col = []
    candidate_flags = []
    extra_counter = Counter()
    cand_counter = Counter()

    for row in df.to_dict("records"):
        text = str(row.get("sentence_text", ""))
        extra = extra_candidate_blockers(text)
        reasons = []

        if row["source_record_id"] not in safe_sources:
            reasons.append("document_not_safe_v1_4b")

        if str(row.get("doc_level_risks_v1_4a", "")).strip():
            reasons.append("has_doc_level_risk")

        if str(row.get("sentence_risks_v1_4a", "")).strip():
            reasons.append("has_sentence_risk")

        # For v1.4b strict seed, mild risks are also blocked.
        if str(row.get("mild_risks_v1_4a", "")).strip():
            reasons.append("has_mild_risk")

        if extra:
            reasons.append("has_extra_local_blocker")

        wc = int(row.get("word_count", 0) or 0)
        cc = int(row.get("char_count", 0) or 0)
        anchors = int(row.get("anchor_count", 0) or 0)
        commas = int(row.get("comma_count", 0) or 0)
        nums = int(row.get("number_word_count", 0) or 0)
        acronyms = int(row.get("upper_acronym_count", 0) or 0)

        if not bool(row.get("terminal_period_bool", False)):
            reasons.append("not_terminal_period")

        if wc < 8:
            reasons.append("too_few_words")
        if wc > 14:
            reasons.append("too_many_words")

        if cc < 50:
            reasons.append("too_short_chars")
        if cc > 125:
            reasons.append("too_long_chars")

        if anchors < 3:
            reasons.append("low_common_uzbek_anchor_count")

        if commas > 0:
            reasons.append("comma_present_strict_seed")

        if nums > 0:
            reasons.append("number_word_present_strict_seed")

        if acronyms > 0:
            reasons.append("uppercase_acronym")

        extra_reason_col.append("|".join(extra))
        candidate_reason_col.append("|".join(sorted(set(reasons))))
        candidate_flags.append(len(reasons) == 0)

        extra_counter.update(extra)
        cand_counter.update(reasons)

    df["extra_local_blockers_v1_4b"] = extra_reason_col
    df["candidate_block_reasons_v1_4b"] = candidate_reason_col
    df["simple_modern_candidate_v1_4b"] = candidate_flags

    candidates = df[df["simple_modern_candidate_v1_4b"]].copy()

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
        "extra_local_blockers_v1_4b",
        "candidate_block_reasons_v1_4b",
    ]

    review = review[[c for c in review_cols if c in review.columns]].copy()

    scored_path = OUT / "fresh_sentences_scored_v1_4b.csv"
    doc_path = OUT / "document_scores_v1_4b.csv"
    safe_doc_path = OUT / "safe_documents_v1_4b.csv"
    rejected_doc_path = OUT / "rejected_documents_v1_4b.csv"
    cand_path = OUT / "document_safe_sentence_candidates_v1_4b.csv"
    review_path = OUT / "manual_review_document_safe_200_v1_4b.csv"
    summary_path = OUT / "document_safe_clean_seed_summary_v1_4b.md"

    df.to_csv(scored_path, index=False)
    doc_scores.to_csv(doc_path, index=False)
    doc_scores[doc_scores["document_safe_v1_4b"]].to_csv(safe_doc_path, index=False)
    doc_scores[~doc_scores["document_safe_v1_4b"]].to_csv(rejected_doc_path, index=False)
    candidates.to_csv(cand_path, index=False)
    review.to_csv(review_path, index=False)

    part_counts = (
        candidates["part_number"].astype(str).value_counts().head(30)
        if len(candidates)
        else pd.Series(dtype=int)
    )

    lines = []
    lines.append("# Document-Safe Clean Seed Extractor v1.4b")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("Development preview after v1.4a preview contamination.")
    lines.append("v1.4b excludes v1.4a opened preview source records and adds stricter local candidate blockers.")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- input_fresh_sentences_from_v1_4a: `{len(pd.read_csv(IN_SCORED, dtype=str))}`")
    lines.append(f"- opened_v1_4a_source_record_ids_excluded: `{len(opened_sources)}`")
    lines.append(f"- remaining_sentences_after_v1_4a_exclusion: `{len(df)}`")
    lines.append(f"- remaining_source_documents: `{df['source_record_id'].nunique()}`")
    lines.append(f"- safe_documents_v1_4b: `{int(doc_scores['document_safe_v1_4b'].sum())}`")
    lines.append(f"- rejected_documents_v1_4b: `{int((~doc_scores['document_safe_v1_4b']).sum())}`")
    lines.append(f"- document_safe_sentence_candidates_v1_4b: `{len(candidates)}`")
    lines.append(f"- review_preview_rows: `{len(review)}`")
    lines.append("")
    lines.append("## Top document reject reasons")
    lines.append("")
    for k, v in doc_reject_counter.most_common(50):
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Top extra local blockers")
    lines.append("")
    for k, v in extra_counter.most_common(80):
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Top candidate block reasons")
    lines.append("")
    for k, v in cand_counter.most_common(80):
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Candidate distribution by part")
    lines.append("")
    for part, count in part_counts.items():
        lines.append(f"- part {part}: `{count}`")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- scored sentences: `{scored_path}`")
    lines.append(f"- document scores: `{doc_path}`")
    lines.append(f"- safe documents: `{safe_doc_path}`")
    lines.append(f"- rejected documents: `{rejected_doc_path}`")
    lines.append(f"- candidates: `{cand_path}`")
    lines.append(f"- review preview: `{review_path}`")
    lines.append("")
    lines.append("## Decision rule")
    lines.append("")
    lines.append("- Inspect preview first.")
    lines.append("- Do not label if visibly contaminated.")
    lines.append("- If preview looks clean, label `manual_clean_label` as `CLEAN_TARGET` or `NOT_CLEAN_TARGET`.")
    lines.append("- If manual precision >= 95%, v1.4b can be accepted as fresh-preview passed.")
    lines.append("- If manual precision < 95%, do not use v1.4b for corpus-wide clean target extraction.")

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
