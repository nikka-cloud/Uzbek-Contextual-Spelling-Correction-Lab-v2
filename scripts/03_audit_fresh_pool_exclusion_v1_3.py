from __future__ import annotations

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

OUT = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1/"
    "strict_clean_seed_extractor_v1_3_lexical_gate/"
    "fresh_pool_exclusion_audit_v1_3"
)
OUT.mkdir(parents=True, exist_ok=True)


def parse_listish(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s or s == "[]":
        return []
    # handles strings like "['A', 'B']" safely enough for our audit
    return re.findall(r"[A-Z_]+", s)


def plain_shape_prefilter(text: str, boundary_flags: list[str], parent_flags: list[str]) -> tuple[bool, list[str]]:
    reasons = []

    t = str(text).strip()
    tokens = t.split()

    if boundary_flags:
        reasons.append("has_boundary_flags")
    if parent_flags:
        reasons.append("has_parent_flags")
    if not t.endswith("."):
        reasons.append("not_plain_period")
    if "?" in t or "!" in t or ":" in t:
        reasons.append("question_exclamation_or_colon")
    if "," in t:
        reasons.append("comma_present")
    if re.search(r"\d", t):
        reasons.append("digit_present")

    if len(tokens) < 6:
        reasons.append("too_few_words")
    if len(tokens) > 16:
        reasons.append("too_many_words")
    if len(t) < 30:
        reasons.append("too_short_chars")
    if len(t) > 120:
        reasons.append("too_long_chars")

    alpha = sum(ch.isalpha() for ch in t)
    punct = sum((not ch.isalnum() and not ch.isspace()) for ch in t)
    chars = len(t) or 1

    if alpha / chars < 0.78:
        reasons.append("low_alpha_ratio")
    if punct / chars > 0.08:
        reasons.append("punctuation_heavy")

    # v1.3-style local blockers we already learned
    lower = t.lower()

    suspicious_tokens = {
        "shaxt",
        "ananalar",
        "yeki",
        "dadamiing",
        "meditsinada",
        "dorivorlar",
    }
    for tok in re.findall(r"[A-Za-zʻʼ'`-]+", lower):
        norm = tok.replace("ʻ", "'").replace("ʼ", "'").replace("`", "'")
        if norm in suspicious_tokens:
            reasons.append(f"suspicious_token:{norm}")

    suspicious_phrases = [
        "yil may",
        "bir maygacha",
        "beshinchi hujayralar",
        "har qanday faoliyati",
    ]
    for phrase in suspicious_phrases:
        if phrase in lower:
            reasons.append(f"suspicious_phrase:{phrase.replace(' ', '_')}")

    return (len(reasons) == 0), reasons


old = pd.read_csv(OLD_MASTER, dtype=str).fillna("")
old.columns = [c.lstrip("\ufeff") for c in old.columns]

old_sentence_ids = set(old["sentence_id"].astype(str).str.strip())
old_source_ids = set(old["source_record_id"].astype(str).str.strip())

counts = Counter()
reason_counts = Counter()
part_counts = Counter()
candidate_rows = []

with POOL.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)

        sid = str(obj.get("sentence_id", "")).strip()
        src = str(obj.get("source_record_id", "")).strip()
        text = str(obj.get("sentence_text", "")).strip()

        counts["pool_total"] += 1

        if sid in old_sentence_ids:
            counts["excluded_old_sentence_id"] += 1
            continue

        if src in old_source_ids:
            counts["excluded_old_source_record_id"] += 1
            continue

        counts["fresh_after_source_exclusion"] += 1

        boundary_flags = parse_listish(obj.get("boundary_flags"))
        parent_flags = parse_listish(obj.get("parent_review_flags"))

        ok, reasons = plain_shape_prefilter(text, boundary_flags, parent_flags)

        if ok:
            counts["plain_shape_prefilter_pass"] += 1
            part_counts[str(obj.get("part_number", ""))] += 1

            candidate_rows.append({
                "sentence_id": sid,
                "source_record_id": src,
                "part_number": obj.get("part_number", ""),
                "document_sample_id": obj.get("document_sample_id", ""),
                "sentence_index_in_record": obj.get("sentence_index_in_record", ""),
                "sentence_character_count": obj.get("sentence_character_count", ""),
                "sentence_token_count": obj.get("sentence_token_count", ""),
                "boundary_flags": "|".join(boundary_flags),
                "parent_review_flags": "|".join(parent_flags),
                "sentence_text": text,
            })
        else:
            for r in reasons:
                reason_counts[r] += 1


candidate_df = pd.DataFrame(candidate_rows)

candidate_path = OUT / "fresh_pool_plain_shape_prefilter_candidates_v1_3.csv"
candidate_df.to_csv(candidate_path, index=False)

summary_lines = []
summary_lines.append("# Fresh Pool Exclusion Audit v1.3")
summary_lines.append("")
summary_lines.append("## Counts")
summary_lines.append("")
for k, v in counts.items():
    summary_lines.append(f"- {k}: `{v}`")

summary_lines.append("")
summary_lines.append("## Top prefilter block reasons")
summary_lines.append("")
for k, v in reason_counts.most_common(50):
    summary_lines.append(f"- {k}: `{v}`")

summary_lines.append("")
summary_lines.append("## Candidate distribution by part")
summary_lines.append("")
for k, v in part_counts.most_common(30):
    summary_lines.append(f"- part {k}: `{v}`")

summary_lines.append("")
summary_lines.append("## Output")
summary_lines.append("")
summary_lines.append(f"- candidates: `{candidate_path}`")

summary_path = OUT / "fresh_pool_exclusion_audit_v1_3.md"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

print(summary_path.read_text(encoding="utf-8"))

print("\nFIRST 30 PREFILTER CANDIDATES")
for i, row in candidate_df.head(30).iterrows():
    print("\n" + "=" * 120)
    print("ROW:", i + 1)
    print("sentence_id:", row["sentence_id"])
    print("source_record_id:", row["source_record_id"])
    print("part:", row["part_number"])
    print("text:", row["sentence_text"])
