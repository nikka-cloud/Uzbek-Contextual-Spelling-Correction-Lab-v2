#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, hashlib, json, random, re, statistics, sys, unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import ijson

APOSTROPHES = ["'", "`", "´", "ʻ", "ʼ", "‘", "’", "‛", "′"]
SUSPICIOUS = {"\u00a0", "\u00ad", "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\ufffd"}
OUTPUTS = [
    "corpus_schema.json", "record_type_counts.csv", "field_presence_counts.csv",
    "text_value_type_counts.csv", "text_length_summary.csv",
    "apostrophe_variant_counts.csv", "script_distribution.csv",
    "quality_flag_counts.csv", "sample_records.csv", "duplicate_examples.csv",
    "mixed_script_examples.csv", "suspicious_unicode_examples.csv",
    "corpus_audit_metrics.md",
]

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream-audit a top-level JSON array.")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-records", type=int, default=50000, help="0 means full corpus")
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--example-limit", type=int, default=50)
    p.add_argument("--preview-chars", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress-every", type=int, default=10000)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    for name in ("max_records", "sample_size", "example_limit", "progress_every"):
        if getattr(a, name) < 0:
            p.error(f"--{name.replace('_','-')} must be >= 0")
    if a.preview_chars <= 0:
        p.error("--preview-chars must be > 0")
    return a

def tname(value: Any) -> str:
    return "null" if value is None else type(value).__name__

def preview(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text.replace("\r","\\r").replace("\n","\\n").replace("\t","\\t")[:limit]

def pct(count: int, total: int) -> float:
    return round(100 * count / total, 4) if total else 0.0

def script_info(text: str) -> tuple[str,int,int,int]:
    latin = cyr = other = 0
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if "LATIN" in name: latin += 1
        elif "CYRILLIC" in name: cyr += 1
        else: other += 1
    if latin == cyr == other == 0: label = "NO_LETTERS"
    elif latin and not cyr and not other: label = "LATIN"
    elif cyr and not latin and not other: label = "CYRILLIC"
    elif latin and cyr: label = "LATIN_CYRILLIC_MIXED"
    elif other and not latin and not cyr: label = "OTHER_SCRIPT"
    else: label = "MULTISCRIPT_OTHER"
    return label, latin, cyr, other

def suspicious_chars(text: str) -> Counter[str]:
    out = Counter()
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in SUSPICIOUS or cat == "Cf" or (cat == "Cc" and ch not in "\n\r\t"):
            out[ch] += 1
    return out

def flags_for(text: str, label: str, suspicious: Counter[str]) -> list[str]:
    flags = []
    if not text.strip(): flags.append("empty_text")
    if text != text.lstrip(): flags.append("leading_whitespace")
    if text != text.rstrip(): flags.append("trailing_whitespace")
    if re.search(r"[ \t]{2,}", text): flags.append("repeated_horizontal_whitespace")
    if "\t" in text: flags.append("contains_tab")
    if "\n" in text or "\r" in text: flags.append("contains_newline")
    if any(unicodedata.category(ch) == "Cc" and ch not in "\n\r\t" for ch in text):
        flags.append("contains_control_character")
    if any(unicodedata.category(ch) == "Cf" for ch in text):
        flags.append("contains_invisible_format_character")
    if "\ufffd" in text: flags.append("contains_replacement_character")
    if "\u00a0" in text: flags.append("contains_nonbreaking_space")
    if label == "LATIN_CYRILLIC_MIXED": flags.append("latin_cyrillic_mixed")
    if suspicious and not any(x.startswith("contains_") for x in flags):
        flags.append("contains_suspicious_unicode")
    return flags

def percentile(values: list[int], q: float) -> int | None:
    if not values: return None
    ordered = sorted(values)
    return ordered[round((len(ordered)-1)*q)]

def length_rows(chars: list[int], words: list[int], lines: list[int]) -> list[dict[str,Any]]:
    specs = [
        ("minimum", min), ("median", statistics.median),
        ("p90", lambda x: percentile(x,.90)), ("p95", lambda x: percentile(x,.95)),
        ("p99", lambda x: percentile(x,.99)), ("maximum", max),
        ("mean", lambda x: round(statistics.fmean(x),2)),
    ]
    rows = []
    for name, fn in specs:
        rows.append({
            "metric": name,
            "character_count": fn(chars) if chars else "",
            "word_count_estimate": fn(words) if words else "",
            "line_count": fn(lines) if lines else "",
        })
    return rows

def write_csv(path: Path, fields: list[str], rows: list[dict[str,Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def main() -> int:
    a = args()
    source = a.input.expanduser().resolve()
    out = a.output_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    out.mkdir(parents=True, exist_ok=True)
    present = [name for name in OUTPUTS if (out/name).exists()]
    if present and not a.overwrite:
        raise FileExistsError("Existing outputs: " + ", ".join(present))

    before = source.stat()
    with source.open("rb") as f:
        first = f.read(65536).lstrip(b"\xef\xbb\xbf \t\r\n")[:1]
    if first != b"[":
        raise ValueError(f"Expected top-level JSON array, found {first!r}")

    rng = random.Random(a.seed)
    record_types = Counter()
    field_counts = Counter()
    keysets = Counter()
    text_types = Counter()
    scripts = Counter()
    qflags = Counter()
    apostrophe_hits = Counter()
    apostrophe_records = Counter()
    suspicious_hits = Counter()
    suspicious_records = Counter()

    char_lengths, word_lengths, line_lengths = [], [], []
    random_samples, mixed_examples, unicode_examples = [], [], []
    special_samples: dict[int,dict[str,Any]] = {}
    dup_index: dict[tuple[str,int],dict[str,Any]] = {}

    scanned = dict_records = with_text = string_texts = dup_after_first = 0

    with source.open("rb") as f:
        for rid, rec in enumerate(ijson.items(f, "item")):
            if a.max_records and rid >= a.max_records:
                break
            scanned += 1
            record_types[tname(rec)] += 1

            if not isinstance(rec, dict):
                qflags["non_dictionary_record"] += 1
                if len(special_samples) < a.example_limit:
                    special_samples[rid] = {
                        "source_record_id": rid, "sample_reason": "non_dictionary_record",
                        "text_preview": preview(rec,a.preview_chars), "character_count": "",
                        "word_count_estimate": "", "line_count": "", "script_label": "",
                        "quality_flags": "non_dictionary_record",
                    }
                continue

            dict_records += 1
            keysets[tuple(sorted(map(str,rec.keys())))] += 1
            for key in rec: field_counts[str(key)] += 1

            if "text" not in rec:
                qflags["missing_text"] += 1
                continue

            with_text += 1
            text = rec["text"]
            text_types[tname(text)] += 1
            if not isinstance(text, str):
                qflags["non_string_text"] += 1
                continue

            string_texts += 1
            cc = len(text)
            wc = len(re.findall(r"\S+", text))
            lc = text.count("\n") + 1
            char_lengths.append(cc); word_lengths.append(wc); line_lengths.append(lc)

            label, latin, cyr, other = script_info(text)
            scripts[label] += 1
            suspicious = suspicious_chars(text)
            flags = flags_for(text, label, suspicious)
            for flag in flags: qflags[flag] += 1

            for ch in APOSTROPHES:
                n = text.count(ch)
                if n:
                    apostrophe_hits[ch] += n
                    apostrophe_records[ch] += 1

            for ch, n in suspicious.items():
                suspicious_hits[ch] += n
                suspicious_records[ch] += 1
                if len(unicode_examples) < a.example_limit:
                    pos = text.find(ch)
                    context = text[max(0,pos-80):min(len(text),pos+81)]
                    unicode_examples.append({
                        "source_record_id": rid, "character_display": repr(ch),
                        "unicode_codepoint": f"U+{ord(ch):04X}",
                        "unicode_name": unicodedata.name(ch,"UNKNOWN"),
                        "unicode_category": unicodedata.category(ch),
                        "occurrences_in_record": n,
                        "context_preview": preview(context,a.preview_chars),
                    })

            if label == "LATIN_CYRILLIC_MIXED" and len(mixed_examples) < a.example_limit:
                mixed_examples.append({
                    "source_record_id": rid, "latin_letter_count": latin,
                    "cyrillic_letter_count": cyr, "other_letter_count": other,
                    "script_label": label, "text_preview": preview(text,a.preview_chars),
                })

            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            key = (digest, cc)
            if key not in dup_index:
                dup_index[key] = {
                    "duplicate_hash": digest, "occurrence_count": 1,
                    "first_source_record_id": rid, "other_source_record_ids": [],
                    "character_count": cc, "text_preview": preview(text,a.preview_chars),
                }
            else:
                dup_index[key]["occurrence_count"] += 1
                dup_after_first += 1
                qflags["exact_duplicate_after_first"] += 1
                if len(dup_index[key]["other_source_record_ids"]) < a.example_limit:
                    dup_index[key]["other_source_record_ids"].append(rid)

            row = {
                "source_record_id": rid, "sample_reason": "deterministic_random_sample",
                "text_preview": preview(text,a.preview_chars), "character_count": cc,
                "word_count_estimate": wc, "line_count": lc, "script_label": label,
                "quality_flags": ";".join(flags),
            }
            if len(random_samples) < a.sample_size:
                random_samples.append(row)
            elif a.sample_size:
                pick = rng.randint(1,string_texts)
                if pick <= a.sample_size:
                    random_samples[pick-1] = row
            if flags and len(special_samples) < a.example_limit:
                special_samples.setdefault(rid,{**row,"sample_reason":"quality_flag_example"})

            if a.progress_every and scanned % a.progress_every == 0:
                print(f"PROGRESS records={scanned:,} string_texts={string_texts:,}",
                      file=sys.stderr, flush=True)

    after = source.stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
        raise RuntimeError("Raw corpus changed during audit")

    dups = [v for v in dup_index.values() if v["occurrence_count"] > 1]
    dups.sort(key=lambda r:(-r["occurrence_count"],r["first_source_record_id"]))
    dup_rows = []
    for row in dups[:a.example_limit]:
        dup_rows.append({**row,
            "other_source_record_ids":";".join(map(str,row["other_source_record_ids"]))})

    samples = dict(special_samples)
    for row in random_samples: samples.setdefault(row["source_record_id"],row)
    sample_rows = sorted(samples.values(),key=lambda r:r["source_record_id"])

    record_rows = [{"record_type":k,"record_count":v,
                    "percentage_of_records_scanned":pct(v,scanned)}
                   for k,v in record_types.most_common()]
    field_rows = [{"field_name":k,"records_with_field":v,
                   "percentage_of_dictionary_records":pct(v,dict_records)}
                  for k,v in field_counts.most_common()]
    text_rows = [{"text_value_type":k,"record_count":v,
                  "percentage_of_text_fields":pct(v,with_text)}
                 for k,v in text_types.most_common()]
    script_rows = [{"script_label":k,"record_count":v,
                    "percentage_of_string_text_records":pct(v,string_texts)}
                   for k,v in scripts.most_common()]
    flag_rows = [{"quality_flag":k,"record_count":v,
                  "percentage_of_records_scanned":pct(v,scanned)}
                 for k,v in qflags.most_common()]
    apostrophe_rows = [{
        "character":ch, "unicode_codepoint":f"U+{ord(ch):04X}",
        "unicode_name":unicodedata.name(ch,"UNKNOWN"),
        "total_occurrences":apostrophe_hits[ch],
        "records_containing_character":apostrophe_records[ch],
        "possible_normalization_group":"APOSTROPHE_LIKE",
        "automatic_change_allowed":"false",
    } for ch in APOSTROPHES if apostrophe_hits[ch]]
    apostrophe_rows.sort(key=lambda r:-r["total_occurrences"])
    lengths = length_rows(char_lengths,word_lengths,line_lengths)

    write_csv(out/"record_type_counts.csv",
              ["record_type","record_count","percentage_of_records_scanned"],record_rows)
    write_csv(out/"field_presence_counts.csv",
              ["field_name","records_with_field","percentage_of_dictionary_records"],field_rows)
    write_csv(out/"text_value_type_counts.csv",
              ["text_value_type","record_count","percentage_of_text_fields"],text_rows)
    write_csv(out/"text_length_summary.csv",
              ["metric","character_count","word_count_estimate","line_count"],lengths)
    write_csv(out/"apostrophe_variant_counts.csv",
              ["character","unicode_codepoint","unicode_name","total_occurrences",
               "records_containing_character","possible_normalization_group",
               "automatic_change_allowed"],apostrophe_rows)
    write_csv(out/"script_distribution.csv",
              ["script_label","record_count","percentage_of_string_text_records"],script_rows)
    write_csv(out/"quality_flag_counts.csv",
              ["quality_flag","record_count","percentage_of_records_scanned"],flag_rows)
    write_csv(out/"sample_records.csv",
              ["source_record_id","sample_reason","text_preview","character_count",
               "word_count_estimate","line_count","script_label","quality_flags"],sample_rows)
    write_csv(out/"duplicate_examples.csv",
              ["duplicate_hash","occurrence_count","first_source_record_id",
               "other_source_record_ids","character_count","text_preview"],dup_rows)
    write_csv(out/"mixed_script_examples.csv",
              ["source_record_id","latin_letter_count","cyrillic_letter_count",
               "other_letter_count","script_label","text_preview"],mixed_examples)
    write_csv(out/"suspicious_unicode_examples.csv",
              ["source_record_id","character_display","unicode_codepoint","unicode_name",
               "unicode_category","occurrences_in_record","context_preview"],unicode_examples)

    schema = {
        "input_path": str(source), "input_size_bytes": before.st_size,
        "input_modified_time_ns": before.st_mtime_ns, "format": "JSON_ARRAY",
        "parser": {"library":"ijson","version":getattr(ijson,"__version__","unknown"),
                   "backend":getattr(ijson,"backend","unknown")},
        "scan_configuration": {
            "max_records":a.max_records,"sample_size":a.sample_size,
            "example_limit":a.example_limit,"preview_chars":a.preview_chars,"seed":a.seed,
        },
        "counts": {
            "records_scanned":scanned,"dictionary_records":dict_records,
            "non_dictionary_records":scanned-dict_records,"records_with_text":with_text,
            "records_missing_text":dict_records-with_text,"string_text_records":string_texts,
            "non_string_text_records":with_text-string_texts,"duplicate_groups":len(dups),
            "duplicate_occurrences_after_first":dup_after_first,
        },
        "record_type_counts":dict(record_types),
        "field_presence_counts":dict(field_counts),
        "key_combination_counts":{"|".join(k):v for k,v in keysets.items()},
        "text_value_type_counts":dict(text_types),"script_counts":dict(scripts),
        "quality_flag_counts":dict(qflags),"length_summary":lengths,
        "apostrophe_occurrences":{f"U+{ord(k):04X}":v for k,v in apostrophe_hits.items()},
        "suspicious_unicode_occurrences":{f"U+{ord(k):04X}":v for k,v in suspicious_hits.items()},
        "raw_corpus_immutability_check":"PASS",
        "known_limitations":[
            "A partial scan cannot prove full-corpus consistency.",
            "Script labels describe Unicode scripts, not language identity.",
            "Word counts are whitespace-based estimates, not tokenizer counts.",
            "Duplicate detection uses SHA-256 plus character length.",
            "The in-memory duplicate index is suitable for the development scan; a full scan may require a disk-backed index.",
            "Quality flags are descriptive and do not automatically reject records.",
        ],
    }
    (out/"corpus_schema.json").write_text(
        json.dumps(schema,ensure_ascii=False,indent=2),encoding="utf-8")

    report = [
        "# Corpus Audit Metrics","",
        "## Run configuration","",
        f"- Input: `{source}`",f"- Output: `{out}`",
        f"- Maximum records requested: {a.max_records}",
        f"- Records scanned: {scanned}",f"- Random seed: {a.seed}",
        f"- ijson version: {getattr(ijson,'__version__','unknown')}",
        f"- ijson backend: {getattr(ijson,'backend','unknown')}","",
        "## Core schema findings","",
        f"- Dictionary records: {dict_records}",
        f"- Non-dictionary records: {scanned-dict_records}",
        f"- Records with `text`: {with_text}",
        f"- Records missing `text`: {dict_records-with_text}",
        f"- String text records: {string_texts}",
        f"- Non-string text records: {with_text-string_texts}","",
        "## Duplicate findings","",
        f"- Duplicate groups: {len(dups)}",
        f"- Duplicate occurrences after the first: {dup_after_first}","",
        "## Safety","",
        "- Raw corpus size and modification timestamp were unchanged: PASS",
        "- No corpus text was normalized or corrected.",
        "- Exported text is limited to configured previews.","",
        "## Interpretation limits","",
        "- A partial scan cannot prove complete-corpus consistency.",
        "- Script labels are not language labels.",
        "- Word counts are whitespace-based estimates.",
        "- Quality flags require human review.",
        "- Full-corpus duplicate tracking may need a disk-backed index.","",
    ]
    (out/"corpus_audit_metrics.md").write_text("\n".join(report),encoding="utf-8")

    print("--- AUDIT COMPLETE ---")
    print(f"records_scanned={scanned}")
    print(f"dictionary_records={dict_records}")
    print(f"string_text_records={string_texts}")
    print(f"duplicate_groups={len(dups)}")
    print(f"duplicate_occurrences_after_first={dup_after_first}")
    print(f"output_dir={out}")
    print("immutability_check=PASS")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}",file=sys.stderr)
        raise SystemExit(1)
