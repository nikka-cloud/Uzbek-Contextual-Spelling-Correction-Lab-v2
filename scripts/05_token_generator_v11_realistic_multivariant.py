from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple


RANDOM_SEED = 42
MAX_VARIANTS_PER_TOKEN = 3
random.seed(RANDOM_SEED)

INPUT_PATH = Path(
    "outputs/05_synthetic_typo_generation/v10_deterministic_sweep/"
    "frozen_clean_targets_v1.csv"
)

OUT_DIR = Path("outputs/05_synthetic_typo_generation/v11_realistic_multivariant")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUT_DIR / "token_generator_v11_output.csv"
AUDIT_PATH = OUT_DIR / "token_generator_v11_generation_audit.csv"
SUMMARY_PATH = OUT_DIR / "token_generator_v11_summary.md"

TOKEN_RE = re.compile(r"\S+")
CORE_RE = re.compile(r"^([\"'“”‘’\(\[\{«»„“]*)(.*?)([.,!?;:…\"'“”‘’\)\]\}«»]*)$")


def split_token_surface(token: str) -> Tuple[str, str, str]:
    m = CORE_RE.match(token)
    if not m:
        return "", token, ""
    return m.group(1), m.group(2), m.group(3)


def is_eligible_core(core: str) -> bool:
    if len(core) < 2:
        return False
    if core.isdigit():
        return False
    if not re.search(r"[A-Za-zÀ-ÿʻʼ‘’'`-]", core):
        return False
    return True


def add_variant(
    variants: List[Dict[str, str]],
    core: str,
    corrupted: str,
    family: str,
    rule: str,
    priority: int,
) -> None:
    if not corrupted:
        return
    if corrupted == core:
        return
    if any(v["corrupted_core"] == corrupted for v in variants):
        return

    variants.append({
        "corrupted_core": corrupted,
        "typo_family": family,
        "typo_rule": rule,
        "priority": str(priority),
    })


def generate_core_variants(core: str) -> List[Dict[str, str]]:
    variants: List[Dict[str, str]] = []

    # 1. Apostrophe-related mistakes: very important for Uzbek Latin text
    apostrophes = ["'", "‘", "’", "ʻ", "ʼ", "`"]

    if any(a in core for a in apostrophes):
        no_ap = core
        for a in apostrophes:
            no_ap = no_ap.replace(a, "")
        add_variant(variants, core, no_ap, "apostrophe", "apostrophe_drop", 1)

        if "'" in core:
            add_variant(variants, core, core.replace("'", "`"), "apostrophe", "apostrophe_ascii_to_backtick", 1)
            add_variant(variants, core, core.replace("'", "‘"), "apostrophe", "apostrophe_ascii_to_curly_left", 2)
        if "‘" in core or "’" in core or "ʻ" in core or "ʼ" in core or "`" in core:
            norm = core
            for a in ["‘", "’", "ʻ", "ʼ", "`"]:
                norm = norm.replace(a, "'")
            add_variant(variants, core, norm, "apostrophe", "apostrophe_to_ascii", 2)

    # 2. Phonetic / digraph simplification
    replacements = [
        ("sh", "s", "phonetic_sh_to_s"),
        ("Sh", "S", "phonetic_Sh_to_S"),
        ("ch", "c", "phonetic_ch_to_c"),
        ("Ch", "C", "phonetic_Ch_to_C"),
        ("g'", "g", "apostrophe_g_drop"),
        ("G'", "G", "apostrophe_G_drop"),
        ("o'", "o", "apostrophe_o_drop"),
        ("O'", "O", "apostrophe_O_drop"),
    ]

    for old, new, rule in replacements:
        if old in core:
            family = "phonetic" if "phonetic" in rule else "apostrophe"
            add_variant(variants, core, core.replace(old, new, 1), family, rule, 2)

    # 3. Deletion: common typo/OCR style
    if len(core) >= 4:
        add_variant(variants, core, core[:-1], "deletion", "drop_last_char", 3)

    if len(core) >= 5:
        i = len(core) // 2
        add_variant(variants, core, core[:i] + core[i + 1:], "deletion", "drop_middle_char", 3)

    # 4. Insertion / duplicated char
    if len(core) >= 3:
        i = len(core) // 2
        add_variant(variants, core, core[:i] + core[i] + core[i:], "insertion", "duplicate_middle_char", 3)

    if len(core) >= 2 and len(core) <= 4:
        add_variant(variants, core, core + core[-1], "insertion", "duplicate_last_char_short_token", 4)

    # 5. Swap / transposition
    if len(core) >= 4:
        chars = list(core)
        i = max(1, len(chars) // 2 - 1)
        if i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            add_variant(variants, core, "".join(chars), "transposition", "swap_inner_chars", 3)

    # 6. Spacing / split word — useful but capped, because it changes whitespace tokenization
    if len(core) >= 7:
        i = len(core) // 2
        add_variant(variants, core, core[:i] + " " + core[i:], "spacing", "split_word_middle", 5)

    # 7. Final guaranteed fallback, only if nothing else works
    if not variants and len(core) >= 2:
        add_variant(variants, core, core + core[-1], "fallback", "duplicate_last_char_fallback", 99)

    # Sort by priority first, then stable rule name
    variants = sorted(variants, key=lambda x: (int(x["priority"]), x["typo_rule"]))

    # Weighted/capped selection:
    # keep highest-priority variants, but shuffle within same priority a little
    selected = []
    by_priority: Dict[str, List[Dict[str, str]]] = {}
    for v in variants:
        by_priority.setdefault(v["priority"], []).append(v)

    for pr in sorted(by_priority, key=lambda x: int(x)):
        bucket = by_priority[pr]
        random.shuffle(bucket)
        for v in bucket:
            selected.append(v)
            if len(selected) >= MAX_VARIANTS_PER_TOKEN:
                return selected

    return selected


def get_row_value(row: Dict[str, str], candidates: List[str]) -> str:
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            return row[c]
    return ""


def generate_for_sentence(row: Dict[str, str], sentence: str, sent_num: int):
    samples = []
    audit_rows = []

    tokens = list(TOKEN_RE.finditer(sentence))

    for token_index, m in enumerate(tokens):
        surface = m.group(0)
        start = m.start()
        end = m.end()

        prefix, core, suffix = split_token_surface(surface)
        eligible = is_eligible_core(core)

        if not eligible:
            audit_rows.append({
                "sentence_num": sent_num,
                "token_index": token_index,
                "token": surface,
                "core": core,
                "eligible": False,
                "generated_variants": 0,
                "reason": "ineligible_core",
            })
            continue

        variants = generate_core_variants(core)

        if not variants:
            audit_rows.append({
                "sentence_num": sent_num,
                "token_index": token_index,
                "token": surface,
                "core": core,
                "eligible": True,
                "generated_variants": 0,
                "reason": "no_valid_variant",
            })
            continue

        for variant_num, v in enumerate(variants, start=1):
            corrupted_core = v["corrupted_core"]
            corrupted_surface = prefix + corrupted_core + suffix
            noisy = sentence[:start] + corrupted_surface + sentence[end:]

            if noisy == sentence:
                continue

            row_id = f"v11-{sent_num:06d}-{token_index:03d}-{variant_num:02d}"

            samples.append({
                "row_id": row_id,
                "review_id": get_row_value(row, ["review_id", "seed_id", "id"]),
                "source_record_id": get_row_value(row, ["source_record_id"]),
                "sentence_id": get_row_value(row, ["sentence_id"]),
                "target": sentence,
                "input": noisy,
                "token_index": token_index,
                "token_start_char": start,
                "token_end_char": end,
                "clean_token": surface,
                "corrupted_token": corrupted_surface,
                "error_word": corrupted_surface,
                "correction": surface,
                "typo_family": v["typo_family"],
                "typo_rule": v["typo_rule"],
                "variant_rank": variant_num,
                "max_variants_per_token": MAX_VARIANTS_PER_TOKEN,
                "is_no_error": False,
                "synthetic_is_gold": False,
                "auto_replace_allowed": False,
                "review_required": True,
            })

        audit_rows.append({
            "sentence_num": sent_num,
            "token_index": token_index,
            "token": surface,
            "core": core,
            "eligible": True,
            "generated_variants": len(variants),
            "reason": "generated",
        })

    return samples, audit_rows


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    with INPUT_PATH.open("r", encoding="utf-8", newline="") as f:
        clean_rows = list(csv.DictReader(f))

    all_samples = []
    all_audit = []

    for sent_num, row in enumerate(clean_rows, start=1):
        sentence = get_row_value(row, ["sentence_text", "sentence", "target", "text"])
        if not sentence:
            continue

        samples, audit_rows = generate_for_sentence(row, sentence, sent_num)
        all_samples.extend(samples)
        all_audit.extend(audit_rows)

    sample_fields = [
        "row_id",
        "review_id",
        "source_record_id",
        "sentence_id",
        "target",
        "input",
        "token_index",
        "token_start_char",
        "token_end_char",
        "clean_token",
        "corrupted_token",
        "error_word",
        "correction",
        "typo_family",
        "typo_rule",
        "variant_rank",
        "max_variants_per_token",
        "is_no_error",
        "synthetic_is_gold",
        "auto_replace_allowed",
        "review_required",
    ]

    audit_fields = [
        "sentence_num",
        "token_index",
        "token",
        "core",
        "eligible",
        "generated_variants",
        "reason",
    ]

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sample_fields)
        w.writeheader()
        w.writerows(all_samples)

    with AUDIT_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=audit_fields)
        w.writeheader()
        w.writerows(all_audit)

    eligible = [r for r in all_audit if r["eligible"] is True]
    missing = [r for r in eligible if int(r["generated_variants"]) < 1]

    family_counts: Dict[str, int] = {}
    rule_counts: Dict[str, int] = {}

    for r in all_samples:
        family_counts[r["typo_family"]] = family_counts.get(r["typo_family"], 0) + 1
        rule_counts[r["typo_rule"]] = rule_counts.get(r["typo_rule"], 0) + 1

    summary_lines = []
    summary_lines.append("# Token Generator v11 Realistic Multivariant")
    summary_lines.append("")
    summary_lines.append("## Inputs")
    summary_lines.append(f"- clean targets: `{INPUT_PATH}`")
    summary_lines.append("")
    summary_lines.append("## Counts")
    summary_lines.append(f"- clean target sentences: `{len(clean_rows)}`")
    summary_lines.append(f"- eligible token positions: `{len(eligible)}`")
    summary_lines.append(f"- generated rows: `{len(all_samples)}`")
    summary_lines.append(f"- missing eligible positions: `{len(missing)}`")
    summary_lines.append(f"- max variants per token: `{MAX_VARIANTS_PER_TOKEN}`")
    summary_lines.append("")
    summary_lines.append("## Typo family counts")
    for k, v in sorted(family_counts.items(), key=lambda x: (-x[1], x[0])):
        summary_lines.append(f"- {k}: `{v}`")
    summary_lines.append("")
    summary_lines.append("## Typo rule counts")
    for k, v in sorted(rule_counts.items(), key=lambda x: (-x[1], x[0])):
        summary_lines.append(f"- {k}: `{v}`")
    summary_lines.append("")
    summary_lines.append("## Files")
    summary_lines.append(f"- output: `{OUTPUT_PATH}`")
    summary_lines.append(f"- audit: `{AUDIT_PATH}`")

    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))

    if len(missing) == 0:
        print("\nV11 COVERAGE STATUS: PASS")
    else:
        print("\nV11 COVERAGE STATUS: FAIL")


if __name__ == "__main__":
    main()
