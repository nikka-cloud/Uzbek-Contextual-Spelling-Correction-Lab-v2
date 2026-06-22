from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


RANDOM_SEED = 42
random.seed(RANDOM_SEED)

INPUT_PATH = Path(
    "outputs/03_sentence_inventory/global_sentence_health_review_v1/"
    "strict_clean_seed_extractor_v1_3_lexical_gate/manual_clean_seed_bank_v1/"
    "manual_labeled_outputs_v1/gold_seed_bank_v1.csv"
)

OUTPUT_PATH = Path("token_generator_v10_output.csv")
AUDIT_PATH = Path("token_generator_v10_generation_audit.csv")


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
    if not re.search(r"[A-Za-zÀ-ÿʻʼ‘’'`-]", core):
        return False
    if core.isdigit():
        return False
    return True


def apply_rule(core: str, rule: str) -> Optional[str]:
    if rule == "apostrophe_drop":
        if "'" in core:
            return core.replace("'", "")
        return None

    if rule == "apostrophe_ascii_to_backtick":
        if "'" in core:
            return core.replace("'", "`")
        return None

    if rule == "phonetic_sh":
        if "sh" in core:
            return core.replace("sh", "s", 1)
        if "Sh" in core:
            return core.replace("Sh", "S", 1)
        return None

    if rule == "phonetic_ch":
        if "ch" in core:
            return core.replace("ch", "c", 1)
        if "Ch" in core:
            return core.replace("Ch", "C", 1)
        return None

    if rule == "drop_last":
        if len(core) >= 4:
            return core[:-1]
        return None

    if rule == "drop_inner":
        if len(core) >= 5:
            i = len(core) // 2
            return core[:i] + core[i + 1:]
        return None

    if rule == "duplicate_inner":
        if len(core) >= 4:
            i = len(core) // 2
            return core[:i] + core[i] + core[i:]
        return None

    if rule == "swap_inner":
        if len(core) >= 4:
            chars = list(core)
            i = max(1, len(chars) // 2 - 1)
            if i + 1 < len(chars):
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                return "".join(chars)
        return None

    if rule == "split_word":
        if len(core) >= 7:
            i = len(core) // 2
            return core[:i] + " " + core[i:]
        return None

    if rule == "duplicate_last_char_fallback":
        if len(core) >= 2:
            return core + core[-1]
        return None

    return None


RULE_ORDER = [
    "apostrophe_drop",
    "apostrophe_ascii_to_backtick",
    "phonetic_sh",
    "phonetic_ch",
    "drop_last",
    "drop_inner",
    "duplicate_inner",
    "swap_inner",
    "split_word",
    "duplicate_last_char_fallback",
]


RULE_FAMILY = {
    "apostrophe_drop": "apostrophe",
    "apostrophe_ascii_to_backtick": "apostrophe",
    "phonetic_sh": "phonetic",
    "phonetic_ch": "phonetic",
    "drop_last": "deletion",
    "drop_inner": "deletion",
    "duplicate_inner": "insertion",
    "swap_inner": "transposition",
    "split_word": "spacing",
    "duplicate_last_char_fallback": "insertion_fallback",
}


def choose_valid_corruption(core: str) -> Tuple[str, str, str]:
    valid = []

    for rule in RULE_ORDER:
        corrupted = apply_rule(core, rule)
        if corrupted and corrupted != core:
            valid.append((rule, RULE_FAMILY[rule], corrupted))

    if not valid:
        raise ValueError(f"No valid corruption rule for eligible token core={core!r}")

    return random.choice(valid)


def get_row_value(row: Dict[str, str], candidates: List[str]) -> str:
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            return row[c]
    return ""


def generate_for_sentence(row: Dict[str, str], sentence: str, sent_num: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
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
                "generated": False,
                "reason": "ineligible_core",
            })
            continue

        try:
            typo_rule, typo_family, corrupted_core = choose_valid_corruption(core)
        except ValueError:
            audit_rows.append({
                "sentence_num": sent_num,
                "token_index": token_index,
                "token": surface,
                "core": core,
                "eligible": True,
                "generated": False,
                "reason": "no_valid_rule",
            })
            continue

        corrupted_surface = prefix + corrupted_core + suffix
        noisy = sentence[:start] + corrupted_surface + sentence[end:]

        if noisy == sentence:
            audit_rows.append({
                "sentence_num": sent_num,
                "token_index": token_index,
                "token": surface,
                "core": core,
                "eligible": True,
                "generated": False,
                "reason": "no_surface_change",
            })
            continue

        row_id = f"v10-{sent_num:06d}-{token_index:03d}"

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
            "typo_family": typo_family,
            "typo_rule": typo_rule,
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
            "generated": True,
            "reason": "generated",
        })

    return samples, audit_rows


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    all_samples = []
    all_audit = []

    with INPUT_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for sent_num, row in enumerate(rows, start=1):
        sentence = get_row_value(row, ["sentence_text", "sentence", "target", "text"])
        if not sentence:
            continue

        samples, audit_rows = generate_for_sentence(row, sentence, sent_num)
        all_samples.extend(samples)
        all_audit.extend(audit_rows)

    fieldnames = [
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
        "is_no_error",
        "synthetic_is_gold",
        "auto_replace_allowed",
        "review_required",
    ]

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_samples)

    audit_fieldnames = [
        "sentence_num",
        "token_index",
        "token",
        "core",
        "eligible",
        "generated",
        "reason",
    ]

    with AUDIT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=audit_fieldnames)
        writer.writeheader()
        writer.writerows(all_audit)

    total_eligible = sum(1 for r in all_audit if r["eligible"] is True)
    total_generated = len(all_samples)
    missing = total_eligible - total_generated

    print("INPUT:", INPUT_PATH)
    print("OUTPUT:", OUTPUT_PATH)
    print("AUDIT:", AUDIT_PATH)
    print("SENTENCES:", len(rows))
    print("ELIGIBLE TOKEN POSITIONS:", total_eligible)
    print("GENERATED ROWS:", total_generated)
    print("MISSING ELIGIBLE POSITIONS:", missing)

    if missing == 0:
        print("COVERAGE STATUS: PASS")
    else:
        print("COVERAGE STATUS: FAIL")


if __name__ == "__main__":
    main()
