from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path


RANDOM_SEED = 42
random.seed(RANDOM_SEED)

ROOT = Path("outputs/05_synthetic_typo_generation/v10_deterministic_sweep")

V10_PATH = ROOT / "token_generator_v10_output.csv"
CLEAN_PATH = ROOT / "frozen_clean_targets_v1.csv"

OUT_DIR = Path("outputs/07_training_views/detector_training_view_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_OUT = OUT_DIR / "detector_training_view_all_v1.csv"
TRAIN_OUT = OUT_DIR / "detector_train_v1.csv"
DEV_OUT = OUT_DIR / "detector_dev_v1.csv"
TEST_OUT = OUT_DIR / "detector_test_v1.csv"
SUMMARY_OUT = OUT_DIR / "detector_training_view_summary_v1.md"


def stable_group_id(row: dict) -> str:
    for key in ["review_id", "sentence_id", "source_record_id"]:
        val = row.get(key, "")
        if val:
            return str(val)
    return hashlib.sha256(row["target"].encode("utf-8")).hexdigest()[:16]


def split_groups(group_ids: list[str]) -> dict[str, str]:
    ids = sorted(set(group_ids))
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * 0.80)
    n_dev = int(n * 0.10)

    split = {}
    for gid in ids[:n_train]:
        split[gid] = "train"
    for gid in ids[n_train:n_train + n_dev]:
        split[gid] = "dev"
    for gid in ids[n_train + n_dev:]:
        split[gid] = "test"

    return split


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    if not V10_PATH.exists():
        raise FileNotFoundError(V10_PATH)
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(CLEAN_PATH)

    v10_rows = read_csv(V10_PATH)
    clean_rows = read_csv(CLEAN_PATH)

    group_ids = []

    for r in v10_rows:
        group_ids.append(stable_group_id(r))

    for r in clean_rows:
        target = r.get("sentence_text") or r.get("target") or r.get("sentence") or r.get("text")
        if not target:
            continue
        group_ids.append(
            r.get("review_id")
            or r.get("sentence_id")
            or r.get("source_record_id")
            or hashlib.sha256(target.encode("utf-8")).hexdigest()[:16]
        )

    split_map = split_groups(group_ids)

    out_rows = []

    # CORRECT_THIS rows from v10
    for r in v10_rows:
        gid = stable_group_id(r)
        out_rows.append({
            "row_id": r["row_id"],
            "group_id": gid,
            "split": split_map[gid],
            "task": "detector",
            "label": "CORRECT_THIS",
            "input": r["input"],
            "target": r["target"],
            "token_index": r["token_index"],
            "clean_token": r["clean_token"],
            "observed_token": r["corrupted_token"],
            "correction": r["correction"],
            "typo_family": r["typo_family"],
            "typo_rule": r["typo_rule"],
            "source": "v10_deterministic_synthetic",
            "is_no_error": "false",
            "synthetic_is_gold": "false",
            "auto_replace_allowed": "false",
            "review_required": "true",
        })

    # KEEP rows from frozen clean targets
    keep_i = 0
    for r in clean_rows:
        target = r.get("sentence_text") or r.get("target") or r.get("sentence") or r.get("text")
        if not target:
            continue

        gid = (
            r.get("review_id")
            or r.get("sentence_id")
            or r.get("source_record_id")
            or hashlib.sha256(target.encode("utf-8")).hexdigest()[:16]
        )

        keep_i += 1
        out_rows.append({
            "row_id": f"keep-v1-{keep_i:06d}",
            "group_id": gid,
            "split": split_map[gid],
            "task": "detector",
            "label": "KEEP",
            "input": target,
            "target": target,
            "token_index": "",
            "clean_token": "",
            "observed_token": "",
            "correction": "",
            "typo_family": "NO_ERROR",
            "typo_rule": "NO_ERROR",
            "source": "frozen_clean_target",
            "is_no_error": "true",
            "synthetic_is_gold": "false",
            "auto_replace_allowed": "false",
            "review_required": "true",
        })

    fieldnames = list(out_rows[0].keys())

    def write_rows(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    write_rows(ALL_OUT, out_rows)
    write_rows(TRAIN_OUT, [r for r in out_rows if r["split"] == "train"])
    write_rows(DEV_OUT, [r for r in out_rows if r["split"] == "dev"])
    write_rows(TEST_OUT, [r for r in out_rows if r["split"] == "test"])

    counts = {}
    for r in out_rows:
        key = (r["split"], r["label"])
        counts[key] = counts.get(key, 0) + 1

    groups_by_split = {}
    for gid, split in split_map.items():
        groups_by_split[split] = groups_by_split.get(split, 0) + 1

    summary = []
    summary.append("# Detector Training View v1")
    summary.append("")
    summary.append("## Inputs")
    summary.append(f"- v10 synthetic: `{V10_PATH}`")
    summary.append(f"- frozen clean targets: `{CLEAN_PATH}`")
    summary.append("")
    summary.append("## Output rows")
    summary.append(f"- total rows: `{len(out_rows)}`")
    summary.append(f"- CORRECT_THIS rows: `{sum(1 for r in out_rows if r['label'] == 'CORRECT_THIS')}`")
    summary.append(f"- KEEP rows: `{sum(1 for r in out_rows if r['label'] == 'KEEP')}`")
    summary.append("")
    summary.append("## Groups by split")
    for split in ["train", "dev", "test"]:
        summary.append(f"- {split}: `{groups_by_split.get(split, 0)}` groups")
    summary.append("")
    summary.append("## Rows by split/label")
    for split in ["train", "dev", "test"]:
        for label in ["CORRECT_THIS", "KEEP"]:
            summary.append(f"- {split} / {label}: `{counts.get((split, label), 0)}`")
    summary.append("")
    summary.append("## Files")
    summary.append(f"- all: `{ALL_OUT}`")
    summary.append(f"- train: `{TRAIN_OUT}`")
    summary.append(f"- dev: `{DEV_OUT}`")
    summary.append(f"- test: `{TEST_OUT}`")

    SUMMARY_OUT.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n".join(summary))


if __name__ == "__main__":
    main()
