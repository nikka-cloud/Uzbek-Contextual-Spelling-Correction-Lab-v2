#!/usr/bin/env python3
"""
Exact duplicate analysis for hash shards produced by audit v2.2.

The script first discovers repeated SHA-256 + character-count identities within
individual hash shards. It then streams the immutable corpus again using the
pure-Python ijson backend and compares the candidate texts exactly. Therefore,
a hash collision cannot be reported as an exact duplicate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import os
import shutil
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

SCRIPT_VERSION = "2.2"
LEGACY_SCRIPT_NAME = "01_inspect_raw_corpus_v2_1.py"
HASH_RECORD_STRUCT = struct.Struct(">32sQI")
HASH_RECORD_SIZE = HASH_RECORD_STRUCT.size
DEFAULT_INPUT = Path("/mnt/uzbekvoice_storage/text_data/normalized.json")


def load_v2_1_module() -> Any:
    path = Path(__file__).resolve().with_name(LEGACY_SCRIPT_NAME)
    if not path.is_file():
        raise SystemExit(f"ERROR: required helper script is missing: {path}")
    name = "uzbek_contextual_duplicate_v2_1_helpers"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"ERROR: could not load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V21 = load_v2_1_module()

try:
    from ijson.backends import python as IJSON_BACKEND
except ImportError:
    IJSON_BACKEND = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify exact duplicates from v2.2 compact hash shards."
    )
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-chars", type=int, default=320)
    parser.add_argument("--progress-every", type=int, default=500_000)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    return parser.parse_args()


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise SystemExit(f"ERROR: output path is not a directory: {path}")
        if any(path.iterdir()):
            if not overwrite:
                raise SystemExit(f"ERROR: output directory is not empty: {path}")
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "_DUPLICATE_ANALYSIS_INCOMPLETE").write_text(
        "Duplicate analysis has not completed successfully yet.\n",
        encoding="utf-8",
    )


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def iter_hash_records(path: Path) -> Iterator[Tuple[bytes, int, int]]:
    size = path.stat().st_size
    if size % HASH_RECORD_SIZE:
        raise RuntimeError(
            f"invalid hash shard size for {path}: {size} is not divisible by {HASH_RECORD_SIZE}"
        )
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_RECORD_SIZE)
            if not chunk:
                break
            if len(chunk) != HASH_RECORD_SIZE:
                raise RuntimeError(f"truncated hash record in {path}")
            yield HASH_RECORD_STRUCT.unpack(chunk)


def markdown_report(summary: Mapping[str, Any]) -> str:
    rows = [
        ("candidate_hash_groups", summary["candidate_hash_groups"]),
        ("candidate_record_count", summary["candidate_record_count"]),
        ("exact_duplicate_groups", summary["exact_duplicate_groups"]),
        ("duplicate_occurrences_after_first", summary["duplicate_occurrences_after_first"]),
        ("hash_collision_groups", summary["hash_collision_groups"]),
        ("candidate_records_verified", summary["candidate_records_verified"]),
        ("immutability_check", summary["immutability_check"]),
    ]
    table = "\n".join(["| Metric | Value |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in rows])
    return (
        "# Phase 1 Exact Duplicate Analysis — v2.2\n\n"
        "Duplicate candidates were discovered from SHA-256 + character-count shards "
        "and then verified by exact raw-text equality in a second corpus stream.\n\n"
        + table
        + "\n\nThe raw corpus was not modified.\n"
    )


def main() -> None:
    args = parse_args()
    if IJSON_BACKEND is None:
        raise SystemExit("ERROR: the pure-Python ijson backend is required.")

    audit_dir = args.audit_dir.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not (audit_dir / "_AUDIT_COMPLETE").is_file():
        raise SystemExit(f"ERROR: audit is not complete: {audit_dir}")
    if not input_path.is_file():
        raise SystemExit(f"ERROR: corpus not found: {input_path}")
    prepare_output(output_dir, args.overwrite_output_dir)

    started = time.time()
    audit_summary = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((audit_dir / "hash_manifest.json").read_text(encoding="utf-8"))
    expected_records = int(audit_summary["records_scanned"])
    expected_string_records = int(audit_summary["record_structure"]["string_text_records"])
    if int(manifest["total_records"]) != expected_string_records:
        raise RuntimeError("hash manifest total does not match audit string-text count")
    if int(manifest["record_size_bytes"]) != HASH_RECORD_SIZE:
        raise RuntimeError("unsupported hash record size")

    atomic_json(
        output_dir / "run_configuration.json",
        {
            "script_version": SCRIPT_VERSION,
            "audit_dir": str(audit_dir),
            "input": str(input_path),
            "output_dir": str(output_dir),
            "records_to_verify": expected_records,
            "parser_backend": "ijson.backends.python",
        },
    )

    candidate_groups: Dict[int, Dict[str, Any]] = {}
    target_rows: List[Tuple[int, int, bytes, int]] = []
    candidate_group_id = 0
    shard_record_total = 0

    for file_info in manifest["files"]:
        shard_path = audit_dir / "hash_shards" / file_info["filename"]
        records = list(iter_hash_records(shard_path))
        shard_record_total += len(records)
        if len(records) != int(file_info["record_count"]):
            raise RuntimeError(f"record-count mismatch in {shard_path}")
        records.sort(key=lambda row: (row[0], row[2], row[1]))
        for (digest, character_count), group_iter in itertools.groupby(
            records, key=lambda row: (row[0], row[2])
        ):
            group = list(group_iter)
            if len(group) < 2:
                continue
            ids = sorted(row[1] for row in group)
            candidate_group_id += 1
            candidate_groups[candidate_group_id] = {
                "digest": digest,
                "character_count": character_count,
                "record_ids": ids,
                "texts": [],
            }
            for record_id in ids:
                target_rows.append((record_id, candidate_group_id, digest, character_count))

    if shard_record_total != expected_string_records:
        raise RuntimeError(
            f"shard total mismatch: observed={shard_record_total}, expected={expected_string_records}"
        )

    target_rows.sort(key=lambda row: row[0])
    duplicate_ids = [row[0] for row in target_rows]
    if len(duplicate_ids) != len(set(duplicate_ids)):
        raise RuntimeError("a source_record_id appeared in more than one hash candidate group")

    with (output_dir / "hash_duplicate_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_group_id", "sha256", "character_count", "candidate_count", "source_record_ids"],
        )
        writer.writeheader()
        for group_id in sorted(candidate_groups):
            group = candidate_groups[group_id]
            writer.writerow(
                {
                    "candidate_group_id": group_id,
                    "sha256": group["digest"].hex(),
                    "character_count": group["character_count"],
                    "candidate_count": len(group["record_ids"]),
                    "source_record_ids": ";".join(map(str, group["record_ids"])),
                }
            )

    pre_fingerprint = V21.fingerprint_file(input_path)
    target_index = 0
    verified = 0
    with input_path.open("rb") as corpus_handle:
        items: Iterable[Any] = itertools.islice(
            IJSON_BACKEND.items(corpus_handle, "item"), expected_records
        )
        for record_id, record in enumerate(items):
            while target_index < len(target_rows) and target_rows[target_index][0] < record_id:
                raise RuntimeError(
                    f"candidate record was not encountered: {target_rows[target_index][0]}"
                )
            if target_index < len(target_rows) and target_rows[target_index][0] == record_id:
                _, group_id, expected_digest, expected_char_count = target_rows[target_index]
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    raise RuntimeError(f"candidate record {record_id} is not a string text record")
                text = record["text"]
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                if digest != expected_digest or len(text) != expected_char_count:
                    raise RuntimeError(f"candidate identity mismatch at record {record_id}")
                candidate_groups[group_id]["texts"].append((record_id, text))
                target_index += 1
                verified += 1
            if args.progress_every and (record_id + 1) % args.progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"[duplicate v{SCRIPT_VERSION}] records={record_id + 1:,}/{expected_records:,} "
                    f"verified_candidates={verified:,}/{len(target_rows):,} "
                    f"rate={(record_id + 1) / max(elapsed, 1e-9):,.1f} records/s",
                    file=sys.stderr,
                    flush=True,
                )

    if target_index != len(target_rows):
        raise RuntimeError(
            f"not all candidate records were verified: {target_index}/{len(target_rows)}"
        )

    post_fingerprint = V21.fingerprint_file(input_path)
    if pre_fingerprint != post_fingerprint:
        raise RuntimeError("raw corpus fingerprint changed during duplicate verification")

    exact_rows: List[Dict[str, Any]] = []
    hash_collision_groups = 0
    duplicate_group_id = 0
    for candidate_id in sorted(candidate_groups):
        group = candidate_groups[candidate_id]
        by_text: Dict[str, List[int]] = defaultdict(list)
        for record_id, text in group["texts"]:
            by_text[text].append(record_id)
        exact_subgroups = [
            (text, sorted(ids)) for text, ids in by_text.items() if len(ids) > 1
        ]
        if len(by_text) > 1:
            hash_collision_groups += 1
        for text, ids in sorted(exact_subgroups, key=lambda item: item[1][0]):
            duplicate_group_id += 1
            exact_rows.append(
                {
                    "duplicate_group_id": duplicate_group_id,
                    "first_record_id": ids[0],
                    "occurrence_count": len(ids),
                    "duplicate_occurrences_after_first": len(ids) - 1,
                    "character_count": len(text),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "record_ids": ";".join(map(str, ids)),
                    "text_preview": V21.visible_preview(text, args.preview_chars),
                }
            )

    with (output_dir / "duplicate_groups.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "duplicate_group_id", "first_record_id", "occurrence_count",
            "duplicate_occurrences_after_first", "character_count", "sha256",
            "record_ids", "text_preview",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(exact_rows)

    summary = {
        "script_version": SCRIPT_VERSION,
        "audit_directory": str(audit_dir),
        "input_path": str(input_path),
        "records_scanned_in_source_audit": expected_records,
        "hash_inventory_records": expected_string_records,
        "candidate_hash_groups": len(candidate_groups),
        "candidate_record_count": len(target_rows),
        "candidate_records_verified": verified,
        "exact_duplicate_groups": len(exact_rows),
        "duplicate_occurrences_after_first": sum(
            row["duplicate_occurrences_after_first"] for row in exact_rows
        ),
        "hash_collision_groups": hash_collision_groups,
        "immutability_check": "PASS",
        "parser": {"library": "ijson", "backend": "python", "module": "ijson.backends.python"},
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(markdown_report(summary), encoding="utf-8")
    incomplete = output_dir / "_DUPLICATE_ANALYSIS_INCOMPLETE"
    if incomplete.exists():
        incomplete.unlink()
    (output_dir / "_DUPLICATE_ANALYSIS_COMPLETE").write_text(
        "Exact duplicate analysis completed successfully. Immutability check: PASS.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # A normal Python exception receives an explicit failure marker. A native crash
        # leaves the incomplete marker behind, which is also treated as failure.
        try:
            args = parse_args()
            output_dir = args.output_dir.expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "_DUPLICATE_ANALYSIS_FAILED.txt").write_text(
                "Duplicate analysis failed. Do not approve partial outputs.\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        raise
