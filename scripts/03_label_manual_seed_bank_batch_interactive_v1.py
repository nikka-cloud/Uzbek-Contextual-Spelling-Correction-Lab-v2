from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd


DEFAULT_PATH = Path(
    "outputs/03_sentence_inventory/global_sentence_health_review_v1/"
    "strict_clean_seed_extractor_v1_3_lexical_gate/"
    "manual_clean_seed_bank_v1/"
    "manual_review_batches_v1/"
    "manual_seed_bank_batch_001.csv"
)

path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH

if not path.exists():
    raise SystemExit(f"Missing batch file: {path}")

df = pd.read_csv(path, dtype=str).fillna("")

valid_labels = {"", "CLEAN_TARGET", "NOT_CLEAN_TARGET"}

bad_existing = sorted(set(df["manual_clean_label"]) - valid_labels)
if bad_existing:
    raise SystemExit(f"Invalid existing labels found: {bad_existing}")

print("\nManual seed-bank labeling")
print("=" * 80)
print("c = CLEAN_TARGET")
print("n = NOT_CLEAN_TARGET")
print("s = skip")
print("q = save and quit")
print("=" * 80)

for idx, row in df.iterrows():
    if row["manual_clean_label"].strip():
        continue

    print("\n" + "=" * 100)
    print(f"Row: {idx + 1}/{len(df)}")
    print(f"review_id: {row['review_id']}")
    print(f"source_preview: {row['source_preview']}")
    print(f"part: {row['part_number']} | source_record_id: {row['source_record_id']}")
    print("-" * 100)
    print(row["sentence_text"])
    print("-" * 100)

    while True:
        ans = input("Label [c/n/s/q]: ").strip().lower()

        if ans == "c":
            df.at[idx, "manual_clean_label"] = "CLEAN_TARGET"
            note = input("Optional note, Enter to skip: ").strip()
            df.at[idx, "manual_notes"] = note
            df.to_csv(path, index=False)
            print("Saved: CLEAN_TARGET")
            break

        if ans == "n":
            df.at[idx, "manual_clean_label"] = "NOT_CLEAN_TARGET"
            note = input("Optional note, Enter to skip: ").strip()
            df.at[idx, "manual_notes"] = note
            df.to_csv(path, index=False)
            print("Saved: NOT_CLEAN_TARGET")
            break

        if ans == "s":
            print("Skipped.")
            break

        if ans == "q":
            df.to_csv(path, index=False)
            print(f"\nSaved and quit: {path}")
            raise SystemExit(0)

        print("Use only: c, n, s, q")

df.to_csv(path, index=False)
print(f"\nDone. Saved labels to: {path}")
