from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1/"
    "strict_clean_seed_extractor_v1_3_lexical_gate/"
    "manual_clean_seed_bank_v1"
)

BATCH_DIR = BASE / "manual_review_batches_v1"

batch_paths = sorted(BATCH_DIR.glob("manual_seed_bank_batch_*.csv"))

if not batch_paths:
    raise SystemExit(f"No batch files found: {BATCH_DIR}")

rows = []

for path in batch_paths:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["batch_path"] = str(path)
    df["batch_file"] = path.name
    clean = df[df["manual_clean_label"] == "CLEAN_TARGET"].copy()
    rows.append(clean)

if not rows:
    raise SystemExit("No CLEAN_TARGET rows found.")

review_df = pd.concat(rows, ignore_index=True)

print("\nSecond-pass CLEAN_TARGET review")
print("=" * 80)
print("k = keep CLEAN_TARGET")
print("n = change to NOT_CLEAN_TARGET")
print("s = skip")
print("q = save and quit")
print("=" * 80)

for i, row in review_df.iterrows():
    batch_path = Path(row["batch_path"])
    review_id = row["review_id"]

    print("\n" + "=" * 100)
    print(f"Clean row: {i + 1}/{len(review_df)}")
    print(f"review_id: {review_id}")
    print(f"batch_file: {row['batch_file']}")
    print(f"part: {row['part_number']} | source_record_id: {row['source_record_id']}")
    print("-" * 100)
    print(row["sentence_text"])
    print("-" * 100)

    while True:
        ans = input("Decision [k/n/s/q]: ").strip().lower()

        if ans == "k":
            print("Kept as CLEAN_TARGET.")
            break

        if ans == "n":
            batch_df = pd.read_csv(batch_path, dtype=str).fillna("")
            mask = batch_df["review_id"].eq(review_id)

            if not mask.any():
                raise SystemExit(f"Could not find review_id {review_id} in {batch_path}")

            batch_df.loc[mask, "manual_clean_label"] = "NOT_CLEAN_TARGET"
            note = input("Reason note, Enter to skip: ").strip()
            if note:
                batch_df.loc[mask, "manual_notes"] = note
            elif batch_df.loc[mask, "manual_notes"].iloc[0] == "":
                batch_df.loc[mask, "manual_notes"] = "second_pass_reject"

            batch_df.to_csv(batch_path, index=False)
            print("Changed to NOT_CLEAN_TARGET and saved.")
            break

        if ans == "s":
            print("Skipped.")
            break

        if ans == "q":
            print("Stopped. Previous changes were already saved.")
            raise SystemExit(0)

        print("Use only: k, n, s, q")

print("\nDone second-pass review.")
