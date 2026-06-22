from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


INPUT_PATH = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1/"
    "text_health_routing_policy_v1/"
    "calibration_routing_predictions_v1.csv"
)

OUTPUT_DIR = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1/"
    "text_health_routing_policy_v1/"
    "safety_error_audit_v1"
)

AUDIT_PATH = OUTPUT_DIR / "routing_safety_errors_v1.csv"
REASON_SUMMARY_PATH = OUTPUT_DIR / "routing_safety_error_reasons_v1.csv"


def resolve_column(
    fieldnames: list[str],
    candidates: list[str],
    *,
    required: bool = True,
) -> str | None:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate

    if required:
        raise SystemExit(
            "Could not resolve required column.\n"
            f"Tried: {candidates}\n"
            f"Available columns: {fieldnames}"
        )

    return None


def clean_route(value: str) -> str:
    return (value or "").strip().upper()


if not INPUT_PATH.exists():
    raise SystemExit(f"Missing input file: {INPUT_PATH}")

with INPUT_PATH.open("r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)

if not rows:
    raise SystemExit("Prediction file contains no rows.")

gold_column = resolve_column(
    fieldnames,
    [
        "reference_route",
        "gold_route",
        "reference_lane",
        "gold_lane",
        "manual_route",
        "reference_health_lane",
        "target_route",
    ],
)

predicted_column = resolve_column(
    fieldnames,
    [
        "predicted_route",
        "prediction_route",
        "routed_lane",
        "predicted_lane",
        "route_prediction",
    ],
)

reason_column = resolve_column(
    fieldnames,
    [
        "routing_reason",
        "primary_routing_reason",
        "prediction_reason",
        "route_reason",
    ],
    required=False,
)

text_column = resolve_column(
    fieldnames,
    [
        "sentence_text",
        "text",
        "current_sentence",
        "input_text",
    ],
    required=False,
)

id_column = resolve_column(
    fieldnames,
    [
        "review_sample_id",
        "sentence_id",
        "sample_id",
    ],
    required=False,
)

VALID_ROUTES = {
    "DIRECT_HEALTHY",
    "REPAIRABLE_USABLE",
    "QUARANTINE",
    "REJECT",
}

USABLE_ROUTES = {
    "DIRECT_HEALTHY",
    "REPAIRABLE_USABLE",
}

NONUSABLE_ROUTES = {
    "QUARANTINE",
    "REJECT",
}

issue_counter: Counter[str] = Counter()
transition_counter: Counter[tuple[str, str]] = Counter()
reason_counter: Counter[tuple[str, str]] = Counter()

audited_rows: list[dict[str, str]] = []

for position, row in enumerate(rows, start=1):
    gold = clean_route(row.get(gold_column, ""))
    predicted = clean_route(row.get(predicted_column, ""))

    if gold not in VALID_ROUTES:
        raise SystemExit(
            f"Invalid gold route at prediction row {position}: {gold!r}"
        )

    if predicted not in VALID_ROUTES:
        raise SystemExit(
            f"Invalid predicted route at prediction row {position}: "
            f"{predicted!r}"
        )

    if gold == predicted:
        continue

    reason = (
        (row.get(reason_column, "") or "").strip()
        if reason_column
        else "UNKNOWN_REASON"
    )

    flags: list[str] = []

    if gold in NONUSABLE_ROUTES and predicted in USABLE_ROUTES:
        flags.append("CRITICAL_NONUSABLE_LEAK_TO_USABLE")

    if predicted == "DIRECT_HEALTHY" and gold != "DIRECT_HEALTHY":
        flags.append("DIRECT_HEALTHY_CONTAMINATION")

    if gold in USABLE_ROUTES and predicted in NONUSABLE_ROUTES:
        flags.append("USABLE_SENTENCE_OVERQUARANTINED")

    if gold == "REJECT" and predicted != "REJECT":
        flags.append("MISSED_REJECT")

    if gold == "QUARANTINE" and predicted == "DIRECT_HEALTHY":
        flags.append("QUARANTINE_SENT_DIRECTLY_TO_HEALTHY")

    if not flags:
        flags.append("OTHER_ROUTE_MISMATCH")

    if "CRITICAL_NONUSABLE_LEAK_TO_USABLE" in flags:
        priority = "1_CRITICAL"
    elif (
        "DIRECT_HEALTHY_CONTAMINATION" in flags
        or "MISSED_REJECT" in flags
    ):
        priority = "2_HIGH"
    elif "USABLE_SENTENCE_OVERQUARANTINED" in flags:
        priority = "3_MEDIUM"
    else:
        priority = "4_LOW"

    transition_counter[(gold, predicted)] += 1

    for flag in flags:
        issue_counter[flag] += 1
        reason_counter[(flag, reason)] += 1

    audited = {
        "audit_priority": priority,
        "audit_flags": "|".join(flags),
        "audit_gold_route": gold,
        "audit_predicted_route": predicted,
        "audit_routing_reason": reason,
    }

    audited.update(row)
    audited_rows.append(audited)

audited_rows.sort(
    key=lambda row: (
        row["audit_priority"],
        row["audit_flags"],
        row.get(id_column, "") if id_column else "",
    )
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

audit_fields = [
    "audit_priority",
    "audit_flags",
    "audit_gold_route",
    "audit_predicted_route",
    "audit_routing_reason",
]

for field in fieldnames:
    if field not in audit_fields:
        audit_fields.append(field)

with AUDIT_PATH.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=audit_fields)
    writer.writeheader()
    writer.writerows(audited_rows)

reason_rows = [
    {
        "audit_flag": flag,
        "routing_reason": reason,
        "error_count": count,
    }
    for (flag, reason), count in sorted(
        reason_counter.items(),
        key=lambda item: (
            item[0][0],
            -item[1],
            item[0][1],
        ),
    )
]

with REASON_SUMMARY_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "audit_flag",
            "routing_reason",
            "error_count",
        ],
    )
    writer.writeheader()
    writer.writerows(reason_rows)

print("=" * 118)
print("TEXT HEALTH ROUTING SAFETY-ERROR AUDIT — CALIBRATION ONLY")
print("=" * 118)

print("\nRESOLVED SCHEMA")
print("-" * 118)
print(f"gold_route       : {gold_column}")
print(f"predicted_route  : {predicted_column}")
print(f"routing_reason   : {reason_column or 'NOT AVAILABLE'}")
print(f"text             : {text_column or 'NOT AVAILABLE'}")
print(f"id               : {id_column or 'NOT AVAILABLE'}")

print("\nAUDIT COUNTS")
print("-" * 118)
print(f"prediction_rows               : {len(rows)}")
print(f"misrouted_rows                : {len(audited_rows)}")

for issue, count in issue_counter.most_common():
    print(f"{issue:<48}: {count}")

print("\nROUTE TRANSITIONS")
print("-" * 118)

for (gold, predicted), count in sorted(
    transition_counter.items(),
    key=lambda item: -item[1],
):
    print(
        f"{gold:<22} -> {predicted:<22}: {count}"
    )

print("\nTOP ERROR REASONS")
print("-" * 118)

for (flag, reason), count in sorted(
    reason_counter.items(),
    key=lambda item: -item[1],
)[:30]:
    print(
        f"{flag:<44} | {reason:<48} | rows={count:>3}"
    )

sample_groups = [
    "CRITICAL_NONUSABLE_LEAK_TO_USABLE",
    "DIRECT_HEALTHY_CONTAMINATION",
    "MISSED_REJECT",
    "USABLE_SENTENCE_OVERQUARANTINED",
]

print("\nERROR EXAMPLES")
print("=" * 118)

for target_flag in sample_groups:
    matches = [
        row
        for row in audited_rows
        if target_flag in row["audit_flags"].split("|")
    ]

    print(f"\n{target_flag}")
    print("-" * 118)
    print(f"matched_rows: {len(matches)}")

    for row in matches[:5]:
        sample_id = (
            row.get(id_column, "UNKNOWN")
            if id_column
            else "UNKNOWN"
        )
        text = (
            row.get(text_column, "")
            if text_column
            else ""
        )
        text = " ".join((text or "").split())

        if len(text) > 300:
            text = text[:297] + "..."

        print(
            f"\n{sample_id}\n"
            f"gold={row['audit_gold_route']} | "
            f"predicted={row['audit_predicted_route']} | "
            f"reason={row['audit_routing_reason']}\n"
            f"{text or '[TEXT NOT AVAILABLE]'}"
        )

print("\nOUTPUT FILES")
print("-" * 118)
print(AUDIT_PATH)
print(REASON_SUMMARY_PATH)

print("\nmaster_csv_modified: NO")
print("routing_policy_modified: NO")
print("validation_rows_used: NO")
print("validation_labels_exposed: NO")
print("full_corpus_filtering_applied: NO")
print("SAFETY-ERROR AUDIT: PASS")
print("STATUS: READY_TO_TUNE_TEXT_HEALTH_ROUTING_POLICY_V1_1")
