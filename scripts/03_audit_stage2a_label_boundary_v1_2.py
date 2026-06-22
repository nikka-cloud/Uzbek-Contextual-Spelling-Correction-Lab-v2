from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean


ROOT = Path(
    "outputs/03_sentence_inventory/"
    "global_sentence_health_review_v1"
)

OOF_PATH = (
    ROOT
    / "text_aware_router_v1_2_probe"
    / "stage2a_text_aware_oof_predictions_v1_2.csv"
)

OUTPUT_DIR = ROOT / "text_aware_router_v1_2_probe" / "stage2a_label_boundary_audit_v1_2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARY_PATH = OUTPUT_DIR / "stage2a_boundary_examples_v1_2.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def p(row: dict[str, str]) -> float:
    return float(row["p_direct_healthy"])


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    pos = (len(values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    frac = pos - low

    return values[low] * (1 - frac) + values[high] * frac


rows = read_csv(OOF_PATH)

if not rows:
    raise SystemExit("OOF file is empty.")

healthy = [row for row in rows if row["gold_route"] == "DIRECT_HEALTHY"]
repairable = [row for row in rows if row["gold_route"] == "REPAIRABLE_USABLE"]

if len(healthy) != 239:
    raise SystemExit(f"Expected 239 healthy rows, found {len(healthy)}")

if len(repairable) != 251:
    raise SystemExit(f"Expected 251 repairable rows, found {len(repairable)}")

healthy_scores = [p(row) for row in healthy]
repairable_scores = [p(row) for row in repairable]

thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]

print("=" * 120)
print("STAGE 2A LABEL BOUNDARY AUDIT V1.2")
print("=" * 120)

print("\nSCORE DISTRIBUTION")
print("-" * 120)

for name, scores in [
    ("DIRECT_HEALTHY", healthy_scores),
    ("REPAIRABLE_USABLE", repairable_scores),
]:
    print(f"\n{name}")
    print(f"rows: {len(scores)}")
    print(f"mean: {mean(scores):.4f}")
    for q in [0.00, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]:
        print(f"p{int(q * 100):>02}: {percentile(scores, q):.4f}")

print("\nTHRESHOLD BEHAVIOR")
print("-" * 120)

for threshold in thresholds:
    tp = sum(1 for row in healthy if p(row) >= threshold)
    fp = sum(1 for row in repairable if p(row) >= threshold)
    fn = len(healthy) - tp
    tn = len(repairable) - fp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / len(healthy) if healthy else 0.0
    repairable_leak = fp / len(repairable) if repairable else 0.0
    coverage = (tp + fp) / len(rows)

    print(
        f"thr={threshold:.2f} "
        f"precision={precision:>7.2%} "
        f"recall={recall:>7.2%} "
        f"coverage={coverage:>7.2%} "
        f"repairable_leak={repairable_leak:>7.2%} "
        f"tp={tp:>3} fp={fp:>3} fn={fn:>3} tn={tn:>3}"
    )

groups: list[tuple[str, list[dict[str, str]]]] = [
    (
        "A_TRUE_HEALTHY_HIGHEST_SCORE",
        sorted(healthy, key=p, reverse=True)[:30],
    ),
    (
        "B_REPAIRABLE_HIGHEST_SCORE_FALSE_DIRECT_RISK",
        sorted(repairable, key=p, reverse=True)[:30],
    ),
    (
        "C_HEALTHY_LOWEST_SCORE_FALSE_REPAIRABLE_RISK",
        sorted(healthy, key=p)[:30],
    ),
    (
        "D_REPAIRABLE_LOWEST_SCORE_CLEAR_REPAIRABLE",
        sorted(repairable, key=p)[:30],
    ),
    (
        "E_BOUNDARY_BAND_0_45_TO_0_65",
        sorted(
            [
                row for row in rows
                if 0.45 <= p(row) <= 0.65
            ],
            key=p,
            reverse=True,
        )[:60],
    ),
]

audit_rows: list[dict[str, str]] = []

for group_name, group_rows in groups:
    for row in group_rows:
        text = " ".join(row["sentence_text"].split())

        audit_rows.append(
            {
                "audit_group": group_name,
                "review_sample_id": row["review_sample_id"],
                "gold_route": row["gold_route"],
                "predicted_route": row["predicted_route"],
                "p_direct_healthy": row["p_direct_healthy"],
                "sentence_text": text,
            }
        )

write_csv(
    BOUNDARY_PATH,
    audit_rows,
    [
        "audit_group",
        "review_sample_id",
        "gold_route",
        "predicted_route",
        "p_direct_healthy",
        "sentence_text",
    ],
)

print("\nEXAMPLES: REPAIRABLE WITH HIGHEST DIRECT-HEALTHY SCORES")
print("-" * 120)

for row in sorted(repairable, key=p, reverse=True)[:10]:
    text = " ".join(row["sentence_text"].split())
    if len(text) > 240:
        text = text[:237] + "..."

    print(
        f"\n{row['review_sample_id']}\n"
        f"gold={row['gold_route']} "
        f"pred={row['predicted_route']} "
        f"p_direct={p(row):.4f}\n"
        f"{text}"
    )

print("\nEXAMPLES: HEALTHY WITH LOWEST DIRECT-HEALTHY SCORES")
print("-" * 120)

for row in sorted(healthy, key=p)[:10]:
    text = " ".join(row["sentence_text"].split())
    if len(text) > 240:
        text = text[:237] + "..."

    print(
        f"\n{row['review_sample_id']}\n"
        f"gold={row['gold_route']} "
        f"pred={row['predicted_route']} "
        f"p_direct={p(row):.4f}\n"
        f"{text}"
    )

print("\nOUTPUT FILE")
print("-" * 120)
print(BOUNDARY_PATH)

print("\nINTERPRETATION GUIDE")
print("-" * 120)
print("If HEALTHY and REPAIRABLE score percentiles strongly overlap, Stage 2A is not learnable enough yet.")
print("If repairable high-score examples contain tiny local defects, we need explicit defect detectors, not broader text n-grams.")
print("If many labels look inconsistent, we need label-boundary refinement before training a full router.")

print("\nSTATUS: READY_TO_REVIEW_STAGE2A_BOUNDARY_EXAMPLES")
