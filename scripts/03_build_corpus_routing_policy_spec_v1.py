#!/usr/bin/env python3

"""
Build the proposed Phase 3 corpus-routing policy specification.

This script does not route corpus rows.

It documents candidate routing rules that must later be evaluated
against the 700 manually reviewed viability examples.
"""

from __future__ import annotations

from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]

PART_REPORT = (
    ROOT
    / "outputs"
    / "03_sentence_inventory"
    / "corpus_viability_audit_v1"
    / "estimation_review_summary_v1"
    / "corpus_viability_by_part_v1.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "03_sentence_inventory"
    / "corpus_routing_policy_v1"
)

CSV_OUTPUT = OUTPUT_DIR / "routing_policy_spec_v1.csv"
MARKDOWN_OUTPUT = OUTPUT_DIR / "routing_policy_spec_v1.md"


RULES = [
    {
        "rule_id": "RP010",
        "priority": 10,
        "route": "REJECT_OR_ARCHIVE",
        "candidate_trigger": (
            "Predominantly foreign-language text, extreme mixed-script "
            "content, or clearly unrecoverable corruption. Exact ratios "
            "and thresholds must be selected using the labelled review set."
        ),
        "reason": (
            "Such material is unsuitable for ordinary Uzbek prose "
            "training and cannot be safely reconstructed."
        ),
        "decision_mode": "PROPOSED_AUTOMATIC_AFTER_EVALUATION",
        "automatic_route_assignment": "AFTER_EVALUATION",
        "manual_review_required": "SAMPLE_AUDIT",
        "eligible_for_clean_prose": False,
        "eligible_for_repair": False,
        "eligible_for_training": False,
        "allowed_downstream_use": (
            "Archive, error analysis, corpus-quality monitoring"
        ),
        "status": "PROPOSED_NOT_DEPLOYED",
    },
    {
        "rule_id": "RP020",
        "priority": 20,
        "route": "FORMAT_OR_LIST_QUARANTINE",
        "candidate_trigger": (
            "High-confidence list, table, heading, score, bibliography, "
            "metadata, URL, numbered-reference, or standalone-label "
            "structure. Numeric and structural thresholds remain to be "
            "evaluated."
        ),
        "reason": (
            "The segment may be valid text but is not ordinary prose and "
            "should not enter the first prose-detector dataset."
        ),
        "decision_mode": "PROPOSED_AUTOMATIC_AFTER_EVALUATION",
        "automatic_route_assignment": "AFTER_EVALUATION",
        "manual_review_required": "SAMPLE_AUDIT",
        "eligible_for_clean_prose": False,
        "eligible_for_repair": False,
        "eligible_for_training": False,
        "allowed_downstream_use": (
            "Formatting analysis, specialized future dataset"
        ),
        "status": "PROPOSED_NOT_DEPLOYED",
    },
    {
        "rule_id": "RP030",
        "priority": 30,
        "route": "BOUNDARY_REVIEW_QUARANTINE",
        "candidate_trigger": (
            "Boundary flags include LOWERCASE_AFTER_TERMINAL, "
            "INITIAL_BOUNDARY_PROTECTED, SHORT_TOKEN_BEFORE_PERIOD, "
            "VERY_LONG_SEGMENT, or "
            "LOWERCASE_AFTER_STRONG_PUNCTUATION_PROTECTED."
        ),
        "reason": (
            "The viability review showed that these flags often indicate "
            "bad splitting, fragments, lists, references, or merged "
            "sentences."
        ),
        "decision_mode": "PROPOSED_AUTOMATIC_AFTER_EVALUATION",
        "automatic_route_assignment": "AFTER_EVALUATION",
        "manual_review_required": "YES",
        "eligible_for_clean_prose": False,
        "eligible_for_repair": False,
        "eligible_for_training": False,
        "allowed_downstream_use": (
            "Segmentation repair, boundary-model evaluation"
        ),
        "status": "PROPOSED_NOT_DEPLOYED",
    },
    {
        "rule_id": "RP040",
        "priority": 40,
        "route": "MANUAL_REVIEW",
        "candidate_trigger": (
            "DOTTED_NAME_OR_DOMAIN_PROTECTED, conflicting warning flags, "
            "uncommon flag combinations, or uncertain structural signals "
            "that are not confidently prose or formatting."
        ),
        "reason": (
            "Current evidence is mixed and insufficient for reliable "
            "automatic routing."
        ),
        "decision_mode": "REVIEW_ONLY",
        "automatic_route_assignment": "NO",
        "manual_review_required": "YES",
        "eligible_for_clean_prose": False,
        "eligible_for_repair": False,
        "eligible_for_training": False,
        "allowed_downstream_use": (
            "Manual adjudication and rule-development evidence"
        ),
        "status": "PROPOSED_NOT_DEPLOYED",
    },
    {
        "rule_id": "RP050",
        "priority": 50,
        "route": "REPAIRABLE_PROSE_CANDIDATE",
        "candidate_trigger": (
            "Sentence-like Uzbek prose with no higher-priority blocker "
            "and parent flags AMBIGUOUS_APOSTROPHE_GAP or "
            "HIGH_CONFIDENCE_APOSTROPHE_GAP, or another validated "
            "recoverable-damage signal."
        ),
        "reason": (
            "Apostrophe-gap rows were usually usable, but they must not "
            "be treated as clean gold before explicit repair and "
            "validation."
        ),
        "decision_mode": "PROPOSED_AUTOMATIC_AFTER_EVALUATION",
        "automatic_route_assignment": "AFTER_EVALUATION",
        "manual_review_required": "YES_BEFORE_CLEAN_GOLD",
        "eligible_for_clean_prose": False,
        "eligible_for_repair": True,
        "eligible_for_training": False,
        "allowed_downstream_use": (
            "Repair pipeline, normalization experiments, hard test sets"
        ),
        "status": "PROPOSED_NOT_DEPLOYED",
    },
    {
        "rule_id": "RP060",
        "priority": 60,
        "route": "CLEAN_PROSE_CANDIDATE",
        "candidate_trigger": (
            "No higher-priority rule matched; sentence-like Uzbek prose; "
            "acceptable character and token length; safe boundary "
            "structure; no apostrophe-gap repair flag; and validated "
            "numeric, mixed-script, uppercase, URL, and list heuristics "
            "all pass."
        ),
        "reason": (
            "Normal segments were usually usable, but this route remains "
            "a candidate-clean lane rather than verified training gold."
        ),
        "decision_mode": "PROPOSED_AUTOMATIC_AFTER_EVALUATION",
        "automatic_route_assignment": "AFTER_EVALUATION",
        "manual_review_required": "SAMPLE_AUDIT",
        "eligible_for_clean_prose": True,
        "eligible_for_repair": False,
        "eligible_for_training": False,
        "allowed_downstream_use": (
            "Candidate clean-sentence inventory after deduplication and "
            "additional validation"
        ),
        "status": "PROPOSED_NOT_DEPLOYED",
    },
    {
        "rule_id": "RP999",
        "priority": 999,
        "route": "MANUAL_REVIEW",
        "candidate_trigger": (
            "Fallback when no earlier rule produces a confident route."
        ),
        "reason": (
            "Conservative abstention is safer than silently accepting or "
            "discarding an unexplained row."
        ),
        "decision_mode": "REVIEW_ONLY",
        "automatic_route_assignment": "NO",
        "manual_review_required": "YES",
        "eligible_for_clean_prose": False,
        "eligible_for_repair": False,
        "eligible_for_training": False,
        "allowed_downstream_use": (
            "Manual review and future routing-policy improvement"
        ),
        "status": "PROPOSED_NOT_DEPLOYED",
    },
]


POLICY_INVARIANTS = [
    "The original corpus and existing inventory files remain immutable.",
    "No route physically deletes source rows.",
    "part_number is used for monitoring and stratified auditing, not as a direct acceptance or rejection trigger.",
    "Manual human_health_label values exist only for the 700-row evaluation sample and must not be expected in the full corpus.",
    "QUOTE_OR_SPEECH_CONTINUATION alone is not a rejection signal.",
    "Apostrophe-gap flags alone are not rejection signals.",
    "No route is eligible_for_training in policy version 1.",
    "Thresholds for numeric ratio, script ratio, uppercase ratio, URL detection, and list structure must be validated against the 700 labelled examples.",
    "Higher-priority quarantine rules take precedence over repairable or clean-prose rules.",
    "Any unresolved or conflicting case must abstain into MANUAL_REVIEW.",
]


def read_part_report() -> list[dict[str, str]]:
    if not PART_REPORT.exists():
        raise FileNotFoundError(
            f"Required part report does not exist: {PART_REPORT}"
        )

    with PART_REPORT.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("The part report is empty.")

    required = {
        "part",
        "total",
        "potentially_usable",
        "potentially_usable_pct",
    }

    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"Part report is missing columns: {sorted(missing)}"
        )

    return rows


def validate_part_report(rows: list[dict[str, str]]) -> None:
    total = sum(int(row["total"]) for row in rows)

    unknown = [
        row
        for row in rows
        if row["part"].strip().upper() in {"", "UNKNOWN"}
    ]

    if total != 700:
        raise ValueError(
            f"Expected 700 reviewed rows, found {total}."
        )

    if unknown:
        raise ValueError(
            "UNKNOWN or empty part groups remain in the report."
        )


def validate_policy() -> None:
    rule_ids = [row["rule_id"] for row in RULES]
    priorities = [row["priority"] for row in RULES]

    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("Duplicate routing rule IDs found.")

    if len(priorities) != len(set(priorities)):
        raise ValueError("Duplicate routing priorities found.")

    required_routes = {
        "CLEAN_PROSE_CANDIDATE",
        "REPAIRABLE_PROSE_CANDIDATE",
        "FORMAT_OR_LIST_QUARANTINE",
        "BOUNDARY_REVIEW_QUARANTINE",
        "REJECT_OR_ARCHIVE",
        "MANUAL_REVIEW",
    }

    observed_routes = {row["route"] for row in RULES}

    missing_routes = required_routes - observed_routes
    if missing_routes:
        raise ValueError(
            f"Missing required routes: {sorted(missing_routes)}"
        )

    deployed = [
        row["rule_id"]
        for row in RULES
        if row["status"] != "PROPOSED_NOT_DEPLOYED"
    ]

    if deployed:
        raise ValueError(
            f"Unexpected deployed rules: {deployed}"
        )

    training_enabled = [
        row["rule_id"]
        for row in RULES
        if row["eligible_for_training"] is True
    ]

    if training_enabled:
        raise ValueError(
            "Policy v1 must not mark any route as training-ready."
        )


def write_csv() -> None:
    fieldnames = list(RULES[0])

    with CSV_OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(RULES)


def write_markdown(
    part_rows: list[dict[str, str]],
) -> None:
    sorted_parts = sorted(
        part_rows,
        key=lambda row: int(row["part"]),
    )

    lines = [
        "# Corpus Routing Policy Specification v1",
        "",
        "## Status",
        "",
        "`PROPOSED_NOT_DEPLOYED`",
        "",
        "This policy documents candidate routing rules. "
        "It does not yet route the full corpus.",
        "",
        "## Evidence base",
        "",
        "- Manual viability review rows: 700",
        "- Potentially usable rows: 567",
        "- Potentially usable rate: 81.00%",
        "- Corpus decision: `GO_WITH_CURATION`",
        "",
        "## Part-level diagnostic results",
        "",
        "| Part | Reviewed | Potentially usable | Usable rate |",
        "|---:|---:|---:|---:|",
    ]

    for row in sorted_parts:
        lines.append(
            f"| {row['part']} "
            f"| {row['total']} "
            f"| {row['potentially_usable']} "
            f"| {row['potentially_usable_pct']}% |"
        )

    lines.extend(
        [
            "",
            "Part number is diagnostic metadata only. "
            "It is not an acceptance or rejection rule.",
            "",
            "## Policy invariants",
            "",
        ]
    )

    for invariant in POLICY_INVARIANTS:
        lines.append(f"- {invariant}")

    lines.extend(
        [
            "",
            "## Proposed routing rules",
            "",
        ]
    )

    for rule in sorted(
        RULES,
        key=lambda item: item["priority"],
    ):
        lines.extend(
            [
                f"### {rule['rule_id']} — {rule['route']}",
                "",
                f"- **Priority:** {rule['priority']}",
                f"- **Trigger:** {rule['candidate_trigger']}",
                f"- **Reason:** {rule['reason']}",
                f"- **Decision mode:** {rule['decision_mode']}",
                f"- **Manual review:** "
                f"{rule['manual_review_required']}",
                f"- **Eligible for clean prose:** "
                f"{rule['eligible_for_clean_prose']}",
                f"- **Eligible for repair:** "
                f"{rule['eligible_for_repair']}",
                f"- **Eligible for training:** "
                f"{rule['eligible_for_training']}",
                f"- **Allowed downstream use:** "
                f"{rule['allowed_downstream_use']}",
                f"- **Status:** {rule['status']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Required next evaluation",
            "",
            "The proposed rules must be evaluated against the "
            "700 manually labelled examples before full-corpus use.",
            "",
            "Required measurements:",
            "",
            "- route coverage;",
            "- clean-lane precision;",
            "- usable-sentence recall;",
            "- false acceptance counts;",
            "- false rejection counts;",
            "- confusion counts by manual health label;",
            "- examples of every important routing error;",
            "- performance by corpus part and warning flag.",
            "",
        ]
    )

    MARKDOWN_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    part_rows = read_part_report()
    validate_part_report(part_rows)
    validate_policy()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv()
    write_markdown(part_rows)

    weak_parts = [
        {
            "part": row["part"],
            "usable_pct": float(
                row["potentially_usable_pct"]
            ),
        }
        for row in part_rows
        if float(row["potentially_usable_pct"]) <= 70.0
    ]

    print("--- ROUTING POLICY SPECIFICATION BUILT ---")
    print(f"input_part_report: {PART_REPORT.relative_to(ROOT)}")
    print(f"review_rows_validated: 700")
    print(f"routing_rules: {len(RULES)}")
    print(
        "routes: "
        + str(sorted({row["route"] for row in RULES}))
    )
    print(f"policy_status: PROPOSED_NOT_DEPLOYED")
    print(f"parts_at_or_below_70pct_usable: {weak_parts}")
    print()
    print("Created files:")
    print(f"- {CSV_OUTPUT.relative_to(ROOT)}")
    print(f"- {MARKDOWN_OUTPUT.relative_to(ROOT)}")
    print()
    print("ROUTING POLICY SPECIFICATION: PASS")


if __name__ == "__main__":
    main()
