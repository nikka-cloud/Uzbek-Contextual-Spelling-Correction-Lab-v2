# Phase 2 — Safe derived-corpus normalization

## Objective

Create a mechanically consistent, provenance-preserving derived corpus without
performing linguistic correction.

The Phase 2 output becomes the official document-level source for sentence
segmentation in Phase 3.

## Input

- Path: `/mnt/uzbekvoice_storage/text_data/normalized.json`
- Format: top-level JSON array
- Records: 13,884,795
- Raw size: approximately 13.85 GB

## Approved operation

Version 1 applies only:

- `TRIM_ONE_TRAILING_ASCII_SPACE`

It does not repair spelling, apostrophe gaps, punctuation, grammar, merged or
split words, source credits, or sentence boundaries.

## Development gates

### Unit tests

Nine controlled tests validated:

- eligible single-space trimming;
- rejection of multiple trailing spaces;
- rejection of mixed trailing whitespace;
- protection of whitespace-only text;
- exact one-character removal;
- idempotence;
- review-flag retention;
- unchanged risky inputs;
- result-contract validation.

### 50,000-record gate

- records scanned: 50,000;
- records written: 50,000;
- records changed: 49,613;
- independent verification problems: 0;
- immutability: PASS.

### 1,000,000-record gate

- records scanned: 1,000,000;
- records written: 1,000,000;
- records changed: 993,672;
- output shards: 10;
- each shard rows: 100,000;
- independent verification problems: 0;
- shard boundaries continuous;
- immutability: PASS.

## Full-corpus results

| Metric | Value |
|---|---:|
| Records scanned | 13,884,795 |
| Records written | 13,884,795 |
| Valid string records | 13,884,795 |
| Invalid input records | 0 |
| Records changed | 13,855,549 |
| Records unchanged | 29,246 |
| Records requiring review | 384,243 |
| Records without review flags | 13,500,552 |
| Characters removed | 13,855,549 |
| Output shards | 139 |
| Output size | approximately 21 GB |
| Validation problems | 0 |
| Independent verification problems | 0 |
| Raw-corpus immutability | PASS |

## Status counts

| Status | Records |
|---|---:|
| `CHANGED_SAFE` | 13,472,845 |
| `CHANGED_SAFE_REVIEW_REQUIRED` | 382,704 |
| `UNCHANGED` | 27,707 |
| `UNCHANGED_REVIEW_REQUIRED` | 1,539 |

The status counts sum exactly to 13,884,795.

## Review-flag counts

| Review flag | Record count |
|---|---:|
| `HIGH_CONFIDENCE_APOSTROPHE_GAP` | 317,746 |
| `AMBIGUOUS_APOSTROPHE_GAP` | 175,900 |
| `LEADING_WHITESPACE` | 1,266 |

Flag counts overlap because one record may contain several flags.

Unique review-required records: 384,243, or approximately 2.7674% of the
corpus.

## Shard layout

- records per complete shard: 100,000;
- complete shards: 138;
- final shard records: 84,795;
- total shards: 139;
- first source record ID: 0;
- final source record ID: 13,884,794.

The final shard covers IDs 13,800,000 through 13,884,794.

Each shard is represented in `output_manifest.json` with its byte size,
record count, and SHA-256 checksum.

## Validation

All producer-side validation checks passed:

- input/output record count;
- manifest record count;
- input partition;
- changed/unchanged partition;
- operation-count accounting;
- character-removal accounting;
- review partition;
- zero validation problems;
- raw-corpus immutability.

The independent verifier reread all 13,884,795 raw and derived records.

It found zero:

- missing or extra derived rows;
- source-record-ID mismatches;
- raw-hash mismatches;
- normalized-hash mismatches;
- character-count mismatches;
- unexplained text changes;
- unapproved operations;
- invalid trim semantics;
- idempotence failures;
- status or review-Boolean inconsistencies.

## Interpretation

The Phase 2 output is approved as a mechanically normalized document corpus.

It is not a gold correction dataset. Existing linguistic defects remain,
including apostrophe corruption, merged words, split words, spelling errors,
punctuation damage, fragments, and source material.

## Next phase

Phase 3 will create a sentence inventory by:

1. detecting sentence boundaries;
2. preserving document-to-sentence provenance;
3. assigning deterministic sentence identifiers;
4. flagging uncertain segmentation;
5. validating segmentation statistics;
6. preparing sentence-level quality screening.
