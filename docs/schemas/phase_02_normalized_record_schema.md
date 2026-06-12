# Phase 2 normalized-record schema

**Schema version:** `phase_02_normalized_record_v1`  
**Storage format:** UTF-8 JSONL  
**Granularity:** One derived row per raw corpus record

## Purpose

This schema stores mechanically normalized document records while preserving
provenance, transformation history, integrity hashes, and unresolved review
signals.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Fixed schema identifier |
| `source_record_id` | integer | Zero-based position in the raw top-level JSON array |
| `input_valid` | boolean | Whether the raw record contained valid string text |
| `raw_text_sha256` | string | SHA-256 of the exact UTF-8 raw text |
| `normalized_text_sha256` | string | SHA-256 of the exact UTF-8 normalized text |
| `raw_character_count` | integer | Python character count before normalization |
| `normalized_character_count` | integer | Python character count after normalization |
| `character_delta` | integer | Normalized count minus raw count |
| `normalized_text` | string | Derived text used by later phases |
| `operations_applied` | array of strings | Ordered allowlisted transformations |
| `operation_count` | integer | Number of applied operations |
| `normalization_status` | string | Result category |
| `review_flags` | array of strings | Unresolved quality signals |
| `review_required` | boolean | True when at least one review flag exists |
| `high_confidence_gap_count` | integer | Number of high-confidence apostrophe-gap matches |
| `ambiguous_gap_count` | integer | Number of ambiguous apostrophe-gap matches |
| `high_confidence_rule_ids` | array of strings | Triggered high-confidence rule identifiers |
| `ambiguous_rule_ids` | array of strings | Triggered ambiguous rule identifiers |

## Normalization statuses

- `CHANGED_SAFE`
- `CHANGED_SAFE_REVIEW_REQUIRED`
- `UNCHANGED`
- `UNCHANGED_REVIEW_REQUIRED`
- `INVALID_INPUT_REVIEW_REQUIRED`

## Version 1 allowlist

- `TRIM_ONE_TRAILING_ASCII_SPACE`

For a changed version 1 record:

- `character_delta` must equal `-1`;
- `normalized_text` must equal `raw_text[:-1]`;
- the removed character must be U+0020;
- the previous raw character must not be whitespace;
- `operations_applied` must contain exactly the approved operation.

For an unchanged record:

- `character_delta` must equal `0`;
- `operations_applied` must be empty;
- normalized text must equal raw text.

## Review flags currently emitted

- `HIGH_CONFIDENCE_APOSTROPHE_GAP`
- `AMBIGUOUS_APOSTROPHE_GAP`
- `LEADING_WHITESPACE`
- other protected boundary flags supported by the implementation

Review flags are heuristic signals. They are not gold error labels.

## Provenance rule

Each valid raw record produces exactly one derived record with the same
`source_record_id`.

The hashes identify the exact raw and derived text content.

## Storage layout

The full output uses 100,000 records per shard:

- 138 complete shards;
- one final shard with 84,795 rows;
- 139 shards total.

Every shard has a manifest entry containing:

- filename;
- record count;
- byte size;
- SHA-256 checksum.
