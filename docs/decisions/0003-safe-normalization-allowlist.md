# Decision 0003 — Safe normalization allowlist

**Status:** Accepted  
**Date:** 2026-06-12  
**Phase:** 2 — Safe derived-corpus normalization

## Context

The raw Uzbek corpus contains 13,884,795 records. Phase 1 showed that
13,855,549 records end with exactly one ordinary ASCII space.

The project requires a mechanically consistent derived corpus before sentence
segmentation. However, the raw text also contains linguistic problems such as
apostrophe gaps, merged words, split words, spelling errors, punctuation
damage, source credits, and fragments.

Automatically repairing those linguistic problems at this stage would risk
changing valid text and creating unreliable correction targets.

## Decision

Phase 2 uses an explicit transformation allowlist.

The only approved automatic operation in version 1 is:

- `TRIM_ONE_TRAILING_ASCII_SPACE`

This operation is eligible only when:

1. the text contains at least two characters;
2. the final character is U+0020 ASCII SPACE;
3. the preceding character is not whitespace;
4. the result is exactly `raw_text[:-1]`;
5. the operation does not create empty text.

## Transformations deliberately not performed

Phase 2 does not automatically repair:

- apostrophe gaps;
- spelling;
- punctuation;
- grammar;
- merged words;
- split words;
- abbreviations;
- names;
- source or photo credits;
- sentence boundaries;
- tabs, newlines, or unusual whitespace.

Detected apostrophe gaps and leading whitespace remain review flags.

## Required guarantees

Every derived record must preserve:

- `source_record_id`;
- raw-text SHA-256;
- normalized-text SHA-256;
- raw and normalized character counts;
- character delta;
- applied operation names;
- review flags;
- normalization status.

The transformation must be idempotent:

`normalize(normalize(text)) == normalize(text)`

The raw corpus must remain immutable.

## Validation result

The full run processed 13,884,795 records.

- records written: 13,884,795;
- records changed: 13,855,549;
- records unchanged: 29,246;
- characters removed: 13,855,549;
- validation problems: 0;
- independent verification problems: 0;
- raw-corpus immutability: PASS.

The full derived corpus is approved as input to Phase 3.

## Consequence

The Phase 2 corpus is mechanically normalized and auditable, but it is not a
gold linguistic corpus. Later phases must perform sentence segmentation,
sentence-quality screening, target eligibility classification, and controlled
synthetic error generation.
