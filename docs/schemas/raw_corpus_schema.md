# Raw Corpus Schema

## Purpose

This document records the observed structure and limitations of the immutable raw
corpus used by the Uzbek Contextual Spelling Correction Lab v2.

## Source

- Path: `/mnt/uzbekvoice_storage/text_data/normalized.json`
- Size: 13,853,548,528 bytes
- Encoding: UTF-8
- Container: one top-level JSON array
- JSONL: no
- UTF-8 BOM: not observed
- First meaningful character: `[`
- Last meaningful character: `]`

The raw corpus must remain read-only. All cleaning, normalisation, sentence
segmentation, filtering, and correction must be written to derived files.

## Record identity

The corpus provides no original record identifier. The project therefore defines:

```text
source_record_id = zero-based position in the top-level JSON array
```

`source_record_id` must be preserved in every derived dataset so that any sentence,
noise example, or model error can be traced back to the raw corpus.

## Required record structure

The full v2.2 streaming audit processed 13,884,795 records and confirmed:

- 13,884,795 dictionary records
- 13,884,795 records containing a `text` field
- 13,884,795 string-valued `text` fields
- 0 missing `text` fields
- 0 non-string `text` fields
- 0 empty or whitespace-only texts

Minimal required form:

```json
{"text": "Raw document-like text"}
```

Earlier controlled probes observed only the `text` key. The full v2.2 audit
validated the required field and type but did not claim that every possible extra
key was exhaustively enumerated.

## Text unit

A raw record is document-like, not guaranteed to be one sentence. Full-corpus
character-length statistics:

| Metric | Value |
|---|---:|
| Minimum | 8 |
| Median | 856 |
| P90 | 1,668 |
| P95 | 1,999 |
| P99 | 2,887 |
| Maximum | 20,216 |
| Mean | 965.750 |

Sentence segmentation must therefore occur in a later derived phase.

## Missing metadata

The corpus does not provide dependable source-level metadata such as:

- publication
- URL
- date
- title
- author
- domain
- language label
- original quality score
- original source identifier

Corpus position can reveal preprocessing regimes, but it cannot identify the
original publisher or collection pipeline.

## Known quality properties

The raw corpus is structurally consistent but is not a gold corpus of correct
Uzbek sentences. Confirmed properties include:

- almost universal trailing ASCII-space artefacts
- exact duplicate records
- apostrophe-to-space corruption
- merged and split words
- short fragments and source-credit lines
- distorted punctuation, numbers, abbreviations, and symbols
- batch-dependent corruption regimes

The script detector classified 13,884,793 records as Latin-script and two as
having no alphabetic characters. Script classification is not proof of language.

## Safety boundary

Phase 1 was descriptive only:

- raw text was not edited
- spelling was not corrected
- records were not deleted
- heuristic labels were not treated as gold
- automatic replacement remained disabled

Any derived transformation must preserve provenance and must never overwrite this
file.
