# Phase 1 Report: Raw Corpus Investigation

## Status

**APPROVED**

The complete corpus-quality audit and exact-duplicate verification finished
successfully on 11 June 2026. All internal validation gates passed, and both
passes confirmed that the immutable raw corpus was unchanged.

## Mission

Phase 1 established what the corpus is, how it is structured, where systematic
quality problems occur, and which limitations must shape later normalisation,
sentence extraction, clean-sentence eligibility, synthetic-noise generation,
model training, and evaluation.

Phase 1 did not clean or correct text.

## Input

- Raw corpus: `/mnt/uzbekvoice_storage/text_data/normalized.json`
- Size: 13,853,548,528 bytes
- Format: top-level UTF-8 JSON array
- Record identity: zero-based JSON-array position
- Full records scanned: 13,884,795

## Execution design

Audit v2.2 used two controlled passes.

1. The quality audit used `ijson.backends.python`, 10,000-record position blocks,
   and 256 compact hash shards.
2. Exact duplicate analysis found repeated hash-and-length candidates, reread
   their raw records, and confirmed equality using complete text.

Audit runtime was 3,672.72 seconds. Exact duplicate verification took 177.42
seconds. The quality output occupied 892 MB, including 584 MB of hash shards.
Duplicate outputs occupied 35 MB. No SQLite, WAL, or SHM files were created.

## Structural results

| Metric | Value |
|---|---:|
| Records scanned | 13,884,795 |
| Dictionary records | 13,884,795 |
| Records with `text` | 13,884,795 |
| String `text` values | 13,884,795 |
| Missing `text` | 0 |
| Non-string `text` | 0 |
| Empty/whitespace-only text | 0 |

The corpus is structurally consistent.

## Text-length results

| Metric | Characters |
|---|---:|
| Minimum | 8 |
| Median | 856 |
| P90 | 1,668 |
| P95 | 1,999 |
| P99 | 2,887 |
| Maximum | 20,216 |
| Mean | 965.750 |

Records are document-like rather than sentence-level units.

## Boundary and whitespace results

| Finding | Records | Share |
|---|---:|---:|
| Exactly one trailing ASCII space | 13,855,549 | 99.789% |
| No trailing whitespace | 29,246 | 0.211% |
| Leading whitespace | 1,266 | 0.009% |
| Repeated horizontal whitespace | 11,619 | 0.084% |

Removing one trailing ASCII space is a strong candidate for a later safe,
derived, idempotent normalisation. It must not be applied to the raw file.

## Script and Unicode results

- Latin-script records: 13,884,793
- Records with no alphabetic characters: 2
- Configured suspicious-Unicode occurrences: none detected
- U+0027 apostrophe occurrences: 236,693,950
- Other configured apostrophe variants: 0

These results describe the configured detectors. They do not prove that every
record is Uzbek or that the corpus is linguistically clean.

## Apostrophe-to-space signals

| Finding | Value |
|---|---:|
| Records with high-confidence signals | 317,746 |
| Share of corpus | 2.288% |
| Records with ambiguous signals | 175,900 |
| Share of corpus | 1.267% |
| Records with any signal | 383,021 |
| Share of corpus | 2.759% |
| High-confidence matches | 863,348 |
| Ambiguous matches | 325,906 |

Rule counts:

| Rule | Matches |
|---|---:|
| `BO_L_PATTERN` | 371,822 |
| `O_Z_PATTERN` | 257,043 |
| `O_T_PATTERN` | 190,798 |
| `KO_R_PATTERN` | 145,912 |
| `O_Q_PATTERN` | 68,863 |
| `MA_LUM_PATTERN` | 66,554 |
| `QO_L_PATTERN` | 40,642 |
| `TO_G_R_PATTERN` | 36,907 |
| `E_LON_PATTERN` | 10,713 |

These are review signals, not automatic corrections.

## Corpus-position finding

Corruption is not uniformly distributed.

- Records 0-999,999: 16.281% contained a high-confidence signal.
- Records 1,000,000-1,999,999: 0.039%.
- Later million-record regions were mostly around 1.0%-1.5%.
- The most damaged 10,000-record blocks were concentrated around record
  positions 130,000-299,999, with high-confidence rates close to 90%.

This strongly indicates multiple source or preprocessing regimes. Because the
raw corpus lacks source metadata, the responsible source cannot be identified
from the corpus alone.

Corpus position and record-level quality flags must therefore be preserved in
later derived datasets.

## Exact duplicates

| Metric | Value |
|---|---:|
| Candidate hash groups | 71,658 |
| Candidate records | 158,496 |
| Candidate records verified | 158,496 |
| Exact duplicate groups | 71,658 |
| Occurrences after first | 86,838 |
| Share of all records | 0.625% |
| Hash-collision groups | 0 |

Exact duplicates must not cross train, development, and test splits. Near
duplicates remain a later investigation.

## Review-only content heuristics

- Records containing a source/credit phrase: 410,023
- Standalone source/credit candidates: 13,423
- Short sentence-like records: 86,834
- Short credit-like records: 9,794
- Short fragment/unknown records: 1,367

These labels are not grounds for automatic deletion or approval. Manual examples
show that the standalone-credit rule can produce false positives when a
substantive record ends with a publication marker.

## Main conclusions

1. The raw corpus is structurally consistent.
2. It is not a clean gold corpus of correct sentences.
3. Almost every record contains one trailing ASCII-space artefact.
4. Apostrophe-to-space corruption is real, large, and batch-dependent.
5. Exact duplicates exist and require split-group protection.
6. Records are document-like and require careful sentence segmentation.
7. Source-credit and short-record labels must remain review-only.
8. Script detection cannot establish language.
9. Safe structural normalisation must remain separate from linguistic correction.
10. Record provenance and quality flags must survive every later phase.

## Safety and approval

The following gates passed:

- complete 13,884,795-record accounting
- block-total accounting
- one hash identity per string-text record
- all internal validation checks
- all 158,496 duplicate candidates verified
- zero observed hash collisions
- quality-audit immutability check
- duplicate-analysis immutability check
- completion markers for both passes

No raw text was modified. No record was automatically removed. No spelling was
automatically corrected. No heuristic label was treated as gold.

## Phase 2 entry conditions

Phase 2 may begin with a derived, auditable normalisation pipeline. Initially
approved candidates are limited to mechanically safe transformations, especially
removing exactly one trailing ASCII space.

Phase 2 must preserve:

- `source_record_id`
- raw-text hash
- normalised text
- operation names
- operation counts
- quality flags
- corpus-position information
- duplicate-group information or linkage
- idempotence checks
- raw-corpus immutability

Apostrophe-gap repair, merged-word repair, split-word repair, source-credit
removal, and contextual spelling correction are not approved as automatic Phase
2 operations.
