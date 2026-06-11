# Decision 0002: Separate Full-Corpus Quality Audit from Exact Duplicate Verification

## Status

Accepted.

## Context

Audit v2.1 combined corpus-quality inspection with live exact-duplicate tracking
inside one SQLite database. It stored the full text of every observed record,
used WAL mode, and placed the database on mounted storage.

The 200,000-record validation passed. The first complete-corpus attempt reached
8.4 million records, built a database of approximately 10.6 GB, and then the
Python process terminated with a native segmentation fault.

The kernel confirmed a segmentation fault, but no core dump was available.
Therefore, the exact native component responsible could not be proven. The
high-risk configuration included both the compiled `yajl2_c` parser and a large
SQLite/WAL workload on mounted storage.

## Decision

Audit v2.2 separates the responsibilities.

### Pass 1: corpus-quality audit

- use `ijson.backends.python`
- stream the top-level JSON array
- calculate structure, whitespace, Unicode, script, length, apostrophe-gap,
  source-credit, short-record, and corpus-position statistics
- write one compact SHA-256/length/record-ID identity to one of 256 hash shards
- do not create SQLite, WAL, or shared-memory files
- preserve the raw corpus fingerprint before and after processing

### Pass 2: exact duplicate verification

- identify repeated hash-and-length candidates from the shard inventory
- reread only the candidate record positions from the raw corpus
- compare complete text before confirming an exact duplicate
- report hash-collision groups separately
- repeat the raw-corpus immutability check

## Validation

The redesign was approved progressively:

1. 200,000-record parity against v2.1
2. 1,000,000-record stability test
3. full 13,884,795-record audit
4. complete verification of 158,496 duplicate candidates

Final results:

- 71,658 exact duplicate groups
- 86,838 occurrences after the first copy
- 0 hash-collision groups
- all audit validation checks passed
- both immutability checks passed
- no SQLite/WAL/SHM files were produced

## Consequences

Benefits:

- stable full-corpus execution
- smaller output footprint
- clear separation of concerns
- exact-text duplicate confirmation retained
- better reproducibility and failure isolation

Trade-offs:

- duplicate verification is a second corpus pass
- hash shards are intermediate files and must remain outside Git
- pure-Python parsing may be slower than a compiled backend in isolation, though
  the removal of live SQLite work made the complete pipeline substantially more
  stable and efficient

## Safety

Hash equality alone is never treated as final proof of duplicate text. Full-text
comparison remains mandatory before a duplicate group is approved.
