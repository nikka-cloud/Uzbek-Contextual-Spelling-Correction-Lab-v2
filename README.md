# Uzbek Contextual Spelling Correction Lab v2

An auditable engineering project for building an Uzbek sentence-level contextual spelling-correction pipeline.

## Main goal

Build a controlled system that can:

1. detect a suspicious word using the complete sentence context;
2. identify the suspicious word position;
3. generate plausible correction candidates;
4. rank candidates using contextual compatibility;
5. suggest a correction or abstain;
6. preserve human review and safety controls.

The initial system must not automatically replace text.

## Planned pipeline

Noisy Uzbek sentence  
→ token-level error detector  
→ suspicious word position  
→ candidate generator  
→ contextual candidate ranker  
→ confidence and safety decision  
→ suggestion, review, or abstention

## Example

Input sentence:

`u kitobni olbi keldi`

Expected detector labels:

- `u` → `KEEP`
- `kitobni` → `KEEP`
- `olbi` → `CORRECT_THIS`
- `keldi` → `KEEP`

Possible correction candidates:

- `olib`
- `oldi`
- `olti`
- `oliy`

Expected suggestion:

`u kitobni olib keldi`

## Why sentence context is required

Some Uzbek words cannot be corrected safely using only dictionaries or edit distance.

For example:

- `U musobaqada g'olib bo'ldi.`
- `U kitobni olib keldi.`

Both `g'olib` and `olib` are valid Uzbek words. The correct choice depends on the full sentence.

Therefore, the long-term system separates:

1. error detection;
2. candidate generation;
3. contextual candidate ranking;
4. confidence and safety decisions.

## Data locations

### Raw immutable corpus

`/mnt/uzbekvoice_storage/text_data/normalized.json`

### Git repository

`/root/uzbek-contextual-spelling-correction-lab-v2`

### Large generated outputs

`/mnt/uzbekvoice_storage/projects/uzbek-contextual-spelling-correction-lab-v2/outputs`

The repository path `outputs` is a symbolic link to the external mounted storage.

Large generated outputs are intentionally excluded from Git.

## Raw-data policy

The source corpus is immutable.

Scripts may read the raw corpus, but they must never:

- overwrite it;
- append to it;
- directly normalize it in place;
- silently correct its contents;
- replace the original file.

All transformations must be written to derived files inside the project output directories.

## Synthetic-data rule

The first sentence-level synthetic dataset follows this rule:

One row = one clean sentence + one changed word position + one typo type.

Example:

Clean sentence:

`u kitobni olib keldi`

Noisy sentence:

`u kitobni olbi keldi`

Target word index:

`2`

Error type:

`adjacent_transposition`

Only one word occurrence is changed in each initial synthetic row.

Multi-error sentences may be explored later after the single-error dataset has been validated.

## Safety defaults

Synthetic rows use these defaults:

- `synthetic_is_gold = false`
- `auto_replace_allowed = false`
- `review_required = true`

The initial system produces correction suggestions for human review.

Protected terms, proper nouns, organisations, countries, brands, official titles, and sensitive terms must not be corrupted during the safe first version.

No-error examples must also be included so that future models learn not to overcorrect already-correct sentences.

## Planned architecture

The long-term controlled pipeline is:

Noisy Uzbek sentence  
→ token-level contextual detector  
→ suspicious word position  
→ correction candidate generator  
→ contextual candidate ranker  
→ confidence and safety policy  
→ suggestion, human review, or abstention

An encoder-only multilingual Transformer is planned for the token-level detector.

The detector will predict one spelling-status label per original word, such as:

- `KEEP`
- `CORRECT_THIS`

Candidate generation and final contextual correction selection remain separate tasks.

## Project phases

1. Project foundation
2. Raw corpus investigation
3. Corpus curation and enrichment
4. Sentence inventory
5. Word eligibility and risk labels
6. Sentence-level synthetic noise
7. Collision analysis and review exports
8. Training views and leakage-safe splits
9. Tokenizer and embedding audit
10. Token-level error detector
11. Candidate generation
12. Contextual candidate ranking
13. Full pipeline and safety evaluation

## Development cycle

Every phase follows:

LEARN  
→ INVESTIGATE  
→ DESIGN  
→ BUILD  
→ RUN  
→ TEST  
→ REVIEW  
→ DOCUMENT  
→ APPROVE  
→ NEXT PHASE

## Repository policy

GitHub may contain:

- source code;
- configuration files;
- documentation;
- tests;
- schemas;
- small reviewed samples;
- metrics summaries;
- reproducible instructions.

GitHub must not contain:

- the full raw corpus;
- private data;
- large generated datasets;
- model checkpoints;
- virtual environments;
- caches;
- credentials;
- SSH keys;
- access tokens;
- secrets.

## Current status

Phase 0: Project foundation.

The repository structure and external output storage have been created.

Corpus investigation and model training have not started.
