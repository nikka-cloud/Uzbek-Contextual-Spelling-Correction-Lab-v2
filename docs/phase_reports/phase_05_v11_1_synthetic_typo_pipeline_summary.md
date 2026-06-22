# Phase 5B — v11.1 Realistic Synthetic Typo Pipeline

## Status

Completed.

## Final dataset version

v11.1_realistic_core

## Purpose

Create a model-facing Uzbek synthetic typo dataset from manually verified clean target sentences.

The dataset is intended for spelling-error detection, correction-pair construction, and future contextual ranking experiments.

## Source clean targets

- Frozen clean targets: 502 sentences
- Normalization audit result: PASS
- Flagged normalization issues: 0

## Generator versions

### v10 deterministic sweep

Purpose:

- detector baseline
- one eligible token position produces one corruption row

Result:

- eligible token positions: 5,117
- generated rows: 5,117
- missing token positions: 0
- structural audit: PASS

### v11 realistic multivariant

Purpose:

- generate multiple realistic typo variants per eligible token position

Result:

- generated rows: 14,557
- eligible token positions: 5,117
- missing token positions: 0
- structural audit: PASS

### v11.1 realistic core

Purpose:

- remove normalization-only apostrophe variants from v11
- keep true misspelling variants

Removed rules:

- apostrophe_ascii_to_backtick
- apostrophe_ascii_to_curly_left

Kept rules include:

- apostrophe_drop
- apostrophe_g_drop
- apostrophe_o_drop
- deletion
- insertion
- transposition
- phonetic sh/ch variants

## v11.1 counts

- original v11 rows: 14,557
- removed normalization-only rows: 1,357
- final v11.1 core rows: 13,200
- eligible token positions covered: 5,117 / 5,117
- missing token positions after filtering: 0

## v11.1 typo family counts

- deletion: 4,896
- insertion: 3,909
- transposition: 2,649
- phonetic: 1,016
- apostrophe: 730

## v11.1 structural audit

PASS.

Checks passed:

- no removed-rule leakage
- no bad constant rows
- no bad alignment rows
- no unchanged input rows
- no bad reconstruction rows
- no missing token positions

## Important artifact hashes

token_generator_v11_1_core_output.csv

fff9dd6031a3cafa30c3760e28fd20f5012e7e2c7653be4ad05d61c175feba22

token_generator_v11_1_core_generation_audit.csv

488d6e48162816343d177a82cf83b49307e2060cce8420590c38dc9c52ed0837

token_generator_v11_1_core_summary.md

2ef7d15fef9ab3c968d24b29b7605515086716dcd6b21d334b92e79940153547

## Git policy

Generated datasets remain under outputs/.

outputs/ is ignored by Git and points to external project storage.

GitHub tracks code, tests, and documentation only.
