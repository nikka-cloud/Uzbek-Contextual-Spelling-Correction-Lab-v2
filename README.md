# Uzbek Contextual Spelling Correction Lab v2

An experimental NLP/ML project for sentence-level contextual spelling correction in Uzbek.

The goal is to detect suspicious words, generate possible corrections, rank them using sentence context, and avoid unsafe overcorrection. The project later expanded to study errors observed in real speech-to-text (STT) systems.

## Pipeline

Noisy Uzbek sentence  
→ error detection  
→ candidate generation  
→ contextual ranking  
→ confidence / safety decision  
→ correction suggestion or abstention

Sentence context is important because two individually valid Uzbek words may still be incorrect depending on the surrounding sentence.

## Project Progress

The lab has progressed through:

1. corpus auditing and duplicate analysis;
2. safe corpus normalization and clean-text selection;
3. realistic synthetic typo generation;
4. detector and ranker training-data preparation;
5. correction candidate generation;
6. baseline token-detection and contextual-ranking experiments;
7. controlled end-to-end correction and safety evaluation;
8. Transformer-ranker experiments;
9. real STT error collection and human review;
10. leakage-safe STT training-data extension experiments.

## Real STT Extension

The latest stage investigates how to make the existing synthetic training data more representative of errors observed in voice-AI systems.

### Existing protected baseline

- **13,200** synthetic correction pairs
- Train: **10,513**
- Dev: **1,262**
- Test: **1,425**

### Human-verified real STT data

**60 real correction pairs** were admitted as training candidates:

- 17 orthographic-like cases
- 15 context-dependent cases
- 28 domain/entity-related cases
- 33 unique error-to-correction mappings

The real examples were checked for duplicates and overlap with the existing baseline.

A small targeted augmentation experiment was then performed for error patterns considered safe to generalize. After leakage filtering, contextual review, and human validation, **one additional synthetic pair** was approved.

## Current Status

The STT extension is still experimental.

The 60 real STT pairs and the approved targeted synthetic pair have **not yet been merged into the final training dataset**, and no STT-enhanced retraining has been performed.

The next step is to compare two strategies:

- directly adding the verified real examples to training; or
- expanding recurring real STT error patterns through carefully controlled augmentation.

The existing baseline splits remain unchanged. Dev/test data was used only for leakage checks, not as a source for synthetic generation.

## Repository Scope

This repository contains code, configuration, documentation, tests, and small reproducible artifacts from the experiments.

Raw/private corpora, large generated datasets, model checkpoints, credentials, and internal data are intentionally excluded.
