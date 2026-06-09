# Phase 0 — Project Foundation

## Goal

Create a clean, separate, and reproducible repository before processing the Uzbek corpus.

## Environment findings

- The new repository is separate from previous spelling-correction projects.
- The raw corpus exists and is approximately 13 GB.
- The raw corpus remains immutable.
- The root filesystem has limited remaining capacity.
- Large derived outputs are stored on mounted UzbekVoice storage.

## Project locations

- Repository: /root/uzbek-contextual-spelling-correction-lab-v2
- Raw corpus: /mnt/uzbekvoice_storage/text_data/normalized.json
- Derived outputs: /mnt/uzbekvoice_storage/projects/uzbek-contextual-spelling-correction-lab-v2/outputs

## Foundation decisions

- Never modify the raw corpus.
- Write all transformations to derived output files.
- Keep large outputs outside Git.
- Use explicit random seeds.
- Treat synthetic typo rows as review-first, not gold data.
- Keep automatic replacement disabled.
- Use sentence_id for future leakage-safe dataset splitting.

## Completed checks

- Repository separation: passed
- Raw corpus existence and safety: passed
- External storage availability: passed
- Output directories and symbolic link: passed
- README and Git ignore policy: passed
- Configuration and requirements files: passed

## Remaining Phase 0 work

- Initialize Git.
- Configure repository-local Git identity.
- Test ignore rules.
- Stage and inspect foundation files.
- Create the initial commit.
- Connect and push to GitHub.
