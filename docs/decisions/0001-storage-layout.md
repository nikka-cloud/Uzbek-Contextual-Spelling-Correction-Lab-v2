# Decision 0001: Separate Repository and Output Storage

## Status

Accepted during Phase 0.

## Context

The server root filesystem reports approximately 8.4 GB available and
100% usage after rounding.

The mounted UzbekVoice storage has approximately 9.5 TB available.

The source corpus is approximately 13 GB and is stored on the mounted
UzbekVoice storage.

## Decision

Store small Git-controlled project materials under:

