#!/usr/bin/env python3
"""
Deterministic Phase 3 sentence-boundary core.

Purpose
-------
Split one Phase 2 normalized document into sentence candidates while
preserving exact character offsets and provenance.

This module performs segmentation only. It does not correct text and
does not declare any sentence linguistically clean.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

SCHEMA_VERSION = "phase_03_sentence_record_v1"

TERMINAL_MARKS = ".!?…"
CLOSING_CHARACTERS = "\"»”’)]}"

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿʻʼ‘’']+")

# These words commonly continue a sentence after quoted speech,
# a slogan, or a reported question.
SPEECH_CONTINUATION_WORDS = {
    "deb",
    "dedi",
    "deya",
    "degan",
    "deyiladi",
    "deganidek",
    "shiori",
}

# Small evidence-based starter list.
# This list will be revised after real-data inspection.
KNOWN_ABBREVIATIONS = {
    "aj",
    "mchj",
    "prof",
    "dr",
    "kv",
}

# Uzbek names can use multi-character initials.
UZBEK_MULTI_LETTER_INITIALS = {
    "sh",
    "ch",
    "o'",
    "g'",
}

# These lowercase tokens are strong evidence that the preceding
# period is not a real sentence boundary.
LOWERCASE_GRAMMATICAL_CONTINUATIONS = {
    "ning",
    "ni",
    "ga",
    "da",
    "dan",
}

# These forms can be legitimate Uzbek ordinals, but the corpus audit
# showed that they also occur after damaged Roman-numeral conversion.
# They are used only for cautious boundary protection.
SUSPICIOUS_ORDINAL_TAIL_RE = re.compile(
    r"(?:"
    r"bir\s+minginchi|"
    r"besh\s+yuzinchi|"
    r"birinchi|"
    r"beshinchi|"
    r"o'ninchi|"
    r"yuzinchi"
    r")$",
    re.IGNORECASE,
)

# Domain suffixes directly supported by the Phase 3 10K audit.
# This list is intentionally narrow and evidence based.
KNOWN_DOMAIN_SUFFIXES = {
    "uz",
    "ru",
    "com",
    "net",
    "org",
    "day",
    "az",
    "info",
}

# Detect a suspicious ordinal phrase immediately following a
# personal initial. This represents the observed pattern where
# Roman letters were converted into Uzbek ordinal words.
#
# Examples:
#     J. bir minginchi.
#     F. birinchi.
#     Sh. beshinchi.
ROMAN_EXPANSION_AFTER_INITIAL_RE = re.compile(
    r"(?:^|\s)"
    r"(?:"
    r"[A-Z]|"
    r"Sh|SH|Ch|CH|"
    r"O['ʻʼ‘’]|"
    r"G['ʻʼ‘’]"
    r")\.\s+"
    r"(?:"
    r"bir\s+minginchi|"
    r"besh\s+yuzinchi|"
    r"birinchi|"
    r"beshinchi|"
    r"o'ninchi|"
    r"yuzinchi"
    r")$"
)

REVIEW_FLAGS = {
    "LOWERCASE_AFTER_TERMINAL",
    "NO_TERMINAL_PUNCTUATION",
    "VERY_LONG_SEGMENT",
    "SHORT_TOKEN_BEFORE_PERIOD",
    "SUSPICIOUS_ORDINAL_PERIOD_PROTECTED",
    "ROMAN_EXPANSION_INITIAL_PROTECTED",
    "LOWERCASE_AFTER_STRONG_PUNCTUATION_PROTECTED",
}


def _skip_spaces(text: str, index: int) -> int:
    """Move forward over whitespace characters."""
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _skip_closing_characters(text: str, index: int) -> int:
    """Move forward over closing quotation marks and brackets."""
    while index < len(text) and text[index] in CLOSING_CHARACTERS:
        index += 1
    return index


def _next_word(text: str, index: int) -> tuple[str | None, int, int]:
    """
    Return the next alphabetic token and its start/end positions.

    Whitespace and closing quote/bracket characters are ignored.
    """
    index = _skip_closing_characters(text, index)
    index = _skip_spaces(text, index)

    match = WORD_RE.match(text, index)

    if not match:
        return None, index, index

    return match.group(0), match.start(), match.end()


def _previous_word(text: str, index: int) -> tuple[str | None, int, int]:
    """Return the alphabetic token immediately before index."""
    match = re.search(
        r"([A-Za-zÀ-ÖØ-öø-ÿʻʼ‘’']+)$",
        text[:index],
    )

    if not match:
        return None, index, index

    return match.group(1), match.start(1), match.end(1)


def _following_character_class(text: str, index: int) -> str:
    """
    Classify the first meaningful character following a boundary.

    Closing quotes, brackets, and whitespace are ignored.
    """
    index = _skip_closing_characters(text, index)
    index = _skip_spaces(text, index)

    if index >= len(text):
        return "END"

    char = text[index]

    if char.isupper():
        return "UPPER"
    if char.islower():
        return "LOWER"
    if char.isdigit():
        return "DIGIT"

    return "OTHER"


def _is_initial_period(text: str, period_index: int) -> bool:
    """
    Detect a period belonging to an Uzbek personal initial.

    Examples:
        S. K. ning
        A. Ismoilov
        Sh. T. fuqaro
        O'. Karimov
    """
    previous_word, _, _ = _previous_word(text, period_index)

    if previous_word is None:
        return False

    is_single_uppercase_initial = (
        len(previous_word) == 1
        and previous_word.isupper()
    )

    is_uzbek_multi_letter_initial = (
        previous_word.lower()
        in UZBEK_MULTI_LETTER_INITIALS
        and previous_word[0].isupper()
    )

    if not (
        is_single_uppercase_initial
        or is_uzbek_multi_letter_initial
    ):
        return False

    next_word, _, _ = _next_word(text, period_index + 1)

    return next_word is not None


def _is_dotted_name_or_domain_period(
    text: str,
    period_index: int,
) -> bool:
    """
    Protect a period inside a spaced dotted name or domain.

    Examples observed in the corpus:
        Gazeta. ru
        Kun. uz
        Facebook. com

    The source text is preserved exactly. This function only
    prevents an incorrect sentence split.
    """
    previous_word, _, _ = _previous_word(
        text,
        period_index,
    )
    next_word, _, _ = _next_word(
        text,
        period_index + 1,
    )

    if previous_word is None or next_word is None:
        return False

    if len(previous_word) < 2:
        return False

    return (
        next_word.lower()
        in KNOWN_DOMAIN_SUFFIXES
    )


def _is_roman_expansion_initial_period(
    text: str,
    period_index: int,
) -> bool:
    """
    Protect an ordinal-looking period after a personal initial.

    Examples observed and manually reviewed:
        N. bir minginchi. foydasiga...
        F. birinchi. bilan...
        Z. bir minginchi. shifoxonada...

    Protection requires:
        1. an immediately preceding personal initial;
        2. a known suspicious ordinal expansion;
        3. a lowercase continuation after the period.

    The damaged text is not corrected and remains review-required.
    """
    prefix = text[:period_index]

    if not ROMAN_EXPANSION_AFTER_INITIAL_RE.search(
        prefix
    ):
        return False

    next_word, _, _ = _next_word(
        text,
        period_index + 1,
    )

    if next_word is None:
        return False

    return next_word[0].islower()


def _is_known_abbreviation_period(
    text: str,
    period_index: int,
) -> bool:
    """Detect a period following a known abbreviation."""
    previous_word, _, _ = _previous_word(text, period_index)

    if previous_word is None:
        return False

    return previous_word.lower() in KNOWN_ABBREVIATIONS


def _is_suspicious_ordinal_period(
    text: str,
    period_index: int,
) -> bool:
    """
    Protect suspicious ordinal-looking periods found in damaged text.

    Examples observed in the corpus:
        bir minginchi. ning
        beshinchi. MCHJ

    This does not correct or reinterpret the source text.
    """
    prefix = text[:period_index]

    if not SUSPICIOUS_ORDINAL_TAIL_RE.search(prefix):
        return False

    next_word, _, _ = _next_word(text, period_index + 1)

    if next_word is None:
        return False

    if (
        next_word.lower()
        in LOWERCASE_GRAMMATICAL_CONTINUATIONS
    ):
        return True

    if next_word.isupper():
        return True

    return False


def _is_lowercase_strong_punctuation_continuation(
    text: str,
    punctuation_start: int,
    punctuation_end: int,
) -> bool:
    """
    Avoid over-splitting after ? or ! when lowercase text follows.

    This covers reported speech and malformed internal punctuation,
    such as:
        minera! moddalar
    """
    if text[punctuation_start] not in "?!":
        return False

    return (
        _following_character_class(
            text,
            punctuation_end,
        )
        == "LOWER"
    )


def _is_speech_continuation(
    text: str,
    punctuation_end: int,
) -> bool:
    """
    Detect a continuation after quotation or reported speech.

    Examples:
        "Bu mumkinmi?" deb so'radi.
        "Birlashaylik!" shiori ostida...
    """
    next_word, _, _ = _next_word(text, punctuation_end)

    if next_word is None:
        return False

    return next_word.lower() in SPEECH_CONTINUATION_WORDS


def _extend_boundary_end(
    text: str,
    punctuation_start: int,
) -> tuple[int, int]:
    """
    Extend over consecutive terminal marks and closing characters.

    Returns:
        punctuation_end:
            End of the punctuation sequence, before closing quotes.

        sentence_end:
            End after attached closing quotes or brackets.
    """
    punctuation_end = punctuation_start + 1

    while (
        punctuation_end < len(text)
        and text[punctuation_end] in TERMINAL_MARKS
    ):
        punctuation_end += 1

    sentence_end = _skip_closing_characters(
        text,
        punctuation_end,
    )

    return punctuation_end, sentence_end


def _trim_span(
    text: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    """Exclude surrounding whitespace without changing sentence text."""
    while start < end and text[start].isspace():
        start += 1

    while end > start and text[end - 1].isspace():
        end -= 1

    return start, end


def _make_sentence_id(
    source_record_id: int,
    sentence_index: int,
    start: int,
    end: int,
    sentence_text: str,
) -> str:
    """
    Create a deterministic sentence identifier.

    The identifier remains stable when the source record, offsets,
    sentence index, text, and schema version remain unchanged.
    """
    payload = (
        f"{SCHEMA_VERSION}\n"
        f"{source_record_id}\n"
        f"{sentence_index}\n"
        f"{start}\n"
        f"{end}\n"
        f"{sentence_text}"
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]

    return (
        f"s3v1-{source_record_id}-"
        f"{sentence_index:04d}-{digest}"
    )


def _build_sentence_record(
    *,
    source_record_id: int,
    sentence_index: int,
    parent_normalized_text_sha256: str,
    parent_review_flags: list[str],
    text: str,
    start: int,
    end: int,
    terminal_punctuation: str,
    segmentation_method: str,
    boundary_flags: Iterable[str],
) -> dict:
    """Build and validate one sentence record."""
    start, end = _trim_span(text, start, end)

    if start >= end:
        raise ValueError("Cannot create an empty sentence record")

    sentence_text = text[start:end]
    flags = sorted(set(boundary_flags))

    if len(sentence_text) >= 400:
        flags = sorted(set(flags) | {"VERY_LONG_SEGMENT"})

    review_required = any(
        flag in REVIEW_FLAGS
        for flag in flags
    )

    sentence_id = _make_sentence_id(
        source_record_id=source_record_id,
        sentence_index=sentence_index,
        start=start,
        end=end,
        sentence_text=sentence_text,
    )

    record = {
        "schema_version": SCHEMA_VERSION,
        "sentence_id": sentence_id,
        "source_record_id": source_record_id,
        "sentence_index_in_record": sentence_index,
        "parent_normalized_text_sha256": (
            parent_normalized_text_sha256
        ),
        "sentence_text": sentence_text,
        "sentence_start_char": start,
        "sentence_end_char": end,
        "sentence_character_count": len(sentence_text),
        "sentence_token_count": len(sentence_text.split()),
        "terminal_punctuation": terminal_punctuation,
        "segmentation_method": segmentation_method,
        "boundary_flags": flags,
        "segmentation_review_required": review_required,
        "parent_review_flags": list(parent_review_flags),
    }

    if text[start:end] != sentence_text:
        raise AssertionError(
            "Character-offset reconstruction failed"
        )

    return record


def segment_document(
    *,
    source_record_id: int,
    normalized_text: str,
    parent_normalized_text_sha256: str = "",
    parent_review_flags: Iterable[str] | None = None,
) -> list[dict]:
    """
    Segment one normalized document into sentence candidates.

    Important:
        - no text is corrected;
        - no capitalization is changed;
        - exact source offsets are retained;
        - ambiguous boundary behaviour is recorded with flags.
    """
    if not isinstance(source_record_id, int):
        raise TypeError("source_record_id must be an integer")

    if not isinstance(normalized_text, str):
        raise TypeError("normalized_text must be a string")

    parent_flags = list(parent_review_flags or [])

    if not normalized_text:
        return []

    records: list[dict] = []
    segment_start = 0
    pending_flags: set[str] = set()

    index = 0

    while index < len(normalized_text):
        char = normalized_text[index]

        if char not in TERMINAL_MARKS:
            index += 1
            continue

        punctuation_end, sentence_end = _extend_boundary_end(
            normalized_text,
            index,
        )

        punctuation_text = normalized_text[
            index:punctuation_end
        ]

        if char == "." and _is_initial_period(
            normalized_text,
            index,
        ):
            pending_flags.add(
                "INITIAL_BOUNDARY_PROTECTED"
            )
            pending_flags.add(
                "SHORT_TOKEN_BEFORE_PERIOD"
            )
            index = punctuation_end
            continue

        if char == "." and _is_dotted_name_or_domain_period(
            normalized_text,
            index,
        ):
            pending_flags.add(
                "DOTTED_NAME_OR_DOMAIN_PROTECTED"
            )
            index = punctuation_end
            continue

        if char == "." and _is_known_abbreviation_period(
            normalized_text,
            index,
        ):
            pending_flags.add(
                "ABBREVIATION_BOUNDARY_PROTECTED"
            )
            index = punctuation_end
            continue

        if char == "." and _is_roman_expansion_initial_period(
            normalized_text,
            index,
        ):
            pending_flags.add(
                "ROMAN_EXPANSION_INITIAL_PROTECTED"
            )
            pending_flags.add(
                "SUSPICIOUS_ORDINAL_PERIOD_PROTECTED"
            )
            index = punctuation_end
            continue

        if char == "." and _is_suspicious_ordinal_period(
            normalized_text,
            index,
        ):
            pending_flags.add(
                "SUSPICIOUS_ORDINAL_PERIOD_PROTECTED"
            )
            index = punctuation_end
            continue

        if _is_speech_continuation(
            normalized_text,
            sentence_end,
        ):
            pending_flags.add(
                "QUOTE_OR_SPEECH_CONTINUATION"
            )
            index = sentence_end
            continue

        if _is_lowercase_strong_punctuation_continuation(
            normalized_text,
            index,
            sentence_end,
        ):
            pending_flags.add(
                "LOWERCASE_AFTER_TERMINAL"
            )
            pending_flags.add(
                "LOWERCASE_AFTER_STRONG_PUNCTUATION_PROTECTED"
            )
            index = sentence_end
            continue

        flags = set(pending_flags)

        following_class = _following_character_class(
            normalized_text,
            sentence_end,
        )

        if following_class == "LOWER":
            flags.add("LOWERCASE_AFTER_TERMINAL")

        start, end = _trim_span(
            normalized_text,
            segment_start,
            sentence_end,
        )

        if start < end:
            record = _build_sentence_record(
                source_record_id=source_record_id,
                sentence_index=len(records),
                parent_normalized_text_sha256=(
                    parent_normalized_text_sha256
                ),
                parent_review_flags=parent_flags,
                text=normalized_text,
                start=start,
                end=end,
                terminal_punctuation=punctuation_text,
                segmentation_method=(
                    "RULE_TERMINAL_PUNCTUATION"
                ),
                boundary_flags=flags,
            )
            records.append(record)

        segment_start = sentence_end
        pending_flags.clear()
        index = sentence_end

    trailing_start, trailing_end = _trim_span(
        normalized_text,
        segment_start,
        len(normalized_text),
    )

    if trailing_start < trailing_end:
        trailing_flags = set(pending_flags)
        trailing_flags.add("NO_TERMINAL_PUNCTUATION")

        record = _build_sentence_record(
            source_record_id=source_record_id,
            sentence_index=len(records),
            parent_normalized_text_sha256=(
                parent_normalized_text_sha256
            ),
            parent_review_flags=parent_flags,
            text=normalized_text,
            start=trailing_start,
            end=trailing_end,
            terminal_punctuation="",
            segmentation_method="TRAILING_FRAGMENT",
            boundary_flags=trailing_flags,
        )
        records.append(record)

    return records
