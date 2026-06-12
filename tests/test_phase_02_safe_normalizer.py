#!/usr/bin/env python3
"""Unit tests for 02_build_safe_normalized_corpus.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "02_build_safe_normalized_corpus.py"
)
MODULE_NAME = "phase_02_safe_normalizer"
spec = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import {SCRIPT_PATH}")
module = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = module
spec.loader.exec_module(module)


class SafeTrailingSpaceTests(unittest.TestCase):
    def test_exact_single_ascii_space_is_eligible(self) -> None:
        self.assertTrue(
            module.is_exactly_one_safe_trailing_ascii_space("Salom. ")
        )

    def test_multiple_trailing_spaces_are_not_eligible(self) -> None:
        self.assertFalse(
            module.is_exactly_one_safe_trailing_ascii_space("Salom.  ")
        )

    def test_tab_before_final_space_is_not_eligible(self) -> None:
        self.assertFalse(
            module.is_exactly_one_safe_trailing_ascii_space("Salom.\t ")
        )

    def test_space_only_is_not_eligible(self) -> None:
        self.assertFalse(module.is_exactly_one_safe_trailing_ascii_space(" "))

    def test_approved_operation_removes_only_final_character(self) -> None:
        raw = "Bugun havo yaxshi. "
        normalized, operations = module.apply_allowed_operations(
            raw, (module.TRIM_ONE_TRAILING_ASCII_SPACE,)
        )
        self.assertEqual(normalized, raw[:-1])
        self.assertEqual(
            operations, (module.TRIM_ONE_TRAILING_ASCII_SPACE,)
        )

    def test_operation_is_idempotent(self) -> None:
        once, _ = module.apply_allowed_operations(
            "Salom. ", (module.TRIM_ONE_TRAILING_ASCII_SPACE,)
        )
        twice, second_operations = module.apply_allowed_operations(
            once, (module.TRIM_ONE_TRAILING_ASCII_SPACE,)
        )
        self.assertEqual(once, twice)
        self.assertEqual(second_operations, ())

    def test_linguistic_gap_is_flagged_but_not_corrected(self) -> None:
        raw = "Bu ma lumot muhim. "
        result = module.normalize_text(
            raw, (module.TRIM_ONE_TRAILING_ASCII_SPACE,)
        )
        self.assertEqual(result.normalized_text, "Bu ma lumot muhim.")
        self.assertIn(
            "HIGH_CONFIDENCE_APOSTROPHE_GAP", result.review_flags
        )
        self.assertNotIn("ma'lumot", result.normalized_text)
        self.assertEqual(
            result.normalization_status,
            "CHANGED_SAFE_REVIEW_REQUIRED",
        )

    def test_multiple_spaces_are_flagged_and_unchanged(self) -> None:
        raw = "Salom.  "
        result = module.normalize_text(
            raw, (module.TRIM_ONE_TRAILING_ASCII_SPACE,)
        )
        self.assertEqual(result.normalized_text, raw)
        self.assertIn("MULTIPLE_TRAILING_ASCII_SPACES", result.review_flags)
        self.assertEqual(result.operations_applied, ())

    def test_result_validator_accepts_approved_change(self) -> None:
        raw = "Salom. "
        result = module.normalize_text(
            raw, (module.TRIM_ONE_TRAILING_ASCII_SPACE,)
        )
        problems = module.validate_normalization_result(
            raw, result, (module.TRIM_ONE_TRAILING_ASCII_SPACE,)
        )
        self.assertEqual(sum(problems.values()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
