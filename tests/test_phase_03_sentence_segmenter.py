import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "03_sentence_segmenter_core.py"
)

SPEC = importlib.util.spec_from_file_location(
    "phase_03_sentence_segmenter_core",
    MODULE_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise RuntimeError(
        f"Could not load module from {MODULE_PATH}"
    )

segmenter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(segmenter)


class Phase03SentenceSegmenterTests(unittest.TestCase):

    def test_simple_sentences_and_offsets(self):
        text = (
            "Bugun havo yaxshi. "
            "Ertaga yomg'ir yog'adi."
        )

        rows = segmenter.segment_document(
            source_record_id=10,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["sentence_text"],
            "Bugun havo yaxshi.",
        )
        self.assertEqual(
            rows[1]["sentence_text"],
            "Ertaga yomg'ir yog'adi.",
        )

        for row in rows:
            start = row["sentence_start_char"]
            end = row["sentence_end_char"]

            self.assertEqual(
                text[start:end],
                row["sentence_text"],
            )

    def test_initial_chain_is_not_split(self):
        text = (
            "Hamshira S. K. ning arizasi ko'rib chiqildi. "
            "Ish davom etdi."
        )

        rows = segmenter.segment_document(
            source_record_id=11,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["sentence_text"],
            (
                "Hamshira S. K. ning arizasi "
                "ko'rib chiqildi."
            ),
        )
        self.assertIn(
            "INITIAL_BOUNDARY_PROTECTED",
            rows[0]["boundary_flags"],
        )

    def test_single_initial_before_name_is_not_split(self):
        text = (
            "A. Ismoilov yig'ilishda qatnashdi. "
            "Majlis yakunlandi."
        )

        rows = segmenter.segment_document(
            source_record_id=12,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["sentence_text"],
            "A. Ismoilov yig'ilishda qatnashdi.",
        )

    def test_speech_continuation_is_not_split(self):
        text = (
            "\"Bu mumkinmi?\" deb so'radi. "
            "Javob berildi."
        )

        rows = segmenter.segment_document(
            source_record_id=13,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["sentence_text"],
            "\"Bu mumkinmi?\" deb so'radi.",
        )
        self.assertIn(
            "QUOTE_OR_SPEECH_CONTINUATION",
            rows[0]["boundary_flags"],
        )

    def test_slogan_continuation_is_not_split(self):
        text = (
            "\"Birlashaylik!\" shiori ostida tadbir o'tdi. "
            "Tadbir yakunlandi."
        )

        rows = segmenter.segment_document(
            source_record_id=14,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["sentence_text"],
            (
                "\"Birlashaylik!\" shiori ostida "
                "tadbir o'tdi."
            ),
        )

    def test_lowercase_after_terminal_is_flagged(self):
        text = (
            "Ma'lumot berildi. "
            "birinchi bosqich boshlandi."
        )

        rows = segmenter.segment_document(
            source_record_id=15,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertIn(
            "LOWERCASE_AFTER_TERMINAL",
            rows[0]["boundary_flags"],
        )
        self.assertTrue(
            rows[0]["segmentation_review_required"]
        )

    def test_trailing_fragment_is_preserved_and_flagged(self):
        text = "Birinchi gap. Manba: Axborot xizmati"

        rows = segmenter.segment_document(
            source_record_id=16,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[1]["sentence_text"],
            "Manba: Axborot xizmati",
        )
        self.assertEqual(
            rows[1]["segmentation_method"],
            "TRAILING_FRAGMENT",
        )
        self.assertIn(
            "NO_TERMINAL_PUNCTUATION",
            rows[1]["boundary_flags"],
        )

    def test_sentence_ids_are_deterministic(self):
        text = "Birinchi gap. Ikkinchi gap."

        rows_a = segmenter.segment_document(
            source_record_id=17,
            normalized_text=text,
        )

        rows_b = segmenter.segment_document(
            source_record_id=17,
            normalized_text=text,
        )

        ids_a = [row["sentence_id"] for row in rows_a]
        ids_b = [row["sentence_id"] for row in rows_b]

        self.assertEqual(ids_a, ids_b)

    def test_surrounding_spaces_do_not_create_empty_rows(self):
        text = "   Birinchi gap.   Ikkinchi gap.   "

        rows = segmenter.segment_document(
            source_record_id=18,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertTrue(
            all(row["sentence_text"] for row in rows)
        )


    def test_uzbek_multi_letter_initial_is_not_split(self):
        text = (
            "Savdo agenti Sh. T. fuqaro N. ga murojaat qildi. "
            "Ish yakunlandi."
        )

        rows = segmenter.segment_document(
            source_record_id=19,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["sentence_text"],
            (
                "Savdo agenti Sh. T. fuqaro N. ga "
                "murojaat qildi."
            ),
        )

    def test_suspicious_ordinal_before_suffix_is_not_split(self):
        text = (
            "Fuqaro J. bir minginchi. ning arizasi ko'rildi. "
            "Ish yakunlandi."
        )

        rows = segmenter.segment_document(
            source_record_id=20,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["sentence_text"],
            (
                "Fuqaro J. bir minginchi. ning "
                "arizasi ko'rildi."
            ),
        )
        self.assertIn(
            "SUSPICIOUS_ORDINAL_PERIOD_PROTECTED",
            rows[0]["boundary_flags"],
        )

    def test_suspicious_ordinal_before_acronym_is_not_split(self):
        text = (
            "To'lov beshinchi. MCHJ tomonidan qabul qilindi. "
            "Ish yakunlandi."
        )

        rows = segmenter.segment_document(
            source_record_id=21,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["sentence_text"],
            (
                "To'lov beshinchi. MCHJ tomonidan "
                "qabul qilindi."
            ),
        )

    def test_lowercase_after_exclamation_is_preserved(self):
        text = (
            "Tarkibida organik, minera! moddalar mavjud. "
            "Suv tekshirildi."
        )

        rows = segmenter.segment_document(
            source_record_id=22,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["sentence_text"],
            (
                "Tarkibida organik, minera! "
                "moddalar mavjud."
            ),
        )
        self.assertIn(
            "LOWERCASE_AFTER_STRONG_PUNCTUATION_PROTECTED",
            rows[0]["boundary_flags"],
        )
        self.assertTrue(
            rows[0]["segmentation_review_required"]
        )

    def test_unit_abbreviation_is_not_split(self):
        text = (
            "Maydon o'n kv. mga teng. "
            "Hisob yakunlandi."
        )

        rows = segmenter.segment_document(
            source_record_id=23,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["sentence_text"],
            "Maydon o'n kv. mga teng.",
        )

    def test_dotted_domain_is_not_split(self):
        text = (
            "Bu haqda Gazeta. ru xabar berdi. "
            "Ish yakunlandi."
        )

        rows = segmenter.segment_document(
            source_record_id=24,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["sentence_text"],
            "Bu haqda Gazeta. ru xabar berdi.",
        )
        self.assertIn(
            "DOTTED_NAME_OR_DOMAIN_PROTECTED",
            rows[0]["boundary_flags"],
        )

    def test_standalone_period_is_not_domain_protected(self):
        text = (
            "Xabar yakunlandi. "
            "Ruhiy holat barqaror."
        )

        rows = segmenter.segment_document(
            source_record_id=25,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertNotIn(
            "DOTTED_NAME_OR_DOMAIN_PROTECTED",
            rows[0]["boundary_flags"],
        )

    def test_roman_expansion_after_initial_is_not_split(self):
        text = (
            "Fuqaro J. bir minginchi. noqonuniy "
            "daromad orttirgan. Ish yakunlandi."
        )

        rows = segmenter.segment_document(
            source_record_id=26,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["sentence_text"],
            (
                "Fuqaro J. bir minginchi. noqonuniy "
                "daromad orttirgan."
            ),
        )
        self.assertIn(
            "ROMAN_EXPANSION_INITIAL_PROTECTED",
            rows[0]["boundary_flags"],
        )
        self.assertTrue(
            rows[0]["segmentation_review_required"]
        )

    def test_legitimate_ordinal_sentence_still_splits(self):
        text = (
            "U musobaqada birinchi. "
            "Keyingi ishtirokchi keldi."
        )

        rows = segmenter.segment_document(
            source_record_id=27,
            normalized_text=text,
        )

        self.assertEqual(len(rows), 2)
        self.assertNotIn(
            "ROMAN_EXPANSION_INITIAL_PROTECTED",
            rows[0]["boundary_flags"],
        )


if __name__ == "__main__":
    unittest.main()
