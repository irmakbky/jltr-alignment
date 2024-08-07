import gzip
import json
import unittest
from typing import Any

from . import ALIGNMENT_APP_DATA_DIR, TEST_DATA_DIR
from .mesa13 import TAGS, AlignedScore, load_mesa13_alignment

if ALIGNMENT_APP_DATA_DIR is None:
    raise FileNotFoundError("$ALIGNMENT_APP_DATA_DIR is not set.")

if TEST_DATA_DIR is None:
    raise FileNotFoundError("Test data not found.")


class TestMesa13(unittest.TestCase):

    def _compare_against_legacy(self, tag: str, expected: dict, actual: dict):
        assert frozenset(expected.keys()) == frozenset(
            ["score_pages", "logical_order", "alignment"]
        )

        bbox_to_key = lambda b: f'{b["left"]}-{b["top"]}-{b["width"]}-{b["height"]}'

        # Check score pages
        old_score_pages = []
        for p in actual["score_pages"]:
            old_page: dict[str, list] = {
                "bbox": [],
            }
            for m in p["measures"]:
                b = m["bbox"]
                old_page["bbox"].append(bbox_to_key(b))
            old_score_pages.append(old_page)
        self.assertEqual(old_score_pages, expected["score_pages"])

        # Check logical order
        old_logical_order = [
            [page_id, bbox_to_key(measure_attrs["bbox"])]
            for page_id, measure_attrs in actual["alignment"]["logical_order"]
        ]
        self.assertEqual(old_logical_order, expected["logical_order"])

        # Check alignment
        old_alignment = {
            "measure_indices": actual["alignment"]["measure_indices"],
            "times": actual["alignment"]["times"],
            "logical_order": [
                [page_id, bbox_to_key(measure_attrs["bbox"])]
                for page_id, measure_attrs in actual["alignment"]["logical_order"]
            ],
            "max_time": actual["alignment"]["max_time"],
        }
        self.assertEqual(old_alignment, expected["alignment"])

    def test_mesa13(self):
        for tag in TAGS:
            aligned = load_mesa13_alignment(tag)
            with gzip.open(TEST_DATA_DIR / "mesa13" / f"{tag}.json.gz", "rt") as f:
                expected = json.load(f)
            self._compare_against_legacy(tag, expected, aligned.attrs_for_hash())
            aligned_hat = AlignedScore.from_project_zip(
                aligned.as_project_zip(), check_consistency=True
            )
            self.assertEqual(aligned_hat.attrs_for_hash(), aligned.attrs_for_hash())
            self.assertEqual(aligned_hat.checksum(), aligned.checksum())


if __name__ == "__main__":
    unittest.main()
