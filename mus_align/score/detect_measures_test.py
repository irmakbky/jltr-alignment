import unittest

from .. import ALIGNMENT_APP_DATA_DIR
from ..types import PDF
from .detect_measures import detect_measures

if ALIGNMENT_APP_DATA_DIR is None:
    raise FileNotFoundError("$ALIGNMENT_APP_DATA_DIR is not set.")


class TestDetectMeasures(unittest.TestCase):

    def test_detect_measures(self):
        pdf = PDF.from_path(ALIGNMENT_APP_DATA_DIR / "berceuse" / "complete.pdf")
        page_images = list(pdf.as_page_images())
        measures = [detect_measures(image) for image in page_images]
        assert [len(m) for m in measures] == [2, 19, 22, 20, 25, 25]


if __name__ == "__main__":
    unittest.main()
