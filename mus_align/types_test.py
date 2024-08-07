import io
import unittest

import librosa

from . import ALIGNMENT_APP_DATA_DIR, TEST_DATA_DIR
from .types import PDF, Audio, Image, Score

if ALIGNMENT_APP_DATA_DIR is None:
    raise FileNotFoundError("$ALIGNMENT_APP_DATA_DIR is not set.")
if TEST_DATA_DIR is None:
    raise FileNotFoundError("Test data not found.")


# Test PDF
class TestTypes(unittest.TestCase):
    def test_image(self):
        pdf = PDF.from_path(ALIGNMENT_APP_DATA_DIR / "berceuse" / "complete.pdf")
        page_images = list(pdf.as_page_images())
        self.assertEqual(len(page_images), 6)
        img = page_images[0]
        self.assertEqual(
            img.checksum(),
            "0643e2fcef896257b8de5258462175585176ccdfd47e263f2e96ccecd0a890f7",
        )
        img_hat = Image.open(io.BytesIO(img.raw_png))
        self.assertEqual(img.checksum(), img_hat.checksum())

    def test_audio(self):
        audio_path = ALIGNMENT_APP_DATA_DIR / "berceuse" / "complete.mp3"
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        audio = Audio.from_path(audio_path)
        self.assertEqual(audio.sample_rate, sr)
        self.assertEqual(audio.samples.shape, (11375284,))
        self.assertEqual(len(audio.raw), 4961176)
        self.assertEqual(audio.ext, ".mp3")
        self.assertEqual(
            audio.checksum(),
            "a23f613d2c340c205234caa44a994d33718d21f3f09b764f7ba97bcacb51fd34",
        )
        audio2 = Audio(samples=y, sample_rate=sr)
        self.assertEqual(audio2.sample_rate, sr)
        self.assertEqual(audio2.samples.shape, (11375284,))
        with self.assertRaises(Exception):
            audio2.raw
        self.assertEqual(audio2.ext, None)
        self.assertEqual(
            audio2.checksum(),
            "196eb0fb13536056cee63e2471cfac24fa67ed79bbd204f7dec2171d0c81c218",
        )

    def test_pdf(self):
        pdf = PDF.from_path(ALIGNMENT_APP_DATA_DIR / "berceuse" / "complete.pdf")
        page_images = list(pdf.as_page_images())
        assert len(page_images) == 6

    def test_score(self):
        # Load unlabeled project file
        project_path = (TEST_DATA_DIR / "mapleleaf_050_unlabeled.zip").resolve()
        with open(project_path, "rb") as f:
            score = Score.from_project_zip(f.read(), check_consistency=True)
        assert len(score.score_pages) == 3
        assert [len(page) for page in score.score_pages] == [
            18,
            34,
            36,
        ]  # graphical order
        assert len(score) == 88  # logical order (same when unlabeled)

        # Load labeled project file
        project_path = (TEST_DATA_DIR / "mapleleaf_050_labeled.zip").resolve()
        with open(project_path, "rb") as f:
            score = Score.from_project_zip(f.read(), check_consistency=True)
        assert len(score.score_pages) == 3
        assert [len(page) for page in score.score_pages] == [
            18,
            33,
            34,
        ]  # graphical order
        assert len(score) == 145  # logical order

        # Check encoding/decoding consistency
        score_hat = score
        for _ in range(2):
            score_hat = Score.from_project_zip(score_hat.as_project_zip())
        assert score.checksum() == score_hat.checksum()


if __name__ == "__main__":
    unittest.main()
