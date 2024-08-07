import unittest

from .. import ALIGNMENT_APP_DATA_DIR
from ..types import Audio
from .madmom import detect_beats_and_downbeats

if ALIGNMENT_APP_DATA_DIR is None:
    raise FileNotFoundError("$ALIGNMENT_APP_DATA_DIR is not set.")


class TestMadmom(unittest.TestCase):
    def test_detect_beats_and_downbeats(self):
        audio_path = (ALIGNMENT_APP_DATA_DIR / "berceuse" / "complete.mp3").resolve()
        audio = Audio.from_path(audio_path)

        # Test detect_beats_and_downbeats
        beat_times, beat_indices = detect_beats_and_downbeats(
            audio, clip_start=0.0, clip_duration=30.0
        )
        assert len(beat_times) == 73
        assert len(beat_indices) == 73
        assert max(beat_indices) == 2


if __name__ == "__main__":
    unittest.main()
