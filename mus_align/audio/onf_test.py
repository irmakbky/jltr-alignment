import hashlib
import unittest

import numpy as np

from .. import ALIGNMENT_APP_DATA_DIR
from ..types import Audio
from .onf import transcribe

if ALIGNMENT_APP_DATA_DIR is None:
    raise FileNotFoundError("$ALIGNMENT_APP_DATA_DIR is not set.")

_EXPECTED_KEYS = [
    "frame_probs",
    "onset_probs",
    "frame_predictions",
    "onset_predictions",
    "offset_predictions",
    "velocity_values",
    "sequence_predictions",
    "sequence_ids",
    "sequence_labels",
    "frame_labels",
    "onset_labels",
    "metrics/frame_precision",
    "metrics/frame_recall",
    "metrics/frame_f1_score",
    "metrics/frame_accuracy",
    "metrics/frame_accuracy_without_true_negatives",
    "metrics/note_density",
    "metrics/note_precision",
    "metrics/note_recall",
    "metrics/note_f1_score",
    "metrics/note_with_velocity_precision",
    "metrics/note_with_velocity_recall",
    "metrics/note_with_velocity_f1_score",
    "metrics/note_with_offsets_precision",
    "metrics/note_with_offsets_recall",
    "metrics/note_with_offsets_f1_score",
    "metrics/note_with_offsets_velocity_precision",
    "metrics/note_with_offsets_velocity_recall",
    "metrics/note_with_offsets_velocity_f1_score",
]


class TestOnsetsAndFrames(unittest.TestCase):
    def test_onf(self):
        audio_path = (ALIGNMENT_APP_DATA_DIR / "berceuse" / "complete.mp3").resolve()
        audio = Audio.from_path(audio_path)

        # Test O&F
        for i in range(2):
            raw_preds, midi_bytes = transcribe(audio)
            self.assertEqual(frozenset(raw_preds.keys()), frozenset(_EXPECTED_KEYS))

            # Check onset predictions
            onset_preds = raw_preds["onset_predictions"]
            assert onset_preds.shape == (1, 8060, 88)
            assert onset_preds.dtype == np.bool
            num_onsets = onset_preds.astype(np.int64).sum()
            self.assertEqual(num_onsets, 3899)

            # Check onset probs
            onset_probs = raw_preds["onset_probs"]
            self.assertEqual(onset_probs.shape, (1, 8060, 88))
            # print(onset_probs.mean())
            # print(onset_probs.sum())
            self.assertLess(np.abs(onset_probs.mean() - 0.0066468967), 1e-4)
            self.assertLess(np.abs(onset_probs.sum() - 4714.5107), 1e-1)
            self.assertEqual(onset_probs.dtype, np.float32)

            # Check MIDI
            midi_checksum = hashlib.sha1(midi_bytes).hexdigest()
            # print(midi_checksum)
            # assert midi_checksum == "b7cb8b8c4f5ce45959ca3dc7f295ee25b66a2c1d"


if __name__ == "__main__":
    unittest.main()
