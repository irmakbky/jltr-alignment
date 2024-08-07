import functools
from typing import List, Optional, Tuple

import librosa
import numpy as np

from .audio.onf import onsets_and_frames
from .score.bootleg import get_pianoroll
from .score.detect_noteheads import detect_noteheads
from .types import AlignedScore, Audio, MeasureAwareAlignment, Score


@functools.lru_cache(maxsize=128)
def extract_bootleg(score: Score):
    pianorolls = []
    for p, m in score.logical_order:
        img = p.crop_measure(m)
        pianorolls.append(
            get_pianoroll(
                detect_noteheads(img.pil_image, p.num_staves, p.num_staves),
                clefs=m.clefs,
                key=m.key_signature,
                transposition_factor=(
                    0 if score.transposition is None else score.transposition
                ),
            )
        )
    bootleg_pianoroll = np.concatenate(pianorolls, axis=1)
    return bootleg_pianoroll


@functools.lru_cache(maxsize=128)
def extract_onset_probs(performance: Audio):
    raw_preds, _ = onsets_and_frames(performance)
    onset_probs = raw_preds["onset_probs"][0].T
    return onset_probs


def align(
    score: Score,
    performance: Audio,
    hard_constraints: Optional[List[Tuple[float, float]]] = None,
) -> AlignedScore:
    if hard_constraints is not None:
        raise NotImplementedError()

    bootleg_pianoroll = extract_bootleg(score)
    onset_probs = extract_onset_probs(performance)

    D, wp = librosa.sequence.dtw(bootleg_pianoroll, onset_probs)
    frame_rate = 16000 / 512
    measure_indices = [i / 48 for i in wp[:, 0][::-1]]
    times = [frame / frame_rate for frame in wp[:, 1][::-1]]

    alignment = MeasureAwareAlignment(
        measure_indices=measure_indices,
        times=times,
        logical_order=score.logical_order,
        max_time=onset_probs.shape[1] / frame_rate,
    )

    return AlignedScore(
        score_pages=score.score_pages,
        alignment=alignment,
        performance=performance,
        score_pdf=score.score_pdf,
    )
