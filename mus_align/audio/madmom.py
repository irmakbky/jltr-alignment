import functools
import io
import math
import tempfile
from typing import List, Optional, Tuple

import librosa
import madmom
import pretty_midi
from scipy.io.wavfile import write as wavwrite

from ..types import Audio


@functools.lru_cache(maxsize=1)
def _get_beat_processor_cached():
    return madmom.features.RNNDownBeatProcessor()


@functools.lru_cache(maxsize=8)
def _get_downbeat_processor_cached(*args, **kwargs):
    for k, v in kwargs.items():
        if isinstance(v, tuple):
            kwargs[k] = list(v)
    return madmom.features.DBNDownBeatTrackingProcessor(*args, **kwargs)


def detect_beats_and_downbeats(
    audio: Audio,
    clip_start: float = 0.0,
    clip_duration: Optional[float] = None,
    beats_per_bar_hints: List[int] = [3, 4],
    bpm_hint: Optional[float] = None,
    min_bpm: float = 55.0,
    max_bpm: float = 215.0,
    num_tempi: int = 60,
    transition_lambda: int = 100,
    observation_lambda: int = 16,
    threshold: float = 0.05,
    correct: bool = True,
    fps: float = 100.0,
) -> Tuple[List[float], List[int]]:
    # Create processor
    if bpm_hint is not None:
        min_bpm = bpm_hint * math.pow(2, -0.5)
        max_bpm = bpm_hint * math.pow(2, 0.5)
    beat_proc = _get_beat_processor_cached()
    downbeat_proc = _get_downbeat_processor_cached(
        beats_per_bar=tuple(beats_per_bar_hints),
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        num_tempi=num_tempi,
        transition_lambda=transition_lambda,
        observation_lambda=observation_lambda,
        threshold=threshold,
        correct=correct,
        fps=fps,
    )

    # Run madmom
    audio, sr = librosa.load(
        io.BytesIO(audio.raw),
        sr=44100,
        offset=clip_start,
        duration=clip_duration,
        mono=True,
    )
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        wavwrite(f.name, sr, audio)
        activations = beat_proc(f.name)
    result = downbeat_proc(activations)
    times = (result[:, 0] + clip_start).tolist()
    assert all(t >= 0.0 for t in times)
    indices = [round(t) - 1 for t in result[:, 1]]
    assert all(0 <= i and i < max(beats_per_bar_hints) for i in indices)
    assert len(times) == len(indices)
    return times, indices


def synthesize_click(
    beat_times,
    beat_indices,
    sr=44100,
    downbeat_pitch=34,
    upbeat_pitch=32,
    downbeat_velocity=100,
    upbeat_velocity=75,
):
    midi = pretty_midi.PrettyMIDI()
    click = pretty_midi.Instrument(program=0, is_drum=True)
    for t, i in zip(beat_times, beat_indices):
        click.notes.append(
            pretty_midi.Note(
                start=t,
                end=t + 0.1,
                pitch=downbeat_pitch if i == 0 else upbeat_pitch,
                velocity=downbeat_velocity if i == 0 else upbeat_velocity,
            )
        )
    midi.instruments.append(click)
    audio = midi.fluidsynth(fs=sr)
    return sr, audio
