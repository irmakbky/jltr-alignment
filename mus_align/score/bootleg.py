import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
from librosa.display import specshow
from PIL import Image

from ..types import Clef, KeySignature, Score


def pitch_to_piano_key(pitch, octave):
    note_name = f"{pitch}{octave}"
    return pretty_midi.note_name_to_number(note_name) - 21


# In[3]:


def get_scale_notes(clef, sharps, flats, transposition_factor):
    if clef.name == "TREBLE":
        start_note = 1
    elif clef.name == "BASS":
        start_note = 3
    else:
        start_note = 0
        print("clef not recognized. assuming start note is A")
    start_note += transposition_factor
    notes = ["A", "B", "C", "D", "E", "F", "G"]
    for i, note in enumerate(notes):
        if note in sharps:
            notes[i] = note + "#"
        elif note in flats:
            notes[i] = note + "b"
    return notes[start_note:] + notes[:start_note]


# In[4]:


def get_info_from_key_sig(clefs, key_sig, transposition_factor):

    key_sigs_info = {
        "A_FLAT_MAJOR": ([], ["A", "B", "D", "E"]),
        "A_MAJOR": (["C", "F", "G"], []),
        "B_FLAT_MAJOR": ([], ["B", "E"]),
        "B_MAJOR": (["C", "D", "F", "G", "A"], []),
        "C_FLAT_MAJOR": ([], ["A", "B", "C", "D", "E", "F", "G"]),
        "C_MAJOR": ([], []),
        "C_SHARP_MAJOR": (["A", "B", "C", "D", "E", "F", "G"], []),
        "D_FLAT_MAJOR": ([], ["D", "E", "G", "A", "B"]),
        "D_MAJOR": (["F", "C"], []),
        "E_FLAT_MAJOR": ([], ["E", "A", "B"]),
        "E_MAJOR": (["F", "G", "C", "D"], []),
        "F_MAJOR": ([], ["B"]),
        "F_SHARP_MAJOR": (["A", "C", "D", "E", "F", "G"], []),
        "G_MAJOR": (["F"], []),
        "G_FLAT_MAJOR": ([], ["A", "B", "C", "D", "E", "G"]),
    }

    sharps, flats = key_sigs_info[key_sig.name]
    return {
        clef.name: get_scale_notes(clef, sharps, flats, transposition_factor)
        for clef in clefs
    }


# In[5]:


def vals_to_piano_keys(val, clef, notes):
    piano_keys = []
    if clef.name == "TREBLE":
        if val < -6:
            octave = 3
        elif val < 1:
            octave = 4
        elif val < 8:
            octave = 5
        else:
            octave = 6
        new_val = val % 7
        piano_keys.append(pitch_to_piano_key(notes[new_val], octave))
    elif clef.name == "BASS":
        if val < -8:
            octave = 1
        elif val < -1:
            octave = 2
        elif val < 6:
            octave = 3
        else:
            octave = 4
        new_val = val % 7
        piano_keys.append(pitch_to_piano_key(notes[new_val], octave))
    else:
        print("Clef not recognized.")
    return piano_keys


def get_pianoroll(nhdata, clefs, key, transposition_factor=0):

    if clefs is None:
        clefs = [Clef.TREBLE, Clef.BASS]
    if key is None:
        key = KeySignature.C_MAJOR

    if len(nhdata) > 0:
        pianokeys = []
        scale_notes = get_info_from_key_sig(clefs, key, transposition_factor)
        for row in nhdata:
            if row[-1] <= 0:
                clef = clefs[0]
            else:
                clef = clefs[1]
            pianokey = vals_to_piano_keys(row[2], clef, scale_notes[clef.name])
            pianokeys.append(pianokey)

        piano_rep = np.zeros((88, 48))
        norm = np.max(nhdata, axis=0)[1] + 1
        for i, row in enumerate(nhdata):
            start_pixel = int(np.round(row[1] * 48 / norm))
            if start_pixel == 48:
                start_pixel = 47
            piano_rep[pianokeys[i], start_pixel] = 1
    else:
        piano_rep = np.zeros((88, 48))

    return piano_rep
