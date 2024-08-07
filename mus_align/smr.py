import functools
import json
import pathlib
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List

from . import SMR_DATA_DIR
from .types import (
    PDF,
    AlignedScore,
    Audio,
    BoundingBox,
    Image,
    Measure,
    MeasureAwareAlignment,
    MeasureOrder,
    Page,
    Score
)
from .score.detect_measures import detect_measures
from .align import align
TAGS = [str(i) for i in range(1, 101)]
DPI = 200


def _check_validity(tag: str):
    if SMR_DATA_DIR is None:
        raise Exception("SMR_DATA_DIR is not set")
    if tag not in TAGS:
        raise ValueError(f"Invalid tag: {tag}")

def _load_pkl(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def _convert_bbox(bbox_abs: dict, page_image: Image) -> BoundingBox:
    x0_abs, y0_abs, x1_abs, y1_abs = bbox_abs
    return BoundingBox(
        left=x0_abs / page_image.width,
        top=y0_abs / page_image.height,
        width=(x1_abs - x0_abs) / page_image.width,
        height=(y1_abs - y0_abs) / page_image.height,
    )


def _get_ordered_bboxes(bboxes: List[BoundingBox]) -> List[BoundingBox]:
    bboxes.sort(key=lambda b: b.top)
    lines = []

    while bboxes:
        bbox = bboxes[0]
        half_measure_height = bbox.height / 2
        line = [b for b in bboxes if abs(b.top - bbox.top) < half_measure_height]
        line.sort(key=lambda b: b.left)
        lines.append(line)
        bboxes = [b for b in bboxes if abs(b.top - bbox.top) >= half_measure_height]

    return [bbox for line in lines for bbox in line]


def _get_line_mappings(midi, wp):
    x = np.ceil(np.interp(np.arange(len(midi['note_events'])), wp[:,1], wp[:,0]))
    line_mappings = []
    curr_line = 0
    note_events = []
    for i, line in enumerate(x):
        line = int(line)
        if line > curr_line:
            if note_events != []:
                line_mappings.append((curr_line, note_events))
            note_events = [i]
        elif line == curr_line:
            note_events.append(i)
        else: # line < curr_lines
            if note_events != []:
                line_mappings.append((curr_line, note_events))
            note_events = [i]
        curr_line = line
    if note_events != []:
        line_mappings.append((int(x[-1]), note_events))
        
    return line_mappings


def _get_line_measure_mappings(bboxes: List[BoundingBox]): # one page at a time
    
    bboxes.sort(key=lambda b: b.top)
    lines = []

    while bboxes:
        bbox = bboxes[0]
        half_measure_height = bbox.height / 2
        line = [b for b in bboxes if abs(b.top - bbox.top) < half_measure_height]
        line.sort(key=lambda b: b.left)
        lines.append(line)
        bboxes = [b for b in bboxes if abs(b.top - bbox.top) >= half_measure_height]
        
    return lines


def _get_pages(page_images):
    
    end = 0
    pages = []
    for page_image in page_images:
        ms = _get_ordered_bboxes(detect_measures(page_image=page_image))
        lines = _get_line_measure_mappings(ms)
        line_ends = []
        for line in lines:
            end += len(line)
            line_ends.append(end)
        pages.append(line_ends)
        
    return pages


def _get_pages_and_order(tag: str, pdf_path, audio_path):
    
    # Load PDF
    page_images = list(PDF.from_path(pdf_path).as_page_images(dpi=DPI))
    
    # Create pages and logical order 
    page_measures = {i: [] for i in range(len(page_images))}
    for page_idx,page_image in enumerate(page_images):
        bboxes = _get_ordered_bboxes(detect_measures(page_image))
        page_measures[page_idx] = [Measure(bbox=bbox) for bbox in bboxes]
            
    pages = [Page(measures=page_measures[i], image=page_images[i]) for i in range(len(page_images))]
    logical_order = [(page, measure) for page in pages for measure in page]
    
    return pages, logical_order


def _get_aligned_score(pages, measure_indices, times, logical_order, max_time, audio_path, pdf_path):
    return AlignedScore(
        score_pages=pages,
        alignment=MeasureAwareAlignment(
            measure_indices=measure_indices,
            times=times,
            logical_order=logical_order,
            max_time=max_time,
        ),
        performance=Audio.from_path(audio_path),
        score_pdf=PDF.from_path(pdf_path),
    )


@functools.lru_cache(maxsize=None)
def is_valid(tag: str) -> bool:
    if SMR_DATA_DIR is None:
        raise Exception("SMR_DATA_DIR is not set")
    score_info_path = SMR_DATA_DIR / "score_info" / f"p{tag}.scoreinfo.csv"
    pdf_path = SMR_DATA_DIR / "pdfs" / f"p{tag}.pdf"
    page_images = list(PDF.from_path(pdf_path).as_page_images(dpi=DPI))
    pages = []
    for page_idx,page_image in enumerate(page_images):
        bboxes = _get_ordered_bboxes(detect_measures(page_image))
        pages.append(Page(measures=[Measure(bbox=bbox) for bbox in bboxes],image=page_image))
    page_lines = [len(line) for page in pages for line in page.iter_graphical_order()]
    
    reference_lines = []
    df = pd.read_csv(score_info_path)
    for line in range(df.shape[0]):
        start, end = df['StartMeasure'][line],  df['EndMeasure'][line]
        reference_lines.append(end - start + 1)
    
    return sum(page_lines) == sum(reference_lines) and (page_lines == reference_lines)


@functools.lru_cache(maxsize=None)
def load_smr_alignment(tag: str) -> AlignedScore:
    # Groundtruth alignment
    
    _check_validity(tag=tag)
    
    audio_info_path = SMR_DATA_DIR / "midi_info" / f"p{tag}_midinfo.csv"
    df = pd.read_csv(audio_info_path, header=None)
    measure_indices, times = list(df[0]-df[0][0]), list(df[1])
    assert times == sorted(times)
    
    pdf_path = SMR_DATA_DIR / "pdfs" / f"p{tag}.pdf"
    audio_path = SMR_DATA_DIR / "wavs" / f"p{tag}.wav"
    pages, logical_order = _get_pages_and_order(tag, pdf_path, audio_path)
        
    return _get_aligned_score(pages, measure_indices, times, logical_order, times[-1], audio_path, pdf_path)
    
    
@functools.lru_cache(maxsize=None)
def load_computed_alignment(tag: str, from_precomputed=False) -> AlignedScore:
    
    _check_validity(tag=tag)
    
    if from_precomputed:
        with open(f"alignments/smr/jltr/{tag}_aligned_score.zip", "rb") as f:
            return AlignedScore.from_project_zip(f.read())
    else:
        pdf_path = SMR_DATA_DIR / "pdfs" / f"p{tag}.pdf"
        audio_path = SMR_DATA_DIR / "wavs" / f"p{tag}.wav"
        pages, logical_order = _get_pages_and_order(tag, pdf_path, audio_path)
        score = Score(score_pages=pages, logical_order=logical_order, score_pdf=PDF.from_path(pdf_path))
        performance = Audio.from_path(audio_path)
        return align(score=score, performance=performance)
    
    
@functools.lru_cache(maxsize=None)
def load_computed_alignment_baseline(tag: str, from_precomputed=False) -> AlignedScore: 
    
    _check_validity(tag=tag)
    
    if from_precomputed:
        with open(f"alignments/smr/baseline/{tag}_aligned_score.zip", "rb") as f:
            return AlignedScore.from_project_zip(f.read())
    else:
        with open(SMR_DATA_DIR / f'hyp/hierarchicalDTW_v3.AD/p{tag}.pkl', 'rb') as f:
            a = pickle.load(f)
            wp = a['wp']

        with open(SMR_DATA_DIR / f'bscore/p{tag}.pkl', 'rb') as f:
            midi = pickle.load(f)

        line_mappings = _get_line_mappings(midi, wp)    
        page_images = list(PDF.from_path(SMR_DATA_DIR / "pdfs" / f"p{tag}.pdf").as_page_images(dpi=DPI))
        pages = _get_pages(page_images)
        pages = [page for lst in pages for page in lst]
        measure_indices = []
        for line_idx, note_events in line_mappings:
            if line_idx < len(pages):
                prev_line = pages[line_idx-1] if line_idx > 0 else 0
                line = pages[line_idx]
                measure_indices.extend(np.linspace(prev_line, line, len(note_events)))
            else:
                measure_indices.extend([pages[-1]-1]*len(note_events))
        times = [event[1] for event in midi['note_events']]
        assert times == sorted(times)
        if len(measure_indices) < len(times):
            measure_indices.extend([measure_indices[-1]] * (len(times) - len(measure_indices)))

    pdf_path = SMR_DATA_DIR / "pdfs" / f"p{tag}.pdf"
    audio_path = SMR_DATA_DIR / "wavs" / f"p{tag}.wav"        
    pages, logical_order = _get_pages_and_order(tag, pdf_path, audio_path)
        
    return _get_aligned_score(pages, measure_indices, times, logical_order, times[-1], audio_path, pdf_path)


@functools.lru_cache(maxsize=None)
def load_audio_feature_alignment(tag: str, audio_feature: str) -> AlignedScore:
    
    _check_validity(tag=tag)
    
    pdf_path = SMR_DATA_DIR / "pdfs" / f"p{tag}.pdf"
    audio_path = SMR_DATA_DIR / "wavs" / f"p{tag}.wav"
    pages, logical_order =_get_pages_and_order(tag, pdf_path, audio_path)

    computed_alignment = _load_pkl(SMR_DATA_DIR / "repr_exps" / "strong" / f"p{tag}_{audio_feature}_alignment.pkl")
    measure_indices, times = list(computed_alignment['measure_indices']), list(computed_alignment['times'])
    assert times == sorted(times)

    return _get_aligned_score(pages, measure_indices, times, logical_order, times[-1], audio_path, pdf_path)
    
