import functools
import json
import pathlib
import pickle
import numpy as np
from collections import defaultdict

from . import ALIGNMENT_APP_DATA_DIR, CACHE_DIR
from .score.detect_measures import detect_measures
from .align import align
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
from typing import Optional, Tuple, List


TAGS = """
a_lesson
andante_con_moto
berceuse
gymnopedie
liebestraum
lux_aeterna
maple_leaf_rag
omaggio_a_brahms
pastorale
pink_chinese_rag
prelude_in_c
promenade
the_maid
""".strip().splitlines()
HAS_REPEATS = ["a_lesson", "maple_leaf_rag"]
DPI = 200
assert len(TAGS) == 13

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


def _convert_from_bbox(bbox: BoundingBox, page_image: Image) -> Tuple[float, float, float, float]:
    x0_abs = bbox.left * page_image.width
    y0_abs = bbox.top * page_image.height
    x1_abs = bbox.width * page_image.width + x0_abs
    y1_abs = bbox.height * page_image.height + y0_abs
    return (x0_abs, y0_abs, x1_abs, y1_abs)


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
    measure_bboxes = []
    for page_image in page_images:
        ms = _get_ordered_bboxes(detect_measures(page_image=page_image))
        measure_bboxes.append([_convert_from_bbox(bbox, page_image) for bbox in ms])
        lines = _get_line_measure_mappings(ms)
        line_ends = []
        for line in lines:
            end += len(line)
            line_ends.append(end)
        pages.append(line_ends)
        
    return pages, measure_bboxes


def _check_validity(tag: str) -> List[pathlib.PosixPath]:
    
    if ALIGNMENT_APP_DATA_DIR is None:
        raise Exception("ALIGNMENT_APP_DATA_DIR is not set")
    if tag not in TAGS:
        raise ValueError(f"Invalid tag: {tag}")
    dir_path = ALIGNMENT_APP_DATA_DIR / tag
    if not dir_path.is_dir():
        raise Exception(f"Directory not found: {dir_path}")
    file_names = list(pathlib.Path(dir_path).iterdir())
    if len(file_names) != 3:
        raise ValueError("Expected 3 files in directory")
    
    return file_names


def _get_pages_and_order(tag, page_images, graphical_order_to_page, graphical_order_to_bbox, logical_order, key_sigs=None, clefs=None):
    
    # Relativize bounding boxes
    graphical_order_to_bbox = {
        graphical_order: _convert_bbox(
            bbox_abs, page_images[graphical_order_to_page[graphical_order]]
        )
        for graphical_order, bbox_abs in graphical_order_to_bbox.items()
    }

    # Create pages
    page_measure_bboxes = defaultdict(list)
    for graphical_order, page_number in graphical_order_to_page.items():
        page_measure_bboxes[page_number].append(
            graphical_order_to_bbox[graphical_order]
        )
    pages = []
    for page_number in range(len(page_images)):
        measure_bboxes = page_measure_bboxes[page_number]
        unique_measures = []
        seen_keys = set()
        for bbox in measure_bboxes:
            key = bbox.get_unique_key()
            if key in seen_keys:
                assert tag == "lux_aeterna"  # one-off bug
            else:
                unique_measures.append(Measure(bbox=bbox))
            seen_keys.add(key)
        pages.append(
            Page(
                measures=unique_measures,
                image=page_images[page_number]
            )
        )

    # Create order
    order: MeasureOrder = []
    for i, graphical_index in enumerate(logical_order):
        page = pages[graphical_order_to_page[graphical_index]]
        m = page.lookup(graphical_order_to_bbox[graphical_index])
        m.key_signature = key_sigs[i] if key_sigs is not None else None
        m.clefs = clefs[i] if clefs is not None else None
        order.append((page, m))
    
    return pages, order


def _get_groundtruth_alignment_info(tag, json_path, page_images, get_times=False):
    # Load and parse JSON
    with open(json_path, "r") as f:
        alignment = json.load(f)["audio_score_alignment"]
    times = []
    logical_order = []
    graphical_order_to_page = {}
    graphical_order_to_bbox = {}
    for m in alignment:
        time_start = float(m["audio_start"])
        time_end = float(m["audio_end"])

        page_number = m["page_number"] - 1
        assert page_number >= 0 and page_number < len(page_images)
        if "bbox_number" in m:
            score_graphical_order = m["bbox_number"] - 1
        else:
            score_graphical_order = m["mapped_measure_number"] - 1
        assert score_graphical_order >= 0

        bbox_abs = m["measure_bbox"]

        # Repeat detected
        if score_graphical_order in graphical_order_to_page:
            assert tag in HAS_REPEATS
            assert graphical_order_to_page[score_graphical_order] == page_number
            assert graphical_order_to_bbox[score_graphical_order] == bbox_abs

        graphical_order_to_page[score_graphical_order] = page_number
        graphical_order_to_bbox[score_graphical_order] = bbox_abs
        logical_order.append(score_graphical_order)
        times.append((time_start, time_end))
    assert times == sorted(times)
    
    if not get_times:
        times = None
    
    return graphical_order_to_page, graphical_order_to_bbox, logical_order, times


def _load_key_and_clef_info(tag: str):
    
    key_sigs = _load_pkl(CACHE_DIR / "mesa13_info" / f"{tag}_key_sigs.pkl")
    clefs = _load_pkl(CACHE_DIR / "mesa13_info" / f"{tag}_clefs.pkl")
    
    return key_sigs, clefs


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
def load_mesa13_alignment(tag: str) -> AlignedScore:
    # Loads groundtruth alignment
    file_names = _check_validity(tag=tag)
    
    json_path = [f for f in file_names if f.suffix == ".json"][0]
    pdf_path = [f for f in file_names if f.suffix == ".pdf"][0]
    audio_path = [f for f in file_names if f not in (json_path, pdf_path)][0]

    # Load PDF
    page_images = list(PDF.from_path(pdf_path).as_page_images(dpi=DPI))
 
    graphical_order_to_page, graphical_order_to_bbox, logical_order, times = _get_groundtruth_alignment_info(tag, json_path, page_images, get_times=True)
    
    # Load key signatures, time signatures, clefs
    key_sigs, clefs = _load_key_and_clef_info(tag=tag)
    assert len(key_sigs) == len(logical_order)
    assert len(clefs) == len(logical_order)
    pages, order = _get_pages_and_order(tag, page_images, graphical_order_to_page, graphical_order_to_bbox, logical_order, key_sigs, clefs)

    return _get_aligned_score(pages, list(range(len(times))), [s for s, _ in times], order, max(e for _, e in times), audio_path, pdf_path)


@functools.lru_cache(maxsize=None)
def load_computed_alignment(tag: str, from_precomputed: Optional[bool] = True, use_repeats: Optional[bool] = True, use_measures: Optional[bool] = True, use_staff_metadata: Optional[bool] = True) -> AlignedScore:
    
    file_names = _check_validity(tag=tag)
        
    json_path = [f for f in file_names if f.suffix == ".json"][0]
    pdf_path = [f for f in file_names if f.suffix == ".pdf"][0]
    audio_path = [f for f in file_names if f not in (json_path, pdf_path)][0]
    
    # Load PDF
    page_images = list(PDF.from_path(pdf_path).as_page_images(dpi=DPI))
    
    # Load and parse alignment
    logical_order = []
    graphical_order_to_page = {}
    graphical_order_to_bbox = {}
    key_sigs, clefs = None, None
    measure_indices, times = None, None
    if not use_repeats and not use_measures and not use_staff_metadata:
        # strict
        # using detected measures
        if from_precomputed:
            with open(f"alignments/mesa13/strict/{tag}_aligned_score.zip", "rb") as f:
                return AlignedScore.from_project_zip(f.read())
        else:
            measure_num = 0
            page_images_temp = page_images
            if tag == "gymnopedie":
                page_images_temp = page_images[:2]
            for page_number,page_image in enumerate(page_images_temp):
                bboxes = detect_measures(page_image=page_image)
                bboxes = _get_ordered_bboxes(bboxes)
                for bbox in bboxes:
                    graphical_order_to_page[measure_num] = page_number
                    graphical_order_to_bbox[measure_num] = _convert_from_bbox(bbox=bbox, page_image=page_image)
                    logical_order.append(measure_num)
                    measure_num += 1
    elif use_repeats and not use_measures and not use_staff_metadata:
        # our recommended setting
        if from_precomputed:
            with open(f"alignments/mesa13/jtlr/{tag}_aligned_score.zip", "rb") as f:
                return AlignedScore.from_project_zip(f.read())
        else:
            if os.path.exists(f"alignments/mesa13/strict/{tag}_aligned_score.zip"):
                with open(f"alignments/mesa13/strict/{tag}_aligned_score.zip", "rb") as f:
                    aligned_score = AlignedScore.from_project_zip(f.read())
            else:
                aligned_score = load_computed_alignment(tag=tag, from_precomputed=False, use_repeats=False, use_measures=False, use_staff_metadata=False)
            assert len(aligned_score.score_pages) == len(page_images)
            measure_num = 0
            for i, p in enumerate(aligned_score.score_pages):
                for m in p:
                    graphical_order_to_page[measure_num] = i
                    graphical_order_to_bbox[measure_num] = _convert_from_bbox(bbox=m.bbox, page_image=page_images[i])
                    logical_order.append(measure_num)
                    measure_num += 1
            
            if tag in ["a_lesson", "maple_leaf_rag"]:
                logical_order = []
                if tag == "a_lesson":
                    assert measure_num == 83
                    a_lesson_repeats = [(0, 7+1), (1, 22+1), (8, 66+1), (54, 82+1), (67, 82+1)]
                    for rng in a_lesson_repeats:
                        logical_order.extend(np.arange(rng[0], rng[1]))
                elif tag == "maple_leaf_rag":
                    assert measure_num == 88
                    maple_leaf_repeats = [(0, 16+1), (1, 15+1),(17, 34+1),(19, 33+1),(35, 68+1),(53, 67+1),(69, 86+1),(71, 85+1)]
                    for rng in maple_leaf_repeats:
                        logical_order.extend(np.arange(rng[0], rng[1]))
                    logical_order.append(87)
                
    elif use_repeats and use_measures and not use_staff_metadata:
        # weak
        if from_precomputed:
            with open(f"alignments/mesa13/weak/{tag}_aligned_score.zip", "rb") as f:
                return AlignedScore.from_project_zip(f.read())
        else:
            graphical_order_to_page, graphical_order_to_bbox, logical_order, times = _get_groundtruth_alignment_info(tag, json_path, page_images)
    elif use_repeats and use_measures and use_staff_metadata:
        # strong
        if from_precomputed:
            with open(f"alignments/mesa13/strong/{tag}_aligned_score.zip", "rb") as f:
                return AlignedScore.from_project_zip(f.read())
        else:
            graphical_order_to_page, graphical_order_to_bbox, logical_order, times = _get_groundtruth_alignment_info(tag, json_path, page_images)
            key_sigs, clefs = _load_key_and_clef_info(tag=tag)
            assert len(key_sigs) == len(logical_order)
            assert len(clefs) == len(logical_order)
    else:
        raise Exception(f"Choose from [], [R], [R, M], and [R, M, S]")
   
    if times is not None:
        assert times == sorted(times)
        
    pages, order = _get_pages_and_order(tag, page_images, graphical_order_to_page, graphical_order_to_bbox, logical_order, key_sigs, clefs)

    if measure_indices is None and times is None:
        score = Score(score_pages=pages, logical_order=order, score_pdf=PDF.from_path(pdf_path))
        performance = Audio.from_path(audio_path)
        return align(score=score, performance=performance)
    else:
        return _get_aligned_score(pages, measure_indices, times, order, times[-1], audio_path, pdf_path)
  

@functools.lru_cache(maxsize=None)
def load_computed_alignment_baseline(tag: str, from_precomputed=True) -> AlignedScore:
    # baseline: hierDTW
    
    file_names = _check_validity(tag=tag)
        
    json_path = [f for f in file_names if f.suffix == ".json"][0]
    pdf_path = [f for f in file_names if f.suffix == ".pdf"][0]
    audio_path = [f for f in file_names if f not in (json_path, pdf_path)][0]
    
    # Load PDF
    page_images = list(PDF.from_path(pdf_path).as_page_images(dpi=DPI))
    
    logical_order = []
    graphical_order_to_page = {}
    graphical_order_to_bbox = {}
    if from_precomputed:
        with open(f"alignments/mesa13/baseline/{tag}_aligned_score.zip", "rb") as f:
            return AlignedScore.from_project_zip(f.read())
    else:
        with open(ALIGNMENT_APP_DATA_DIR / f'hyp/hierarchicalDTW_v3.AD/{tag}.pkl', 'rb') as f:
            a = pickle.load(f)
            wp = a['wp']

        with open(ALIGNMENT_APP_DATA_DIR / f'bscore/{tag}.pkl', 'rb') as f:
            midi = pickle.load(f)
        
        line_mappings = _get_line_mappings(midi, wp)    
        page_images = list(PDF.from_path(pdf_path).as_page_images(dpi=DPI))
        pages, bboxes = _get_pages(page_images)
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
        if len(measure_indices) < len(times):
            measure_indices.extend([measure_indices[-1]] * (len(times) - len(measure_indices)))
        
        measure_idx = 0
        for page_number,page_bboxes in enumerate(bboxes):
            for bbox in page_bboxes:
                graphical_order_to_page[measure_idx] = page_number
                graphical_order_to_bbox[measure_idx] = bbox
                logical_order.append(measure_idx)
                measure_idx += 1
        
    assert times == sorted(times)
    
    pages, order = _get_pages_and_order(tag, page_images, graphical_order_to_page, graphical_order_to_bbox, logical_order)
    
    return _get_aligned_score(pages, measure_indices, times, order, times[-1], audio_path, pdf_path)


@functools.lru_cache(maxsize=None)
def load_audio_feature_alignment(tag: str, audio_feature: str) -> AlignedScore:
    
    file_names = _check_validity(tag=tag)
        
    json_path = [f for f in file_names if f.suffix == ".json"][0]
    pdf_path = [f for f in file_names if f.suffix == ".pdf"][0]
    audio_path = [f for f in file_names if f not in (json_path, pdf_path)][0]
    
    # Load PDF
    page_images = list(PDF.from_path(pdf_path).as_page_images(dpi=DPI))
    
    graphical_order_to_page, graphical_order_to_bbox, logical_order, times = _get_groundtruth_alignment_info(tag, json_path, page_images, get_times=False)
    
    computed_alignment = _load_pkl(ALIGNMENT_APP_DATA_DIR / "repr_exps" / "strong" / f"{tag}_{audio_feature}_alignment.pkl")
    measure_indices, times = list(computed_alignment['measure_indices']), list(computed_alignment['times'])
    assert times == sorted(times)

    key_sigs, clefs = _load_key_and_clef_info(tag=tag)
    assert len(key_sigs) == len(logical_order)
    assert len(clefs) == len(logical_order)
    
    pages, order = _get_pages_and_order(tag, page_images, graphical_order_to_page, graphical_order_to_bbox, logical_order, key_sigs, clefs)
    
    return _get_aligned_score(pages, measure_indices, times, order, times[-1], audio_path, pdf_path)