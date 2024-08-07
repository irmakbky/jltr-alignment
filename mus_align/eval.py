import numpy as np
import scipy
import librosa

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

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
)


def _calculate_centerpoint(bbox: BoundingBox) -> np.ndarray:
    return np.array([bbox.left + bbox.width/2, bbox.top + bbox.height/2])


def _align_pred_to_ref_measures(pred: AlignedScore, ref: AlignedScore) -> Dict[int, int]:
    
    pred_graphical_order_bboxes = {m.bbox.get_unique_key(): (m.bbox, i) for i, p in enumerate(pred.score_pages) for m in p}
    ref_graphical_order_bboxes = {m.bbox: i for i, p in enumerate(ref.score_pages) for m in p}
    
    graphical_measure_mappings = {}
    for m_key, (bbox, page_num) in pred_graphical_order_bboxes.items():
        candidate_measure_bboxes = [k for k in ref_graphical_order_bboxes if ref_graphical_order_bboxes[k] == page_num]
        center_pred = _calculate_centerpoint(bbox)
        distances = [np.linalg.norm(center_pred-_calculate_centerpoint(b)) for b in candidate_measure_bboxes]
        if len(distances) == 0:
            graphical_measure_mappings[m_key] = -2
            continue
        best_measure_idx = candidate_measure_bboxes[np.argmin(distances)]
        graphical_measure_mappings[m_key] = best_measure_idx.get_unique_key()
        
    ref_logical_order_bboxes_nums = {k.get_unique_key(): [] for k in ref_graphical_order_bboxes.keys()}    
    measure_num = 0
    for p, m in ref.logical_order:
        ref_logical_order_bboxes_nums[m.bbox.get_unique_key()].append(measure_num)
        measure_num += 1
    
    mapping = {}
    pred_seen_unique_keys = {k: 0 for k in pred_graphical_order_bboxes}
    for i, (p, m) in enumerate(pred.logical_order):
        m_key = m.bbox.get_unique_key()
        best_gt_unique_key = graphical_measure_mappings[m_key]
        if best_gt_unique_key == -2:
            mapping[i] = -2
            continue
        candidate_measure_nums = ref_logical_order_bboxes_nums[best_gt_unique_key]
        if len(candidate_measure_nums) == 0:
            best_index = -2
        elif len(candidate_measure_nums) == 1:
            best_index = candidate_measure_nums[0]
        else:
            idx = min(pred_seen_unique_keys[m_key], len(candidate_measure_nums)-1)
            best_index = candidate_measure_nums[idx]
        
        mapping[i] = best_index
        pred_seen_unique_keys[m_key] += 1
    
    return mapping

def evaluate(pred: AlignedScore,
             ref: AlignedScore,
             remap_measures: Optional[bool] = True,
             frame_rate: Optional[float] = 0.01,
             error_boundary: Optional[float] = 0.5,
            ) -> float:
    
    pred_measures = pred.alignment.measure_indices
    
    # need to match detected measure indices to ref measure indices to accurately evaluate
    if remap_measures:
        new_measure_mappings = _align_pred_to_ref_measures(pred, ref)
        pred_measure_indices, new_pred_measures_indices = list(new_measure_mappings.keys()), list(new_measure_mappings.values())
        f0 =  scipy.interpolate.interp1d(pred_measure_indices, new_pred_measures_indices, fill_value='extrapolate')
        pred_measures = f0(pred_measures)
    
    eval_times = np.linspace(0, pred.alignment.times[-1], int(pred.alignment.times[-1]/frame_rate))
    f1 = scipy.interpolate.interp1d(pred.alignment.times, pred_measures, fill_value='extrapolate')
    f2 = scipy.interpolate.interp1d(ref.alignment.times, ref.alignment.measure_indices, fill_value='extrapolate')
    
    pred_eval_measures = f1(eval_times)
    ref_eval_measures = f2(eval_times)
    
    acc = np.sum(np.abs(ref_eval_measures - pred_eval_measures) <= error_boundary) / len(eval_times)
    
    return acc, np.abs(ref_eval_measures - pred_eval_measures)
    