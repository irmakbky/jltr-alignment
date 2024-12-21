import os
import json
import numpy as np
from pathlib import Path

from mus_align.types import *
from mus_align.score.detect_measures import detect_measures

imagepaths = [os.path.join(root, file) for root, dirs, files in os.walk("images") for file in files if file[-4:]==".png"]
    
errors = []
for imagepath in imagepaths:
    try:
        page_image = Image.from_path(imagepath)
        bboxes = detect_measures(page_image)
        measures = [Measure(bbox=bbox) for bbox in bboxes]
        page = Page(measures=measures,image=page_image)
        lines = page.iter_graphical_order()
        bboxes_json = [[m.bbox.as_json() for m in line] for line in lines]
        filepath = Path("detected_bboxes")/f"{imagepath[:-4]}_bboxes.json"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(bboxes_json, f, indent=4)
    except:
        errors.append(image)

with open("detected_bboxes/bboxes_errors.json", 'w') as f:
    json.dump(errors, f)
