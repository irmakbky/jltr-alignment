import base64
import json
import pathlib
import zipfile
from io import BytesIO
from typing import List, Tuple

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from . import CACHE_DIR
from .align import align
from .audio.madmom import detect_beats_and_downbeats
from .mesa13 import load_mesa13_alignment
from .score.detect_measures import detect_measures
from .types import PDF, Audio, Image, Score

app = Flask(
    __name__,
    static_folder=str(CACHE_DIR / "frontend"),
    static_url_path="",
)
CORS(app)

_MAX_CACHE_SIZE = 128
_PROJECT_CACHE: List[Tuple[Tuple[str, str], Score, Audio]] = []


def _update_cache(score: Score, performance: Audio) -> Tuple[str, str]:
    global _PROJECT_CACHE
    uuid = (score.checksum(), performance.checksum())
    try:
        _retrieve_from_cache(uuid)
    except KeyError:
        if len(_PROJECT_CACHE) >= _MAX_CACHE_SIZE:
            # evict oldest
            _PROJECT_CACHE.pop(0)
        _PROJECT_CACHE.append((uuid, score, performance))
    return uuid


def _retrieve_from_cache(uuid) -> Tuple[Score, Audio]:
    global _PROJECT_CACHE
    print(_PROJECT_CACHE, uuid)
    for c, score, audio in _PROJECT_CACHE:
        if uuid == c:
            return (score, audio)
    raise KeyError(f"Score with uuid {uuid} not found in cache.")


@app.route("/")
def index():
    return send_from_directory(str(CACHE_DIR / "frontend"), "index.html")


@app.route("/mesa13/<tag>.zip")
def mesa13(tag):
    aligned_score = load_mesa13_alignment(tag)
    result_zip_bytes = aligned_score.as_project_zip()
    return send_file(
        BytesIO(result_zip_bytes),
        download_name=f"{tag}.zip",
        as_attachment=True,
    )


@app.route("/pdf_to_images", methods=["POST"])
def pdf_to_page_images_endpoint():
    if "pdf" not in request.files:
        return jsonify(error="PDF file is required."), 400

    pdf_file = request.files["pdf"]
    dpi = request.form.get("dpi", 200, type=int)

    pdf = PDF(raw=pdf_file.read())
    images = list(pdf.as_page_images(dpi=dpi))
    images_base64 = []
    for image in images:
        img_str = base64.b64encode(image.raw_png).decode()
        images_base64.append(img_str)
    return jsonify(images=images_base64)


@app.route("/detect_measures", methods=["POST"])
def detect_measures_endpoint():
    if "image" not in request.files:
        return jsonify(error="Image file is required."), 400

    image_file = request.files["image"]
    confidence_threshold = request.form.get("confidence_threshold", 0.5, type=float)

    try:
        image = Image.open(image_file)
        bounding_boxes = detect_measures(image, confidence_threshold)
        print(len(bounding_boxes))
        json_bounding_boxes = [box.as_json() for box in bounding_boxes]
        return jsonify(bounding_boxes=json_bounding_boxes)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/detect_beats", methods=["POST"])
def detect_beats_endpoint():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    # Optional parameters
    clip_start = request.form.get("clip_start", 0.0, type=float)
    clip_duration = request.form.get("clip_duration", type=float)
    beats_per_bar_hints = json.loads(request.form.get("beats_per_bar_hints", "[3, 4]"))
    bpm_hint = request.form.get("bpm_hint", type=float)

    beat_times, beat_indices = detect_beats_and_downbeats(
        audio_bytes=audio_bytes,
        clip_start=clip_start,
        clip_duration=clip_duration,
        beats_per_bar_hints=beats_per_bar_hints,
        bpm_hint=bpm_hint,
    )

    return jsonify({"times": beat_times, "indices": beat_indices})


@app.route("/upload_project", methods=["POST"])
def upload_project_endpoint():
    if "project" not in request.files:
        return jsonify(error="Zip file is required."), 400

    # Get the zip file from the request
    zip_file = request.files["project"]
    zip_bytes = zip_file.read()

    # Get the audio from the zip file
    performance = None
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
        for file_name in z.namelist():
            if file_name.startswith("performance"):
                ext = pathlib.Path(file_name).suffix
                performance = Audio(raw=z.read(file_name), ext=ext)
    if performance is None:
        return jsonify(error="Performance audio file is required."), 400

    # Call Score.from_project_zip to process the zip file
    score = Score.from_project_zip(zip_bytes, check_consistency=False)

    return jsonify(uuid=_update_cache(score, performance))


@app.route("/align", methods=["POST"])
def align_endpoint():
    # Get the uuid from the request
    uuid = request.json.get("uuid")

    # Retrieve the stored score and performance from the cache
    try:
        if uuid is None:
            raise KeyError()
        score, performance = _retrieve_from_cache(tuple(uuid))
    except KeyError:
        return jsonify(error="Project not found in cache."), 404

    # Retrieve the hard constraints from the request
    hard_constraints = request.json.get("hard_constraints")

    # Align the score and performance
    aligned_score = align(score, performance, hard_constraints=hard_constraints)

    return jsonify(
        alignment={
            "measure_indices": aligned_score.alignment.measure_indices,
            "times": aligned_score.alignment.times,
            "max_time": aligned_score.alignment.max_time,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
