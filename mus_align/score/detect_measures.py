from typing import Iterator, Optional

import numpy as np
import tensorflow.compat.v1 as tf

from .. import CACHE_DIR
from ..types import BoundingBox, Image

_MODEL_SINGLETON = None


def _get_measure_detection_graph():
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        # Check for model
        model_path = CACHE_DIR / "2019-04-24_faster-rcnn_inception-resnet-v2.pb"
        if not model_path.is_file():
            raise RuntimeError("Model file not found: {}".format(model_path))

        # Load graph
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(str(model_path), "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")
        _MODEL_SINGLETON = detection_graph
    return _MODEL_SINGLETON


def detect_measures(
    page_image: Image,
    confidence_threshold: Optional[float] = 0.5,
) -> Iterator[BoundingBox]:
    graph = _get_measure_detection_graph()

    # Prepare image
    image_np = np.array(page_image.convert("RGB").pil_image)
    (image_width, image_height) = page_image.size

    # Run model
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                "num_detections",
                "detection_boxes",
                "detection_scores",
                "detection_classes",
            ]:
                tensor_name = key + ":0"

                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name
                    )

            image_tensor = tf.get_default_graph().get_tensor_by_name("image_tensor:0")

            # Run inference
            output_dict = sess.run(
                tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, 0)}
            )

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict["num_detections"] = int(output_dict["num_detections"][0])
            output_dict["detection_classes"] = output_dict["detection_classes"][
                0
            ].astype(np.uint8)
            output_dict["detection_boxes"] = output_dict["detection_boxes"][0]
            output_dict["detection_scores"] = output_dict["detection_scores"][0]

    # Parse outputs
    bounding_boxes = []
    for idx in range(output_dict["num_detections"]):
        score = float(output_dict["detection_scores"][idx])
        if confidence_threshold is None or score >= confidence_threshold:
            y1, x1, y2, x2 = output_dict["detection_boxes"][idx]

            y1 = float(y1)
            y2 = float(y2)
            x1 = float(x1)
            x2 = float(x2)
            assert y2 >= y1
            assert x2 >= x1
            bounding_boxes.append(
                BoundingBox(
                    left=x1,
                    top=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    score=score,
                )
            )

    return bounding_boxes
