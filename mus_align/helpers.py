from .types import MeasureOrder


def convert_annotations_to_absolute(order: MeasureOrder):
    current_key_signature = None
    current_time_signature = None
    current_clefs = None
    for _, measure in order:
        if measure.key_signature is not None:
            current_key_signature = measure.key_signature
        if measure.time_signature is not None:
            current_time_signature = measure.time_signature
        if measure.clefs is not None:
            current_clefs = measure.clefs
        measure.key_signature = current_key_signature
        measure.time_signature = current_time_signature
        measure.clefs = current_clefs


def convert_annotations_to_changes(order: MeasureOrder):
    raise NotImplementedError()
