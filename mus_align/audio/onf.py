import io
import tempfile

import librosa
import six
import tensorflow.compat.v1 as tf
from magenta.models.onsets_frames_transcription import (
    audio_label_data_utils as onf_audio_label_data_utils,
)
from magenta.models.onsets_frames_transcription import configs as onf_configs
from magenta.models.onsets_frames_transcription import data as onf_data
from magenta.models.onsets_frames_transcription import infer_util as onf_infer_util
from magenta.models.onsets_frames_transcription import train_util as onf_train_util
from note_seq import midi_io
from note_seq.protobuf import music_pb2
from scipy.io.wavfile import write as wavwrite

from .. import CACHE_DIR
from ..types import Audio

_MODEL_SINGLETON = None


def _get_onf_singleton():
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        # Check for model
        model_path = CACHE_DIR / "2019-04-24_faster-rcnn_inception-resnet-v2.pb"
        if not model_path.is_file():
            raise RuntimeError("Model file not found: {}".format(model_path))

        # Get GPU
        gpus = tf.config.experimental.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)

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


def transcribe(audio: Audio, *, model_dir="/model_ckpts/onf/train"):
    config = onf_configs.CONFIG_MAP["onsets_frames"]
    hparams = config.hparams
    hparams.parse("")
    hparams.batch_size = 1
    hparams.truncated_length_secs = 0
    data_fn = onf_data.provide_batch

    audio, _ = librosa.load(io.BytesIO(audio.raw), sr=hparams.sample_rate, mono=True)

    def create_example(filename, sample_rate, load_audio_with_librosa):
        """Processes an audio file into an Example proto."""
        wav_data = tf.gfile.Open(filename, "rb").read()
        example_list = list(
            onf_audio_label_data_utils.process_record(
                wav_data=wav_data,
                sample_rate=sample_rate,
                ns=music_pb2.NoteSequence(),
                # decode to handle filenames with extended characters.
                example_id=six.ensure_text(filename, "utf-8"),
                min_length=0,
                max_length=-1,
                allow_empty_notesequence=True,
                load_audio_with_librosa=load_audio_with_librosa,
            )
        )
        assert len(example_list) == 1
        return example_list[0].SerializeToString()

    with tf.Graph().as_default():
        examples = tf.placeholder(tf.string, [None])

        dataset = data_fn(
            examples=examples,
            preprocess_examples=True,
            params=hparams,
            is_training=False,
            shuffle_examples=False,
            skip_n_initial_records=0,
        )

        estimator = onf_train_util.create_estimator(config.model_fn, model_dir, hparams)

        iterator = tf.data.make_initializable_iterator(dataset)
        next_record = iterator.get_next()

        with tf.Session() as sess:
            sess.run(
                [tf.initializers.global_variables(), tf.initializers.local_variables()]
            )

            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                filename = f.name
                wavwrite(filename, hparams.sample_rate, audio)
                tf.logging.info("Starting transcription for %s...", filename)

                # The reason we bounce between two Dataset objects is so we can use
                # the data processing functionality in data.py without having to
                # construct all the Example protos in memory ahead of time or create
                # a temporary tfrecord file.
                tf.logging.info("Processing file...")
                sess.run(
                    iterator.initializer,
                    {examples: [create_example(filename, hparams.sample_rate, False)]},
                )

                def transcription_data(params):
                    del params
                    return tf.data.Dataset.from_tensors(sess.run(next_record))

                input_fn = onf_infer_util.labels_to_features_wrapper(transcription_data)

                tf.logging.info("Running inference...")
                checkpoint_path = None
                prediction_list = list(
                    estimator.predict(
                        input_fn,
                        checkpoint_path=checkpoint_path,
                        yield_single_examples=False,
                    )
                )
                assert len(prediction_list) == 1
                sequence_prediction = music_pb2.NoteSequence.FromString(
                    prediction_list[0]["sequence_predictions"][0]
                )
                with tempfile.NamedTemporaryFile(suffix=".midi") as g:
                    midi_filename = g.name
                    midi_io.sequence_proto_to_midi_file(
                        sequence_prediction, midi_filename
                    )
                    tf.logging.info("Transcription written to %s.", midi_filename)
                    with open(midi_filename, "rb") as g:
                        midi_bytes = g.read()

                return prediction_list[0], midi_bytes


# Alias for backwards compatibility
def onsets_and_frames(*args, **kwargs):
    return transcribe(*args, **kwargs)
