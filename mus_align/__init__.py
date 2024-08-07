import os
import pathlib

import tensorflow.compat.v1 as tf

LIB_DIR = pathlib.Path(__file__).resolve().parent

if "ALIGNMENT_APP_CACHE_DIR" in os.environ:
    CACHE_DIR = pathlib.Path(os.environ["ALIGNMENT_APP_CACHE_DIR"])
else:
    CACHE_DIR = pathlib.Path(pathlib.Path.home(), "jltr-alignment")
CACHE_DIR = CACHE_DIR.resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if "ALIGNMENT_APP_DATA_DIR" in os.environ:
    ALIGNMENT_APP_DATA_DIR = pathlib.Path(os.environ["ALIGNMENT_APP_DATA_DIR"])
    ALIGNMENT_APP_DATA_DIR = ALIGNMENT_APP_DATA_DIR.resolve()
else:
    ALIGNMENT_APP_DATA_DIR = None

TEST_DATA_DIR = LIB_DIR / "test_data"

if "SMR_DATA_DIR" in os.environ:
    SMR_DATA_DIR = pathlib.Path(os.environ["SMR_DATA_DIR"])
else:
    SMR_DATA_DIR = pathlib.Path(pathlib.Path.home(), "jltr-alignment/smr_data")

# Configure TensorFlow to only use GPU 0.
# NOTE: This seems to mysteriously change the raw output of Onsets & Frames
# *very* slightly.
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_visible_devices(gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)


# NOTE: This changes the test discovery pattern from "test*.py" (default) to "*test.py".
def load_tests(loader, standard_tests, pattern):
    package_tests = loader.discover(start_dir=LIB_DIR, pattern="*test.py")
    standard_tests.addTests(package_tests)
    return standard_tests
