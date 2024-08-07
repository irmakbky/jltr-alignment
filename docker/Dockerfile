FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# Configure shell
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
RUN apt-get update --fix-missing
RUN apt-get update --fix-missing

# Copy dataset
WORKDIR /jltr-alignment
COPY dataset/ dataset/
COPY smr_data/ smr_data/
ENV ALIGNMENT_APP_DATA_DIR=/jltr-alignment/dataset
ENV SMR_DATA_DIR=/jltr-alignment/smr_data

# Create cache and install model
WORKDIR /jltr-alignment
RUN mkdir -p cache
RUN apt-get update && apt-get install -y wget
RUN mkdir -p cache/simple-frontend
RUN wget https://github.com/OMR-Research/MeasureDetector/releases/download/v1.0/2019-04-24_faster-rcnn_inception-resnet-v2.pb -O cache/2019-04-24_faster-rcnn_inception-resnet-v2.pb
ENV ALIGNMENT_APP_CACHE_DIR=/jltr-alignment/cache
COPY mesa13_info/ $ALIGNMENT_APP_CACHE_DIR/mesa13_info/

# Install Python
ENV LANG C.UTF-8
RUN apt-get install -y python3 python3-dev python3-pip
RUN ln -s $(which python3) /usr/local/bin/python
RUN python -m pip install --no-cache-dir --upgrade pip
ENV NUMBA_CACHE_DIR /tmp/numba

# Install Python dependencies
# Target date: Nov 4 2021 (tensorflow-gpu 2.6.2)
RUN apt-get install -y poppler-utils
RUN python -m pip install --no-cache-dir pdf2image==1.17.0
RUN python -m pip install --no-cache-dir tensorflow-gpu==2.6.2
RUN python -m pip install --no-cache-dir --upgrade protobuf==3.19.4
RUN apt-get install -y ffmpeg
RUN python -m pip install --no-cache-dir Cython==0.29.24
RUN python -m pip install --no-cache-dir madmom==0.16.1
RUN python -m pip install --no-cache-dir Flask Flask-Cors
RUN python -m pip install --no-cache-dir --upgrade librosa==0.7.2
RUN python -m pip install --no-cache-dir --upgrade resampy==0.2.2
RUN apt-get update && apt-get install -y fluidsynth
RUN python -m pip install --no-cache-dir pyfluidsynth
RUN python -m pip install --no-cache-dir pretty_midi
RUN python -m pip install --no-cache-dir jupyter

RUN python -m pip install --no-cache-dir matplotlib==3.3.4
RUN python -m pip install --no-cache-dir opencv-python==4.10.0.84
RUN python -m pip install --no-cache-dir scikit-image==0.18.3
RUN python -m pip install --no-cache-dir scipy==1.4.1

# Install O&F dependencies
WORKDIR /model_ckpts/onf
RUN apt-get update --fix-missing
RUN apt-get install -y unzip
RUN \
    wget https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/maestro_checkpoint.zip \
    && unzip maestro_checkpoint.zip \
    && rm maestro_checkpoint.zip \
    && find .
RUN python -m pip install --no-cache-dir --no-deps magenta==2.1.4
RUN python -m pip install --no-cache-dir tensorflow_probability==0.14.1
RUN python -m pip install --no-cache-dir note_seq==0.0.3
RUN python -m pip install --no-cache-dir --upgrade pandas==1.3.4
RUN apt-get install -y sox
RUN python -m pip install --no-cache-dir --upgrade sox==1.4.0
RUN python -m pip install --no-cache-dir --upgrade tf-slim==1.1.0
RUN python -m pip install --no-cache-dir --upgrade mir_eval==0.6

# Force numpy/numba versions for compatibility with older packages
RUN python -m pip install --no-cache-dir --upgrade numpy==1.19.5
RUN python -m pip install --no-cache-dir --upgrade numba==0.48
ENV NUMBA_CACHE_DIR /tmp/numba

# Create library placeholder
WORKDIR /jltr-alignment
COPY setup.py setup.py
RUN mkdir -p mus_align
COPY mus_align/__init__.py mus_align/__init__.py
RUN mkdir -p mus_align/audio
RUN touch mus_align/audio/__init__.py
RUN mkdir -p mus_align/score
RUN touch mus_align/score/__init__.py
RUN python -m pip install --no-cache-dir --editable .

# Finalize
ENV DOCKER Yes
ENV FLASK_APP main
CMD python -m unittest discover -v
