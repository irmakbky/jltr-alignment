set -e

docker build \
	--progress=plain \
	-f docker/Dockerfile \
	-t jltr-alignment-v2 \
	.

GPU_ID=0
DOCKER_GPUS=$(nvidia-smi -L | python3 -c "import sys; print(','.join([l.strip().split()[-1][:-1] for l in list(sys.stdin)][$GPU_ID:$GPU_ID+1]))")
echo $DOCKER_GPUS
DOCKER_GPU_ARG="--gpus device=${DOCKER_GPUS}"
# Maps $USER to random (but consistent) port between 2000-65000
PORT=$(( ( $(echo -n "$USER" | od -An -tu4 | tr -d ' ' | head -n1) % 63001 ) + 2000 ))
echo $PORT
docker run \
	-it \
	--rm \
	$DOCKER_GPU_ARG \
	--name "${USER}_alignment" \
	-v $(pwd)/mus_align:/jltr-alignment/mus_align \
	-v $(pwd)/score-frontend:/jltr-alignment/cache/frontend \
	-p $PORT:5000 \
	-t jltr-alignment-v2 \
    python -m mus_align.serve