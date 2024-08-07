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
docker run \
	-it \
	--rm \
	$DOCKER_GPU_ARG \
	-v $(pwd)/mus_align:/jltr-alignment/mus_align \
	-t jltr-alignment-v2 \
	$@