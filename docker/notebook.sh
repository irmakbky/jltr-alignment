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
# Adds 1 to port (to differentiate from serve.sh)
PORT=$((PORT+1))
echo $PORT
mkdir -p notebook/
docker run \
	-it \
	--rm \
	$DOCKER_GPU_ARG \
	--name "${USER}_alignment_notebook" \
	-v $(pwd)/mus_align:/jltr-alignment/mus_align \
	-v $(pwd)/notebook:/jltr-alignment/notebook \
	-p $PORT:8888 \
	-t jltr-alignment-v2 \
	jupyter notebook \
		--ip=0.0.0.0 \
		--port 8888 \
		--no-browser \
		--allow-root \
		--notebook-dir=/jltr-alignment/notebook