# BC2 for any machine with an NVIDIA driver, x86-64 or aarch64. The jax cuda13 wheels carry their own
# CUDA and cuDNN, so the base image is plain Ubuntu and one file builds both architectures.
#
#   docker buildx build --platform linux/amd64,linux/arm64 -t bindcraft:1.0 -f containers/Dockerfile .
#   docker run --gpus all -v "$PWD:/work" bindcraft:1.0 design settings.json
#
# ProteinMPNN ships inside the package, so the image redesigns sequences as built. The AlphaFold
# parameters are 5.3 GB and are downloaded into the user's cache on the first campaign instead; build
# with --build-arg ALPHAFOLD_PARAMETERS=bake to put them in the image, which is what a compute node
# with no route to the internet needs.
FROM ubuntu:24.04

# biotraj, which biotite pulls in to read trajectories, publishes no aarch64 wheel, so aarch64
# compiles it here and the compiler stays in the image rather than being purged out from under
# whatever else links against it.
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev build-essential ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
ENV PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

# Editable, like install.sh: the settings/ and scaffolds/ trees live at the repository root rather
# than inside the package, and the runtime finds them relative to it, so the source tree has to stay
# in place and on the import path -- a plain install would copy only the package and orphan them.
COPY . /opt/bindcraft
RUN python3 -m pip install -e "/opt/bindcraft[cuda13]"

# Two things have to line up for a campaign in here to reach the cards, and jax says neither out
# loud: it warns once and runs on the CPU, a hundred times slower. The cuda13 wheels keep their
# libraries under nvidia/*/lib inside site-packages, where the loader does not look; and the NVIDIA
# container hook injects the host's driver only into an image that asks for it by these two
# variables, which a CUDA base image would have set and plain Ubuntu does not.
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN python3 -c "import nvidia, pathlib; print('\n'.join(sorted(str(path) for root in nvidia.__path__ for path in pathlib.Path(root).glob('*/lib'))))" > /etc/ld.so.conf.d/bindcraft-cuda.conf \
 && ldconfig && ldconfig -p | grep -q libcupti

ARG ALPHAFOLD_PARAMETERS=download-on-first-campaign
RUN if [ "$ALPHAFOLD_PARAMETERS" = bake ]; then \
      BINDCRAFT_WEIGHTS="$(python3 -c 'from bindcraft.model_weights import SHIPPED_WEIGHTS; print(SHIPPED_WEIGHTS)')" bindcraft fetch-weights; \
    fi

# The build fails here rather than on someone's first campaign if the wheels do not line up, or if
# the package was built without the checkpoints it is supposed to carry. Run from / so that what is
# tested is the installed package rather than the copy of the repository beside it.
WORKDIR /
RUN python3 -c "import jax, bindcraft.proteinmpnn; print('jax', jax.__version__)" \
 && bindcraft --help > /dev/null \
 && python3 -m bindcraft.selfcheck cuda13 --shipped-only

WORKDIR /work
ENTRYPOINT ["bindcraft"]
CMD ["--help"]
