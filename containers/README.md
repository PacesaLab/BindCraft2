# Containers

One `Dockerfile` for both architectures. The jax cuda13 wheels carry their own CUDA and cuDNN, so the
image is Ubuntu plus wheels and the only thing the host has to provide is an NVIDIA driver and the
container toolkit. ProteinMPNN is inside the package and therefore inside the image; the AlphaFold
parameters are not, unless the build is asked for them.

```bash
bash containers/build.sh bindcraft:1.0                   # x86-64 and aarch64
bash containers/build.sh bindcraft:1.0 linux/arm64       # aarch64 alone, on an aarch64 build node
docker run --gpus all -v "$PWD:/work" bindcraft:1.0 design settings.json
```

## A compute node with no route to the internet

The parameters have to be in the image, or on a filesystem the node mounts.

```bash
bash containers/build.sh bindcraft:1.0 --build-arg ALPHAFOLD_PARAMETERS=bake
```

The alternative is to run `bindcraft fetch-weights` once on a login node, which downloads them to
`~/.cache/bindcraft`, and mount that. `BINDCRAFT_AF2_PARAMS` still overrides everything, which is
what a cluster that already holds a shared copy should set.

## Apptainer and enroot

```bash
apptainer build bindcraft.sif docker-daemon://bindcraft:1.0
apptainer run --nv --bind "$PWD:/work" bindcraft.sif design settings.json
```

A site that runs enroot imports the image to squashfs and launches it through an environment
definition instead:

```bash
podman build -t bindcraft:1.0 -f containers/Dockerfile .
enroot import -o /path/to/bindcraft.sqsh podman://bindcraft:1.0
srun --environment=/path/to/bindcraft.toml bindcraft design settings.json
```

`bindcraft.toml` in this directory is that definition; set the image path and the mounts to the trees
holding the settings file and the output folder. Mount every tree a symlink on those paths points into
as well: a parameter directory that is a link into a filesystem the container does not mount exists to
`ls` and not to the campaign, and preflight refuses it as missing.

## Making sure the cards are reached

A campaign fans one design worker out per GPU by itself, but it reads `CUDA_VISIBLE_DEVICES` to find
them and the image carries no `nvidia-smi` to fall back on, so the allocation has to ask for the
cards: `--gpus-per-node=N`, or `--gres=gpu:N` wherever that is what the scheduler takes. Without it
Slurm sets the variable empty and jax finds no device.

Check that they are really in use before spending a night on a campaign. jax falls back to the CPU on
its own and says so only in a warning buried in a plugin traceback, and the fan-out line above it
still names every GPU, because that line is read off `CUDA_VISIBLE_DEVICES` rather than off the
devices jax opened:

```bash
srun --environment=/path/to/bindcraft.toml python3 -c "import jax; print(jax.devices())"
```

One `CudaDevice` per card is the answer. `[CpuDevice(id=0)]` means the driver never reached the
container, which is what happens when `NVIDIA_VISIBLE_DEVICES` and `NVIDIA_DRIVER_CAPABILITIES` are
missing from both the image and the environment definition: the hook injects the host driver only
into an image that asks for it by name.
