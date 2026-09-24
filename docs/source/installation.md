# BindCraft2 installation and running

First design (see "Run your first design" in your BindCraft2 repo's top-level README.md) · [Reference Documentation](reference.md) · [Outputs and Measurements](outputs.md) · [Examples](examples.md)

[Install](#install) · [Check it](#check-the-installation) · [Weights and caches](#model-weights-and-caches) · [Run a campaign](#run-a-campaign) · [GPU and memory](#gpu-and-memory-controls) · [Clusters](#slurm-and-other-schedulers) · [Containers](#containers) · [No internet](#machines-with-no-route-to-the-internet) · [Troubleshooting](#troubleshooting)

BindCraft2 needs Linux, Python 3.12 or newer and a GPU. One command installs it on a workstation, a login node or inside a container, and chooses the accelerator wheels for the card it finds. **There is no CPU installation**: a single trajectory folds an AlphaFold ensemble hundreds of times, which is days of processor for an hour of card, so `bash install.sh cpu` is refused rather than quietly built.

## Install

```bash
git clone https://github.com/PacesaLab/BindCraft2.git
cd BindCraft2
git clone https://github.com/PacesaLab/BindCraft2.git
cd BindCraft2
bash install.sh
source .venv/bin/activate
```

`install.sh` builds `.venv` beside the repository when no environment is active, installs BindCraft2 and its accelerator wheels, downloads the AlphaFold parameters and then verifies the result. **The install is editable, so the clone is the installation**: moving, renaming or deleting the BindCraft2 directory breaks the `bindcraft` command. Clone it somewhere permanent, not into a scratch directory that gets purged. Re-running `bash install.sh` is safe, reuses an existing `.venv`, does not download the parameters again, and is how you switch accelerator. Activate `.venv` once per terminal. Allow about 20 GB of free space: the 5.3 GB parameter archive and its unpacked contents sit side by side during the transfer before the archive is deleted, the accelerator wheels are several GB more, and results accumulate beside them.

| Your machine | What to run |
| --- | --- |
| GPU workstation | `bash install.sh` |
| Cluster login node with no visible GPU | `bash install.sh`, which assumes CUDA 13 for the card your job will get |
| Compute nodes with V100 or older cards | `bash install.sh cuda12` from the login node |
| AMD | `bash install.sh rocm` |
| A Conda or virtual environment you want to keep using | activate it first, then `bash install.sh` |
| No route to the internet from the compute nodes | `bash install.sh`, then see [machines with no route to the internet](#machines-with-no-route-to-the-internet) |
| No usable Python at all | `bash install.sh`, which fetches `uv` and brings its own interpreter |

### What the installer chooses

| Argument | Effect |
| --- | --- |
| none | Read the accelerator off the driver. |
| `cuda13` | CUDA 13 wheels; needs compute capability 7.5 or newer. |
| `cuda12` | CUDA 12 wheels; the choice for Volta, Pascal and Maxwell cards. |
| `rocm` | AMD accelerators, against a local ROCm 7 installation. |
| `--no-weights` | Skip the one-time AlphaFold download; run `bindcraft fetch-weights` later. |

Automatic selection reads the CUDA version `nvidia-smi` reports, then the oldest visible card's compute capability. A driver serving CUDA 13 or 14 selects `cuda13`, a driver serving 12 selects `cuda12`, and a card below compute capability 7.5 moves the choice back to `cuda12` however new the driver is. **A machine with no driver selects `cuda13` and says so**, which is the ordinary case on a login node where the card belongs to the job rather than to the machine doing the installing. It never falls back to CPU wheels. A driver older than CUDA 12 reads the same way and prints the same line, because only 12, 13 and 14 are matched, so name `cuda12` yourself on such a machine.

Match the choice to the **compute** nodes, not the login node. If they differ, name the accelerator explicitly.

### Installing into an environment you already have

An active virtual or Conda environment on Python 3.12 or newer is installed into directly, and `.venv` is not created.

```bash
# Activate conda environment substitute env-name to your actual environment name
conda activate your-env-name
# OR activate your virtual environment
source /path/to/your-environment/bin/activate
# Install bindcraft
bash install.sh
```

Activate that same environment in later sessions. An active environment older than 3.12 stops the installer with the version it found; deactivate it and run again to get `.venv`. `BINDCRAFT_PYTHON` names an interpreter to install into instead of either.

### When the machine has no usable Python

The installer builds `.venv` with the first of these that works, and reports which one it used: `uv`, the stock `venv` module, then `micromamba`, `mamba` or `conda`, and finally a copy of `uv` downloaded from GitHub with its checksum verified. Each is asked for the newest interpreter it can reach at 3.12 or above, so a machine with 3.14 installs on 3.14. This matters on login nodes, where `venv` often ships without `ensurepip` and can build nothing; `uv` brings its own interpreter and needs no administrator.

## Check the installation

The installer ends with three checks, and prints what each found.

1. **Every module and checkpoint is named.** `python -m bindcraft.selfcheck <accelerator>` reports any missing dependency or checkpoint. A `pip`, `conda` or `uv` install can resolve a wheel and unpack nothing while reporting success, and a campaign is otherwise where that surfaces.
2. **The command answers.** `bindcraft --help`.
3. **JAX is probed for devices.** A login node with tight process limits often cannot start a JAX runtime while its compute nodes are fine, so this reports rather than refuses.

Run this inside a GPU allocation, or on the workstation, to confirm the cards are reached:

```bash
python -c "import jax; print(jax.devices()); assert jax.default_backend() == 'gpu', 'no GPU available to JAX'"
```

One `CudaDevice` per card is the answer. `[CpuDevice(id=0)]` means a campaign here would run on the CPU. The installer flags this case itself when the machine has a GPU that JAX did not take.

```bash
python -m bindcraft.selfcheck cuda13              # after changing an environment by hand
python -m bindcraft.selfcheck cuda13 --shipped-only   # ask only about the checkpoints the package carries
```

## Model weights and caches

| Checkpoints | Size | Where they come from |
| --- | --- | --- |
| ProteinMPNN, all three variants | 6.6 MB each | Inside the package. Nothing to download. |
| AlphaFold parameters | 5.3 GB once | Downloaded to `BINDCRAFT_AF2_PARAMS` (which defaults to `~/.cache/bindcraft/alphafold`) on first use. A campaign needs seven of the models the archive holds. |

A campaign designs against five multimer models and holds two monomer models back to validate on, so a design is never scored by a model that shaped it. `bindcraft fetch-weights` puts both sets on the machine now and prints where they are; run it on a node whose network reaches the internet. A transfer that stopped leaves a file that exists and holds nothing, so presence alone is not enough: checkpoints under 1 MB (ProteinMPNN) or 100 MB (AlphaFold) are reported as unfinished rather than accepted.

### Compiled-graph caches

A prediction shape costs about 60 s to compile and 1.5 s to read back, so a campaign keeps its compiled graphs between runs under `~/.cache/bindcraft/compile_cache/<card>`, filed by GPU type because an executable is not portable across cards. It prints `compiled graphs cached in <path>` when it does.

That per-card cache needs `nvidia-smi` to name the card. **Where `nvidia-smi` cannot be reached, including inside the shipped container image, which does not carry it, graphs go to `${TMPDIR:-/tmp}/bindcraft_xla_cache` instead** and no caching line is printed, so every campaign pays its compiles again. Set `JAX_COMPILATION_CACHE_DIR` to keep them somewhere durable, which is also worth doing on a cluster whose nodes all carry the same card. Where the home cache exists but cannot be written, the graphs go under the campaign's own output folder.

These are working files. Deleting them costs compile time and no results.

### Pointing BindCraft2 at checkpoints you already have

| Variable | What it names |
| --- | --- |
| `BINDCRAFT_WEIGHTS` | The cache directory itself, in place of `~/.cache/bindcraft`. Set it before installing and in every job to share one copy across a group. |
| `BINDCRAFT_AF2_PARAMS` | A directory already holding `params_<model>.npz`. This overrides everything, including the cache, and is what a site with a shared AlphaFold copy should set. |
| `BINDCRAFT_MPNN_WEIGHTS` | The `weights_neutral` directory of a ProteinMPNN checkpoint set. `weights_negative` and `weights_positive` are read beside it, so all three variants move together and a campaign is refused if the one it asks for is missing. Unset it to use the shipped checkpoints. |
| `XDG_CACHE_HOME` | Moves `~/.cache` itself, and with it both the weight cache and the compiled-graph cache. |

`BINDCRAFT_AF2_PARAMS` wins over `BINDCRAFT_WEIGHTS`, which wins over `XDG_CACHE_HOME`, which wins over `~/.cache`. Parameters are accepted as `params/params_<model>.npz` or `params_<model>.npz` directly inside the named directory. Point at a directory, never at a single file.


## Run a campaign

Run inside a GPU allocation, from the repository root, with the environment active:

```bash
bindcraft design examples/pdl1.json
```

| Command | What it does |
| --- | --- |
| `bindcraft design <settings.json>` | Run a campaign. A bare settings filename also works: `bindcraft examples/pdl1.json`. |
| `bindcraft rank <folder>` | Re-rank results on another measurement. |
| `bindcraft filter <folder>` | Re-apply acceptance thresholds to existing candidates. |
| `bindcraft score <structure>` | Measure a complex from any source. |
| `bindcraft campaign_output <folder>` | Rebuild summaries and rankings. |
| `bindcraft archive` / `unarchive <folder>` | Zip or restore completed trajectory folders. |
| `bindcraft fetch-weights` | Download and verify the checkpoints. |
| `bindcraft design --list-targets`, `--list-modalities`, `--list-properties`, `--list-core` | Print the shipped preset names, one per line. |
| `bindcraft design --list-settings` | Print every setting name `--set` accepts. The same names with their defaults are in `settings/core/reference.json` in your BindCraft2 repo. |

`bindcraft design -h` names every shipped target, binder format and design property with a line of description each, and lists the settings changed most often with worked `--set` examples. `archive`, `unarchive` and `fetch-weights` each answer to `-h` with their own usage line and what they are for. See [settings](reference.md) for what each one changes and [outputs](outputs.md) for reading the results.

`design`, `score`, `rank` and `filter` build a JAX runtime; `--help`, the preset listings and `fetch-weights` deliberately do not, so they answer on a login node that cannot start one.

Exit status 0 is a finished campaign, 2 a refusal printed as `campaign refused:` with every reason at once, and any other non-zero status comes from the first design worker that failed.

### Alternative ways to run a campaign

| Starting point | Command |
| --- | --- |
| A clone with nothing activated | `python3 bindcraft.py examples/pdl1.json` |
| Inside Python or a notebook | `from bindcraft import launch_campaign; launch_campaign("examples/pdl1.json")` |
| A module, when the console script is not on `PATH` | `python -m bindcraft.cli design examples/pdl1.json` |


## GPU and memory controls

A campaign uses **every GPU it can see, without being asked**, and packs several design workers onto each card. Nothing needs configuring for this; the settings to override this behaviour are indicated below.

### How a campaign fills a card

Visible cards are read from `CUDA_VISIBLE_DEVICES`, and from `nvidia-smi` when that is unset. A whole-cycle campaign runs up to **seven workers per card**, as many as its free memory holds. On a GH200 a forty-trajectory campaign took 3620 s at one worker and 2021 s at seven.

If `trajectory_only` is set in a campaign, or a card whose memory cannot be read, BindCraft2 runs **one worker per card**.

Each worker is budgeted at `2.0 × (3.4 GB + 38 kB × N²)` for a padded complex of N residues, with 4 GB of the card left as headroom, and each is given its own share of card memory. The node's own memory caps the plan too, at 4 GB per worker, since every worker holds its own copy of the parameters. Workers cannot share a trajectory, so **`max_trajectories` must be greater than the number of workers**.

Where a campaign has several binder lengths, each worker takes a share of them so it compiles fewer shapes. A length group is then drawn as often as its worker finishes a trajectory rather than as often as its share of the range, so **faster length groups can appear more often in results**.

### Override GPU workers default

The first five carry an environment variable that takes precedence over the setting, for changing one run without editing its file. The rest are settings only.

| Setting | Environment variable | Default | Why change it |
| --- | --- | --- | --- |
| `workers_per_gpu` | `BINDCRAFT_WORKERS_PER_GPU` | `auto` | Fix the workers per card instead of packing to memory. |
| `max_workers_per_gpu` | `BINDCRAFT_MAX_WORKERS_PER_GPU` | 8 | Cap automatic packing. |
| `design_workers` | `BINDCRAFT_DESIGN_WORKERS` | Unset | Cap total workers across the allocation. |
| `gpu_ids` | `BINDCRAFT_GPU_IDS` | All visible | Use a subset of the visible cards. |
| `worker_launch_stagger` | `BINDCRAFT_WORKER_LAUNCH_STAGGER` | 0 seconds | Rarely needed. A shape is compiled once per card under a lock the other workers wait on, so packed workers launch together. When set, the wait falls between the launches sharing a card: one worker on every card, then the next on any of them. |
| `auto_multi_gpu` | — | true | False keeps one process on one card. |
| `subbatch_size` | — | `auto` | An integer splits large calculations to save memory; `null` disables chunking. |
| `length_bucket_size` | — | 32 | Pad lengths to reuse compiled shapes; 1 disables padding. |
| `compile_next_length` | — | true | Prepare the next length while the current trajectory runs. |
| `attention_backend` | — | `auto` | Choose the attention implementation; options: 'auto', 'stock', 'cudnn' |
| `use_cueq` | — | false | Enable cuEquivariance kernels, which the CUDA extras install. |

`BINDCRAFT_WORKER_ID`, `BINDCRAFT_WORKER_COUNT` and `BINDCRAFT_BINDER_LENGTHS` are set **for** each worker by the campaign. **Do not set them; a process that carries `BINDCRAFT_WORKER_ID` believes it is a worker and will not fan out.**

### When a campaign runs out of GPU memory

Trouble shoot working down this list; each step costs less than the one below it.

1. Lower `workers_per_gpu`, or set it to 1. Packing is the largest single consumer.
2. Set `subbatch_size` to an integer to chunk the largest calculations.
3. Reduce `binder_lengths`, or pin one length. Memory grows with the square of the padded complex.
4. Raise `length_bucket_size` so fewer distinct shapes are compiled and held.

Memory is sized from the padded complex, so a large target costs the same on every trajectory. Per worker that formula gives 7.1 GB at 64 residues, 8.0 at 128, 11.8 at 256, 18.0 at 384 and 26.7 at 512, which is what to divide a card by.

**The host-memory cap reads the node's available memory, not your job's memory limit**, so on a shared node it can be generous. `XLA_PYTHON_CLIENT_PREALLOCATE` is already false, which is what lets each packed worker grow into its share rather than reserve it up front, and `XLA_FLAGS` already carries the setting BindCraft2 needs. Leave both alone unless you are diagnosing JAX itself.

**Unset an override rather than clearing it.** An exported but empty variable counts as a value, so `BINDCRAFT_WORKERS_PER_GPU=` fails on an empty string instead of falling back to the setting.

### What workers write

`workers/campaign_settings.json` holds the resolved settings the workers were launched on, and `workers/worker_<NN>_gpu_<id>.log` each worker's complete record, written line by line. The shared console prints one whole trajectory at a time and in trajectory order, holding a finished trajectory until no worker is still designing a lower-numbered one, so a packed run still reads down the log as the campaign numbered it. Read a single worker's log when one of them fails.

## Slurm and other schedulers

We provide a default slurm script, though different slurm clusters may have different flavors and quirks.
In that case you can use our script as an example to adapt to your cluster specifications. 

```bash
bash bindcraft.slurm examples/pdl1.json
```

Run that way the script reads the largest GPU count any node of the cluster advertises, sizes cores and memory to it at 4 cores and 24 GB per GPU, and submits itself. The campaign then designs on every GPU of the allocation at once. `BINDCRAFT_GPUS=2` asks for a particular number instead.

```bash
sbatch bindcraft.slurm examples/pdl1.json
sbatch --gres=gpu:4 --time=48:00:00 bindcraft.slurm examples/pdl1.json --modality VHH --humanize
```

Submitted with `sbatch` it takes the allocation it is given. Its declared defaults are one GPU, 8 cores, 48 GB and 24 hours, and any `sbatch` flag overrides them. Design options follow the settings filename and are passed straight through.

### Adapt to your cluster

| What your site needs | How to supply it |
| --- | --- |
| Account and partition | `sbatch --account=... --partition=...`, or fill in the commented `#SBATCH` lines in `bindcraft.slurm`. The script cannot guess these. |
| Submitting from outside the repository | Set `BINDCRAFT_HOME` to the repository root. Slurm runs a copy of the script, and the copy cannot find the repository on its own. |
| An environment that is not `.venv` | The job activates `.venv` when it finds one. Otherwise add your `module load` or `conda activate` lines to the script. |

The job prints its host, job id, the cards it can see and its working directory before starting, and writes `bindcraft_<jobid>.out` in the submission directory. Ask for the cards explicitly: **without a GPU request Slurm sets `CUDA_VISIBLE_DEVICES` empty and JAX finds no device**.

The repository ships a Slurm script only. On another scheduler, or on a bare GPU workstation, run `bindcraft design settings.json` inside the allocation or session; the fan-out across visible cards is automatic either way. Request at least one GPU, about 4 cores and 24 GB of host memory per card.

A campaign resumes by default: run it again against the same folder, repeating its original flags, and it carries on. Counters come back from `.campaign_state.json`, or are recovered by counting result rows if that file is gone. Set `"resume": false` to refuse a non-empty folder instead.

## Containers

One `Dockerfile` builds both x86-64 and aarch64. The CUDA 13 wheels carry their own CUDA and cuDNN, so the image is Ubuntu plus wheels and the host need only provide an NVIDIA driver and the container toolkit. ProteinMPNN is inside the image; the AlphaFold parameters are not unless the build is asked for them.

```bash
bash containers/build.sh bindcraft:1.0                    # both architectures
bash containers/build.sh bindcraft:1.0 linux/arm64        # aarch64 alone, on an aarch64 build node
docker run --gpus all -v "$PWD:/work" bindcraft:1.0 design settings.json
```

The entry point is `bindcraft` itself and the working directory is `/work`, so arguments start at the subcommand. Mount the trees holding your settings file and your output folder.

```bash
apptainer build bindcraft.sif docker-daemon://bindcraft:1.0
apptainer run --nv --bind "$PWD:/work" bindcraft.sif design settings.json
```

A site running enroot imports the image to squashfs and launches it through an environment definition; `containers/bindcraft.toml` is that file, and needs the image path and your mounts.

```bash
podman build -t bindcraft:1.0 -f containers/Dockerfile .
enroot import -o /path/to/bindcraft.sqsh podman://bindcraft:1.0
srun --environment=/path/to/bindcraft.toml bindcraft design settings.json
```

**Mount to the container every tree a symlink on your paths points into.** A parameter directory that is a link into a filesystem the container does not mount exists to `ls` and not to the campaign, and preflight refuses it as missing.

### Reaching the cards from a container

Two things have to line up, and JAX says neither out loud: it warns once and then runs on the CPU.

1. The allocation has to ask for the cards, with `--gpus all`, `--gpus-per-node=N` or `--gres=gpu:N`. A campaign reads `CUDA_VISIBLE_DEVICES` to find them and the image carries no `nvidia-smi` to fall back on.
2. `NVIDIA_VISIBLE_DEVICES` and `NVIDIA_DRIVER_CAPABILITIES` have to be set. The container hook injects the host driver only into an image that asks for it by name. The image declares both, but an environment definition that replaces the environment must carry them too, which is why `containers/bindcraft.toml` repeats them.

Check before spending a night on a campaign. The fan-out line a campaign prints is read off `CUDA_VISIBLE_DEVICES`, not off the devices JAX opened, so it names every card either way:

```bash
srun --environment=/path/to/bindcraft.toml python3 -c "import jax; print(jax.devices())"
```

See `containers/README.md` in your BindCraft2 repo for the build recipes in full.

## Machines with no route to the internet

The AlphaFold parameters are the only thing BindCraft2 downloads. Choose one of these.

| Approach | How |
| --- | --- |
| Fetch once, share the cache | Run `bindcraft fetch-weights` on a login node, then point every job at the result with `BINDCRAFT_WEIGHTS`, or mount `~/.cache/bindcraft`. |
| Use a copy your site already holds | Set `BINDCRAFT_AF2_PARAMS` to the directory holding `params_<model>.npz`. |
| Bake them into the image | `bash containers/build.sh bindcraft:1.0 --build-arg ALPHAFOLD_PARAMETERS=bake` |

Install with `bash install.sh --no-weights` when the installing machine should not download them either. **On a machine holding no parameters yet, that installation ends by reporting `bindcraft: the installation is incomplete` and exiting 1.** The package installed correctly; the closing check asks for all seven AlphaFold checkpoints whether or not the download was skipped. Confirm the package half on its own:

```bash
python -m bindcraft.selfcheck cuda13 --shipped-only    # silence means the package is sound
```

Verify the parameters before submitting, since a campaign otherwise meets a missing AlphaFold checkpoint when it builds the model and a missing ProteinMPNN one only when redesign begins, an hour of GPU time later:

```bash
bindcraft fetch-weights          # prints both resolved directories, or names what is missing
```

The [trajectory viewer](outputs.md#trajectory-records-and-viewers) loads its 3Dmol.js dependency from the internet, so open those HTML files on a connected machine. Structures, tables and PNG plots need nothing.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| A campaign runs, far slower than expected | JAX took the CPU. It warns once, inside a plugin traceback. | Run the device check above. In a container, see [reaching the cards](#reaching-the-cards-from-a-container); otherwise check the allocation asked for a GPU and that the accelerator matches the card. |
| `jax did not start here` during installation | Login-node process limits, which do not apply on compute nodes. | Normal. Repeat the device check inside an allocation. |
| `[CpuDevice(id=0)]` inside an allocation | No GPU in the allocation, a driver the wheels do not match, or a container with no driver injected. | `nvidia-smi` to confirm the card, then reinstall naming the accelerator, or set the two `NVIDIA_*` variables. |
| Works on the login node, fails on compute nodes | The installer read the login node's driver, and the compute cards are older. | `bash install.sh cuda12`. |
| `AlphaFold parameters are not on this machine` | Nothing downloaded, or `BINDCRAFT_AF2_PARAMS` points somewhere without them. | `bindcraft fetch-weights`, or correct the variable. |
| `checkpoints missing or unfinished` | An interrupted download, or an unpack that lost a model. | Delete `~/.cache/bindcraft/alphafold` and run `bindcraft fetch-weights` again. |
| `ProteinMPNN weights are not configured` | `BINDCRAFT_MPNN_WEIGHTS` names a directory this installation does not hold. | Unset it; the checkpoints ship inside the package. |
| `module not installed: <name>` | An install that resolved a wheel and unpacked nothing. | `bash install.sh` again in the same environment. |
| `bindcraft: this environment has no ...` | The command is running under an interpreter that is not the installed one. | `source .venv/bin/activate`, or use `python3 bindcraft.py`. |
| `... is active and BindCraft2 needs 3.12 or newer` | An older Python environment is active. | Deactivate it and run `bash install.sh` to build `.venv`. |
| A path exists to `ls` and the campaign calls it missing | A symlink into a filesystem the container does not mount. | Mount the tree the link points into. |
| Out of GPU memory | Too many packed workers for the complex size. | [When a campaign runs out of GPU memory](#when-a-campaign-runs-out-of-gpu-memory). |
| `already holds campaign output` | `"resume": false` was set and that folder is not empty. | Drop `resume: false` to continue it (resuming is the default), or use a fresh `project_folder`. |
| `output folder ... cannot be written` | No write permission on the nearest existing parent. | Choose a writable location. |

### What a campaign refuses before it starts

`preflight` checks the machine and the request together and prints **every** reason at once under `campaign refused:`, so one run tells you everything to change. It covers the checkpoints the campaign will load, the target structures and scaffold parsing, the output folder, and what each modality asks of the rest of the settings. It then prints one `campaign preflight:` line naming the design models, validation models, redesign checkpoint, targets and output folder. Check that line against what you intended.

Settings are cleaned before use, so stray whitespace in a name, path or residue span is not fatal. Everything else is a refusal with the correction in it. See [settings](reference.md) for the values each check expects.

### Reporting a problem

`campaign_metadata.json` records the resolved settings, the model choices, the source revision (`-dirty` when the tree was modified) and SHA-256 hashes of every checkpoint loaded. Include it, the `campaign refused:` or worker log text, and the output of `python -m bindcraft.selfcheck <accelerator>`.
