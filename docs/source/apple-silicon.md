# Apple Silicon

Apple Silicon support is experimental. Use a native ARM64 Python 3.12 or newer on macOS 14 or newer; Intel Macs and Python under Rosetta are unsupported. Metal uses a pinned runtime: JAX/jaxlib 0.11.2 and metaljax 0.11.9, with psutil 7.x for memory controls.

## Install

Start without an active Python or Conda environment:

```bash
git clone https://github.com/PacesaLab/BindCraft2.git
cd BindCraft2
bash install.sh metal
source .venv/bin/activate
```

Automatic installation also selects Metal on native Apple Silicon. The installer creates or reuses `.venv`, or installs into an already active compatible environment. A manually managed environment can use `python -m pip install -e '.[metal]'`.

ProteinMPNN checkpoints ship with BC2; AlphaFold parameters download during installation. To defer that download and the Metal execution check, use `bash install.sh metal --no-weights --no-backend-check`, then run `bindcraft fetch-weights` before a campaign. To reuse existing AlphaFold parameters, set `BINDCRAFT_AF2_PARAMS` to their directory before installation and in subsequent shells; it must contain `params_<model>.npz` directly or under `params/`. See [weights and caches](installation.md#model-weights-and-caches).

## Check the installation

Check dependencies and shipped checkpoints without initializing JAX:

```bash
python -m bindcraft.selfcheck metal --shipped-only
bindcraft --help
```

When ready to execute a small Metal calculation, verify the backend:

```bash
JAX_PLATFORMS=metal python -m bindcraft.selfcheck metal --shipped-only --check-backend
```

Omit `--shipped-only` to require AlphaFold parameters too. A dependency check alone does not verify Metal execution; the backend check fails on unavailable Metal or CPU fallback. Set backend and memory environment variables before Python starts. An explicit backend choice is respected; `JAX_PLATFORMS=cpu` is suitable for small CPU tests.

## Run and resume

Use the ordinary CLI and campaign JSON. Bound the first run, since the shipped example otherwise requests ten accepted designs without an attempt limit:

```bash
bindcraft design examples/pdl1.json --set 'max_trajectories=4' --set 'project_folder=results/pdl1_apple'
```

This allows four attempts and may produce no accepted designs. Repeat the same command to resume; raise `max_trajectories` to allow more attempts. Results, claims and seeds follow ordinary [campaign bookkeeping](installation.md#run-a-campaign).

## Workers and reuse

Metal uses one device and a shared unified-memory budget. Keep one campaign pool active on the Mac: separate project folders do not coordinate machine headroom. Existing controls apply; environment overrides take precedence:

| Setting | Environment override | Metal behavior |
| --- | --- | --- |
| `workers_per_gpu` | `BINDCRAFT_WORKERS_PER_GPU` | `auto` estimates capacity; an integer requests a count. Memory can reduce it. |
| `max_workers_per_gpu` | `BINDCRAFT_MAX_WORKERS_PER_GPU` | Cap workers on the device. |
| `design_workers` | `BINDCRAFT_DESIGN_WORKERS` | Cap total workers. |
| `auto_multi_gpu` | — | `false` requests one supervised worker, which can still restart. |
| `worker_launch_stagger` | `BINDCRAFT_WORKER_LAUNCH_STAGGER` | Space worker starts to reduce startup pressure. |
| `gpu_ids` | `BINDCRAFT_GPU_IDS` | Leave unset or use `0`. |

Each worker reuses its own models and JAX compilations across designs, using the existing padded length buckets. The current supported Metal runtime cannot share compiled work between workers or preserve it across process restarts. Setting `JAX_COMPILATION_CACHE_DIR` does not enable either form of Metal reuse. New workers pay compilation costs again.

When desperation changes compilation settings, workers stop taking new designs and finish active designs, including ProteinMPNN and validation, with their original settings. All old workers exit before replacements start, releasing memory retained by the old compilations. The campaign then rereads progress: an acceptance can reset the ladder or finish the campaign. Otherwise it starts the current settings with a count that fits memory; heavier settings can reduce that count. This also applies to a single worker. Planned restarts spend no extra attempt; a crashed attempt remains spent. The scientific [desperation ladder](design-guide/05-desperation-and-autotuning.md), thresholds and filters are unchanged.

## Memory headroom

`BINDCRAFT_METAL_RESERVE_GIB` sets headroom for macOS and other applications. The default is **16 GiB on machines with at least 64 GiB physical RAM, otherwise 8 GiB**; values must be finite and nonnegative. For example:

```bash
BINDCRAFT_METAL_RESERVE_GIB=24 bindcraft design examples/pdl1.json --set 'max_trajectories=4'
```

BC2 takes a conservative available-memory snapshot before backend initialization, subtracts the reserve and coordinator overhead, then divides the campaign budget among workers. Replacement pools keep the original ceiling; the native memory governor also responds to current pressure. Stricter explicit metaljax limits remain in effect. The reserve is a best-effort target, not an exclusive macOS reservation; other applications and transient allocations can still create pressure.

Worker estimates are provisional. An unsupported estimate, unknown capacity or insufficient room for even one worker refuses startup; requesting one worker does not bypass the check. Free memory, reduce the workload or use a larger machine. CUDA size tables do not estimate Apple requirements. For resource diagnostics, use process physical footprint together with system pressure, compression and swap; RSS alone can understate unified-memory use.

## Supported configurations and limitations

- The managed Metal sizing envelope covers one structural target and one binder chain, up to 512 padded target-plus-binder residues and 16 assigned length buckets, rematerialized `auto`/`chunked` attention with automatic subbatching, at most three design/validation/beta-sheet recycles and ordinary design losses. Scaffolds, additional states, cyclization and accepted-design relaxation are outside this envelope. Parameter sweeps use the legacy worker path without coordinated restarts; they have not established the managed path's memory guarantees.
- Automatic attention uses `chunked`. Explicit `chunked` and `stock` attention are available, but `stock` is outside the managed sizing envelope. Fused `cudnn`/`xla` attention, cuEquivariance and an active optional `collective_softness` loss are unsupported. Background `compile_next_length` is disabled on Metal.
- Memory estimates remain provisional. Larger worker pools, the full range of supported shapes, long campaigns and throughput scaling have not been validated.
- Native BF16 numerical agreement and equivalent cross-backend design success rates have not been established. Computational acceptance still requires experimental validation.

Model precision, objectives, optimizer, sampling, recycles and filters retain the requested scientific settings. Linux CUDA/ROCm behavior follows the ordinary [installation and execution guide](installation.md).
