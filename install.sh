#!/usr/bin/env bash
# Install BC2 and everything it designs with: bash install.sh, then bindcraft design settings.json.
# The accelerator is read off the driver; pass cuda13, cuda12, rocm or oneapi to name it instead,
# and --no-weights to skip the one-time AlphaFold parameter download.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_MINIMUM=3.12
UV_VERSION=0.12.13

accelerator=""
fetch_weights=1
for argument in "$@"; do
  case "$argument" in
    --no-weights) fetch_weights=0 ;;
    cuda13|cuda12|rocm|oneapi) accelerator="$argument" ;;
    cpu) echo "bindcraft: there is no CPU installation. A trajectory folds an AlphaFold ensemble hundreds of times, which is days of processor for an hour of card, so install where there is a GPU." >&2; exit 2 ;;
    *) echo "usage: bash install.sh [cuda13|cuda12|rocm|oneapi] [--no-weights]" >&2; exit 2 ;;
  esac
done

if [[ -z $accelerator ]]; then
  # nvidia-smi reports the newest CUDA its driver serves, which is what the jax wheels are built for.
  # Its banner labels that "CUDA Version" on older drivers and "CUDA UMD Version" on newer ones, so
  # both spellings are matched: missing the label reads as no driver at all, and a machine holding a
  # working card is then told its GPU is presumed to belong to its job.
  # A machine with no driver at all has no nvidia-smi, and under pipefail that is a failed pipeline
  # rather than an empty answer, so every reading of it ends in a truth this script can act on.
  cuda_major=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA[A-Z ]*Version: *\([0-9][0-9]*\).*/\1/p' | head -1 || true)
  case "${cuda_major:-none}" in
    13|14) accelerator=cuda13 ;;
    12) accelerator=cuda12 ;;
    #No driver is the ordinary case on a login node, where the card belongs to the job rather than to
    #the machine doing the installing, so this installs for a GPU and says which one it assumed. What
    #it will not do is quietly fall back to the CPU wheels, which is an installation a hundred times
    #slower than the person running it expects with nothing afterwards to say so.
    *) accelerator=cuda13; echo "bindcraft: no NVIDIA driver here, so installing for CUDA 13 on the assumption the GPU belongs to your job; pass cuda12 instead for a V100 or older" ;;
  esac
  # CUDA 13 dropped every GPU below compute capability 7.5, and a driver serves 13 whatever card it
  # drives, so a Maxwell, Pascal or Volta card takes the CUDA 12 wheels however new its driver is.
  oldest_card=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -n | head -1 || true)
  if [[ $accelerator == cuda13 && -n $oldest_card ]] && awk "BEGIN{exit !($oldest_card < 7.5)}"; then
    echo "bindcraft: compute capability $oldest_card is below the 7.5 CUDA 13 needs, installing CUDA 12"
    accelerator=cuda12
  fi
fi

fetch_uv() {
  # The last resort for a machine whose Python is too old, or new enough but unable to build an
  # environment because the venv module ships without ensurepip, which is most login nodes. uv builds
  # one either way, bringing its own interpreter when it has to, and needs no administrator.
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) uv_target=x86_64-unknown-linux-gnu ;;
    Linux-aarch64) uv_target=aarch64-unknown-linux-gnu ;;
    Darwin-arm64) uv_target=aarch64-apple-darwin ;;
    Darwin-x86_64) uv_target=x86_64-apple-darwin ;;
    *) echo "bindcraft: no Python here can build an environment, and there is no uv build for $(uname -s)-$(uname -m) to build one with" >&2; exit 1 ;;
  esac
  local release="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$uv_target.tar.gz"
  local checksum=sha256sum
  command -v sha256sum > /dev/null || checksum="shasum -a 256"
  echo "bindcraft: no Python here can build an environment, fetching uv $UV_VERSION from github.com/astral-sh/uv to build one"
  mkdir -p .bindcraft-tools
  curl -fsSL "$release" -o ".bindcraft-tools/uv-$uv_target.tar.gz"
  curl -fsSL "$release.sha256" -o ".bindcraft-tools/uv-$uv_target.tar.gz.sha256"
  (cd .bindcraft-tools && $checksum -c "uv-$uv_target.tar.gz.sha256" > /dev/null)
  tar xzf ".bindcraft-tools/uv-$uv_target.tar.gz" -C .bindcraft-tools --strip-components=1 "uv-$uv_target/uv"
}

build_environment() {
  # uv first: it needs no working venv module and brings its own interpreter, so it answers on the
  # machines the other two cannot. Then the stock venv module, then conda, which is what a machine
  # with no Python 3.12 of any kind usually does have. Each is asked for the newest interpreter it can
  # reach at or above the minimum, so a machine that has 3.14 installs on 3.14. Whichever builds it,
  # the result is ./.venv
  # with a pip inside, so everything after this is the same three commands.
  local uv="$PWD/.bindcraft-tools/uv"
  #a bare && here would be a failed list under set -e on every machine that has no uv
  if command -v uv > /dev/null 2>&1; then uv=uv; fi
  if [[ -x $uv || $uv == uv ]] && rm -rf .venv && "$uv" venv --seed --python ">=$PYTHON_MINIMUM" .venv > /dev/null 2>&1; then
    builder="$("$uv" --version 2> /dev/null | head -1)"; return 0
  fi
  for candidate in python3.14 python3.13 python3.12 python3; do
    command -v "$candidate" > /dev/null 2>&1 || continue
    "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' 2> /dev/null || continue
    rm -rf .venv
    if "$candidate" -m venv .venv > /dev/null 2>&1; then builder="$candidate -m venv"; return 0; fi
  done
  for manager in micromamba mamba conda; do
    command -v "$manager" > /dev/null 2>&1 || continue
    rm -rf .venv .bindcraft-tools/python
    if "$manager" create --yes --quiet --prefix .bindcraft-tools/python "python>=$PYTHON_MINIMUM" pip > /dev/null 2>&1 \
       && .bindcraft-tools/python/bin/python -m venv .venv > /dev/null 2>&1; then
      builder="$manager"; return 0
    fi
  done
  rm -rf .venv
  fetch_uv || return 1
  .bindcraft-tools/uv venv --seed --python ">=$PYTHON_MINIMUM" .venv || return 1
  builder="uv $UV_VERSION, downloaded"
}

# Where the installation goes: the environment that is already active, one this script built here
# before, or a new one under ./.venv built by whichever of uv, venv and conda this machine has.
python="${BINDCRAFT_PYTHON:-}"
builder=""
if [[ -z $python && -n "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]]; then python=$(command -v python3 || true); fi
if [[ -z $python && -x .venv/bin/python ]]; then python="$PWD/.venv/bin/python"; fi
if [[ -z $python ]]; then
  echo "bindcraft: no environment active, building one in ./.venv"
  build_environment || { echo "bindcraft: could not build an environment here with uv, venv or conda" >&2; exit 1; }
  echo "bindcraft: environment built by $builder"
  python="$PWD/.venv/bin/python"
fi
"$python" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' || {
  echo "bindcraft: $("$python" -c 'import sys; print(sys.version.split()[0])') is active and BC2 needs $PYTHON_MINIMUM or newer; deactivate it and run this again to build an environment here" >&2
  exit 1
}

echo "bindcraft: installing for $accelerator"
"$python" -m pip install --quiet --upgrade pip
"$python" -m pip install -e ".[$accelerator]"

bindcraft="$(dirname "$python")/bindcraft"
if [[ $fetch_weights == 1 ]]; then "$bindcraft" fetch-weights; fi
"$bindcraft" --help > /dev/null

# pip, conda and uv all report success for an environment that is missing a module, and a campaign
# is where that otherwise surfaces, so every module and every checkpoint is named here instead.
if ! "$python" -m bindcraft.selfcheck "$accelerator"; then
  echo "bindcraft: the installation is incomplete, see above" >&2
  exit 1
fi
# Whether it also runs here is a different question: a login node with tight process limits cannot
# start a JAX runtime while its compute nodes are fine, so this reports rather than refuses.
if backend=$("$python" -c 'import jax; print(jax.default_backend(), len(jax.devices()))' 2>/dev/null); then
  echo "bindcraft: jax runs here on ${backend% *}, ${backend#* } device(s)"
  if [[ $accelerator == cuda* && ${backend% *} == cpu && -n $(nvidia-smi -L 2>/dev/null) ]]; then
    echo "bindcraft: this machine has a GPU and jax did not take it, so a campaign here would run on the CPU" >&2
  fi
else
  echo "bindcraft: jax did not start here, which is normal on a login node; check inside your allocation with"
  echo "  python -c 'import jax; print(jax.devices())'"
fi
echo
echo "bindcraft: installed, designing on $accelerator."
[[ -n "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]] || echo "  source .venv/bin/activate            # once per terminal"
echo "  bindcraft design examples/pdl1_denovo.json"
