#!/bin/bash
#SBATCH --job-name=bindcraft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=bindcraft_%j.out
#
# Submit a campaign to as many GPUs as one node of this cluster carries:
#
#     bash bindcraft.slurm examples/pdl1_denovo.json
#     bash bindcraft.slurm examples/pdl1_denovo.json number_of_final_designs=20
#
# Run that way it reads the GPU count off sinfo, sizes the cores and the memory to it and submits
# itself. A campaign then designs on every GPU of the allocation at once, one worker per GPU, with
# nothing to configure. BINDCRAFT_GPUS=2 asks for a particular number instead.
#
# Submitted with sbatch it takes the allocation it is given, and any sbatch flag overrides the
# defaults declared above:
#
#     sbatch --gres=gpu:4 --time=48:00:00 bindcraft.slurm examples/pdl1_denovo.json
#
# The two things this file cannot guess are the account and the partition your site charges work to.
# Pass them to sbatch, or fill them in here:
#
# #SBATCH --account=my_account
# #SBATCH --partition=gpu
#
# Submit from the repository root, or point BINDCRAFT_HOME at it, since Slurm runs a copy of this
# script and the copy cannot find the repository on its own.
set -euo pipefail

node_gpu_count() {
  # the largest number of GPUs any one node offers, read off the generic resources sinfo reports
  sinfo --noheader --format='%G' 2>/dev/null | tr ',' '\n' | sed 's/(.*//' \
    | sed -n 's/^gpu:.*:\([0-9][0-9]*\)$/\1/p; s/^gpu:\([0-9][0-9]*\)$/\1/p' | sort -rn | head -1 || true
}

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  gpus=${BINDCRAFT_GPUS:-$(node_gpu_count)}
  gpus=${gpus:-1}
  echo "bindcraft: submitting on ${gpus} GPU(s), $((4 * gpus)) cores and $((24 * gpus))G"
  exec sbatch --gres=gpu:"${gpus}" --cpus-per-task=$((4 * gpus)) --mem=$((24 * gpus))G "$0" "$@"
fi

cd "${BINDCRAFT_HOME:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate          # the environment install.sh builds, for a job that inherits no shell
fi
gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd', ' - || true)
echo "bindcraft: $(hostname), job ${SLURM_JOB_ID}, ${gpus:-no GPU visible}, $(pwd)"
python bindcraft.py "$@"
