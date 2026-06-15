#!/bin/bash -l
#SBATCH -A IscrC_GenAdapt
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --job-name=geco_gta_eval
#SBATCH --output=sbatch/logs/%x-%j.out
#SBATCH --error=sbatch/logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-geco}"
CHECKPOINT="${CHECKPOINT:-work_dirs/geco_dinov2_gta/iter_40000.pth}"
BACKBONE="${BACKBONE:-checkpoints/dinov2_geco_converted.pth}"
EVAL_ROOT="${EVAL_ROOT:-work_dirs/eval_geco_dinov2_gta}"
DOMAINS="${DOMAINS:-cityscapes bdd mapillary}"

cd "${REPO_DIR}"
mkdir -p "${EVAL_ROOT}" sbatch/logs

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "Checkpoint: ${CHECKPOINT}"
echo "Backbone: ${BACKBONE}"
echo "Eval domains: ${DOMAINS}"

if [[ ! -e "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 2
fi
if [[ ! -e "${BACKBONE}" ]]; then
    echo "Missing converted DINOv2 backbone: ${BACKBONE}" >&2
    exit 2
fi

run_eval() {
    local name="$1"
    local config="$2"
    local eval_dir="${EVAL_ROOT}/${name}"

    echo "===== Evaluating ${name} ====="
    python tools/test.py "${config}" "${CHECKPOINT}" --backbone "${BACKBONE}" --work-dir "${eval_dir}"
}

for domain in ${DOMAINS}; do
    case "${domain}" in
        cityscapes)
            run_eval cityscapes configs/geco/eval_geco_gta_to_cityscapes_512x512.py
            ;;
        bdd)
            run_eval bdd configs/geco/eval_geco_gta_to_bdd_512x512.py
            ;;
        mapillary)
            run_eval mapillary configs/geco/eval_geco_gta_to_mapillary_512x512.py
            ;;
        *)
            echo "Unknown domain: ${domain}" >&2
            exit 2
            ;;
    esac
done
