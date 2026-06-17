#!/bin/bash -l
#SBATCH -A IscrC_GenAdapt
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --job-name=geco_gta
#SBATCH --output=sbatch/logs/%x-%j.out
#SBATCH --error=sbatch/logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-geco}"
CONFIG="${CONFIG:-configs/geco/geco_dinov2-L_mask2former_gta.py}"
WORK_DIR="${WORK_DIR:-work_dirs/geco_dinov2_gta}"
EVAL_ROOT="${EVAL_ROOT:-work_dirs/eval_geco_dinov2_gta}"
PRETRAIN="${PRETRAIN:-checkpoints/dinov2_vitl14_pretrain.pth}"
BACKBONE="${BACKBONE:-checkpoints/dinov2_geco_converted.pth}"
DOMAINS="${DOMAINS:-cityscapes bdd mapillary}"

cd "${REPO_DIR}"
mkdir -p "${WORK_DIR}" "${EVAL_ROOT}" sbatch/logs checkpoints

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Conda env: ${CONDA_ENV}"
echo "Train config: ${CONFIG}"
echo "Work dir: ${WORK_DIR}"
echo "Eval root: ${EVAL_ROOT}"
echo "Backbone: ${BACKBONE}"
echo "Eval domains: ${DOMAINS}"
echo "Git commit: $(git rev-parse --short HEAD || true)"
nvidia-smi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this node.")
PY

if [[ ! -e "${PRETRAIN}" ]]; then
    echo "Missing DINOv2 pretrain checkpoint: ${PRETRAIN}" >&2
    exit 2
fi

if [[ ! -e "${BACKBONE}" ]]; then
    echo "Converting ${PRETRAIN} -> ${BACKBONE}"
    python tools/convert_models/convert_dinov2_geco.py "${PRETRAIN}" "${BACKBONE}"
else
    echo "Found converted GeCo backbone: ${BACKBONE}"
fi

for dataset in data/gta data/cityscapes data/bdd100k data/mapillary; do
    if [[ ! -e "${dataset}" ]]; then
        echo "Missing dataset path: ${dataset}" >&2
        exit 2
    fi
done

echo "===== Training ====="
if [[ -f "${WORK_DIR}/last_checkpoint" ]]; then
    echo "Resuming from $(cat "${WORK_DIR}/last_checkpoint")"
    python tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" --resume
else
    echo "No last_checkpoint found in ${WORK_DIR}; starting a fresh run."
    python tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}"
fi

BEST_CHECKPOINT="$(
    find "${WORK_DIR}" -maxdepth 1 -type f -name 'best_*.pth' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | awk 'NR==1 {print $2}'
)"
if [[ -n "${BEST_CHECKPOINT}" ]]; then
    CHECKPOINT="${BEST_CHECKPOINT}"
    echo "Found best-val checkpoint: ${CHECKPOINT}"
elif [[ -f "${WORK_DIR}/last_checkpoint" ]]; then
    CHECKPOINT="$(cat "${WORK_DIR}/last_checkpoint")"
else
    CHECKPOINT="${WORK_DIR}/iter_40000.pth"
fi

if [[ ! -e "${CHECKPOINT}" ]]; then
    echo "Training finished but checkpoint was not found: ${CHECKPOINT}" >&2
    exit 2
fi

echo "===== Evaluation ====="
echo "Checkpoint: ${CHECKPOINT}"

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

echo "Finished: $(date)"
