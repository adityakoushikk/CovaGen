#!/usr/bin/env bash
#SBATCH --job-name=covagen_chain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=covagen_chain_%A_%a.out
#SBATCH --error=covagen_chain_%A_%a.err

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COVAGEN_ROOT="${COVAGEN_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RL_DIR="${RL_DIR:-${COVAGEN_ROOT}/CovaGen_rl_guide}"
UNCOND_DIR="${UNCOND_DIR:-${COVAGEN_ROOT}/CovaGen_uncond_cond}"

PAIR_LIST="${PAIR_LIST:-${SCRIPT_DIR}/pdb_chain_pairs.txt}"
ENV_ACTIVATE="${ENV_ACTIVATE:-${COVAGEN_ROOT}/covagen/bin/activate}"
SAMPLE_SECONDS="${SAMPLE_SECONDS:-120}"

MODEL_PATH="${MODEL_PATH:-${RL_DIR}/Models/cond_model009000.pt}"
CLASSIFIER_PATH="${CLASSIFIER_PATH:-${RL_DIR}/Models/acrylamide_classifier_model010000.pt}"
CLASSIFIER_SCALE="${CLASSIFIER_SCALE:-1000}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-300}"
NOISE_SCHEDULE="${NOISE_SCHEDULE:-linear}"
DATASET_SAVE_PATH="${DATASET_SAVE_PATH:-${RL_DIR}/data}"
PDB_CACHE_DIR="${PDB_CACHE_DIR:-${COVAGEN_ROOT}/.pdb_cache}"
PDB_SOURCE="${PDB_SOURCE:-seqres}"

WARHEADS_YAML="${WARHEADS_YAML:-/scratch/users/akoushik/covmedsage/covalent/configs/warheads.yaml}"
WARHEAD_ID="${WARHEAD_ID:-3723}"
COVALENT_DATA_DIR="${COVALENT_DATA_DIR:-/scratch/users/akoushik/covmedsage/covalent/data}"
CLASSIFIER_KEEP_N="${CLASSIFIER_KEEP_N:-100}"
RANDOM_SEED="${RANDOM_SEED:-114514}"

mkdir -p "${RL_DIR}/samples" "${DATASET_SAVE_PATH}" "${PDB_CACHE_DIR}" "${COVALENT_DATA_DIR}"

if [[ ! -f "${PAIR_LIST}" ]]; then
    echo "Pair list not found: ${PAIR_LIST}" >&2
    echo "Copy slurm/pdb_chain_pairs.example.txt to ${PAIR_LIST} and edit it." >&2
    exit 2
fi

mapfile -t PAIRS < <(grep -Ev '^[[:space:]]*(#|$)' "${PAIR_LIST}" | tr -d '\r')
if [[ "${#PAIRS[@]}" -eq 0 ]]; then
    echo "Pair list contains no PDB:CHAIN entries: ${PAIR_LIST}" >&2
    exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
if (( TASK_ID < 1 || TASK_ID > ${#PAIRS[@]} )); then
    echo "SLURM_ARRAY_TASK_ID=${TASK_ID} is outside 1..${#PAIRS[@]}" >&2
    exit 2
fi

PAIR="${PAIRS[$((TASK_ID - 1))]}"
PDB_ID="${PAIR%%:*}"
CHAIN_ID="${PAIR#*:}"
TAG="$(printf '%s_%s' "${PDB_ID}" "${CHAIN_ID}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_' '_')"

SAMPLE_PKL="${RL_DIR}/samples/my_run_${TAG}.pkl"
COVGEN_SMI="${COVALENT_DATA_DIR}/covgen_${TAG}.smi"
LIGPREP_DIR="${COVALENT_DATA_DIR}/enumerate_frag/covagen_enum/${TAG}"
LIGPREP_MAE="${LIGPREP_DIR}/${TAG}_placed.mae"
SEQUENCE_TXT="${RL_DIR}/samples/${TAG}_sequence.txt"

echo "[$(date)] Starting CovaGen chain job"
echo "Pair: ${PAIR}"
echo "Tag: ${TAG}"
echo "CovaGen root: ${COVAGEN_ROOT}"

if [[ ! -f "${ENV_ACTIVATE}" ]]; then
    echo "Environment activation script not found: ${ENV_ACTIVATE}" >&2
    echo "Set ENV_ACTIVATE=/path/to/bin/activate when submitting if needed." >&2
    exit 2
fi

# Cluster module systems usually expose either `module` or `ml`.
if command -v module >/dev/null 2>&1; then
    module load openmpi/4.1.6
elif command -v ml >/dev/null 2>&1; then
    ml load openmpi/4.1.6
fi

source "${ENV_ACTIVATE}"

# mpi4py on this cluster may need conda/venv shared libs visible at runtime.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH:-}"
fi

FETCH_ARGS=("${PAIR}" "--cache-dir" "${PDB_CACHE_DIR}" "--source" "${PDB_SOURCE}" "--output" "${SEQUENCE_TXT}")
if [[ -n "${PDB_DIR:-}" ]]; then
    FETCH_ARGS+=("--pdb-dir" "${PDB_DIR}")
fi
python "${SCRIPT_DIR}/fetch_pdb_chain_sequence.py" "${FETCH_ARGS[@]}"
PROTEIN_SEQ="$(tr -d '[:space:]' < "${SEQUENCE_TXT}")"
echo "Protein sequence length: ${#PROTEIN_SEQ}"

cd "${RL_DIR}"
rm -f "${SAMPLE_PKL}"

echo "[$(date)] Sampling for ${SAMPLE_SECONDS}s, then sending SIGINT"
set +e
timeout --signal=INT --kill-after=30s "${SAMPLE_SECONDS}s" \
    python scripts/rl_sample_guided_singleseq.py \
        --model_path "${MODEL_PATH}" \
        --classifier_path "${CLASSIFIER_PATH}" \
        --classifier_scale "${CLASSIFIER_SCALE}" \
        --protein_seq "${PROTEIN_SEQ}" \
        --save_path "${SAMPLE_PKL}" \
        --class_cond False \
        --rescale_timesteps False \
        --diffusion_steps "${DIFFUSION_STEPS}" \
        --noise_schedule "${NOISE_SCHEDULE}" \
        --dataset_save_path "${DATASET_SAVE_PATH}"
SAMPLE_STATUS=$?
set -e

case "${SAMPLE_STATUS}" in
    0|124|130|137)
        echo "Sampling finished or was interrupted as planned; status=${SAMPLE_STATUS}"
        ;;
    *)
        echo "Sampling failed with status=${SAMPLE_STATUS}" >&2
        exit "${SAMPLE_STATUS}"
        ;;
esac

if [[ ! -s "${SAMPLE_PKL}" ]]; then
    echo "No sampled pickle was produced at ${SAMPLE_PKL}" >&2
    echo "Increase SAMPLE_SECONDS if the first 300-step sample did not finish before timeout." >&2
    exit 3
fi

echo "[$(date)] Attaching/filtering warhead"
python "${UNCOND_DIR}/scripts/attach_warhead.py" \
    --input "${SAMPLE_PKL}" \
    --warheads_yaml "${WARHEADS_YAML}" \
    --warhead_id "${WARHEAD_ID}" \
    --output "${COVGEN_SMI}" \
    --covagen_atoms C,N,O,S \
    --with_classifier \
    --classifier_keep_n "${CLASSIFIER_KEEP_N}" \
    --random_seed "${RANDOM_SEED}"

if [[ ! -s "${COVGEN_SMI}" ]]; then
    echo "No SMILES written at ${COVGEN_SMI}" >&2
    exit 4
fi

echo "[$(date)] Running LigPrep"
if command -v module >/dev/null 2>&1; then
    module load chemistry schrodinger
elif command -v ml >/dev/null 2>&1; then
    ml load chemistry schrodinger
fi

: "${SCHRODINGER:?SCHRODINGER is not set after loading the Schrodinger module}"
mkdir -p "${LIGPREP_DIR}"
"${SCHRODINGER}/ligprep" \
    -WAIT \
    -epik \
    -ismi "${COVGEN_SMI}" \
    -omae "${LIGPREP_MAE}"

echo "[$(date)] Done"
echo "Sample pickle: ${SAMPLE_PKL}"
echo "Warhead SMILES: ${COVGEN_SMI}"
echo "LigPrep MAE: ${LIGPREP_MAE}"
