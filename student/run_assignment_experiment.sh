#!/bin/bash
#SBATCH --job-name=a3-exp
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --requeue
#SBATCH --mail-type=all
#SBATCH --mail-user=sv2279@nyu.edu


set -euo pipefail

if [ -z "${EXPERIMENT_NAME:-}" ]; then
  echo "Set EXPERIMENT_NAME before submitting this job."
  exit 1
fi

CLUSTER_PROJECT_ROOT="${CLUSTER_PROJECT_ROOT:-/scratch/sv2279/SFT-RL}"
SINGULARITY_OVERLAY="${SINGULARITY_OVERLAY:-/scratch/sv2279/overlay-25GB-500K.ext3:r}"
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-/scratch/sv2279/ubuntu-20.04.3.sif}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-llmr}"
HPC_ENV_FILE="${HPC_ENV_FILE:-${CLUSTER_PROJECT_ROOT}/student/hpc_sv2279.env}"

mkdir -p logs

singularity exec --bind /scratch --nv \
--overlay /scratch/sv2279/overlay-25GB-500K.ext3:r \
/scratch/sv2279/ubuntu-20.04.3.sif \
/bin/bash -c

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
set -euo pipefail

if [ -f '${HPC_ENV_FILE}' ]; then
  set -a
  source '${HPC_ENV_FILE}'
  set +a
fi

conda activate ${CONDA_ENV_NAME}
cd ${CLUSTER_PROJECT_ROOT}

mkdir -p logs
CMD=\$(python student/assignment_experiments.py command --name '${EXPERIMENT_NAME}')
echo Running experiment: ${EXPERIMENT_NAME}
echo \$CMD
eval \$CMD
"
