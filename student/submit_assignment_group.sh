#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <group>"
  exit 1
fi

GROUP="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python3 "${REPO_ROOT}/student/assignment_experiments.py" list --group "${GROUP}" | while IFS=$'\t' read -r name group readiness description; do
  if [ "${readiness}" = "missing-runner" ]; then
    echo "Skipping ${name}: runner is not implemented yet."
    continue
  fi

  echo "Submitting ${name}"
  sbatch --job-name="${name}" --export=ALL,EXPERIMENT_NAME="${name}" "${REPO_ROOT}/student/run_assignment_experiment.sh"
done
