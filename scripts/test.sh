#!/bin/bash

# Set conda env
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate tau

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# set default path
CONFIG_FILE="${SCRIPT_DIR}/../configs/config.yaml"

while getopts ":c:" opt; do
  case ${opt} in
    c )
      CONFIG_FILE="${SCRIPT_DIR}/../configs/${OPTARG}"
      ;;
    \? )
      echo "Usage: cmd [-c] config_file"
      exit 1
      ;;
  esac
done

# config check
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Config file '${CONFIG_FILE}' not found!"
    exit 1
fi

NAME=$(basename "$CONFIG_FILE" .yaml)

DATA_DIR=$(yq '.DATA_DIR' "$CONFIG_FILE")
RESULTS_DIR=$(yq '.RESULTS_DIR' "$CONFIG_FILE")

# make RESULTS_DIR
mkdir -p "$RESULTS_DIR"

# make LOG_DIR
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "$LOG_DIR"

for candidate in $(yq '.MSSM_CANDIDATES[]' "$CONFIG_FILE"); do
    ./test_0_mssm.sh -m "$candidate" -c "$CONFIG_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') $candidate end" >> "${LOG_DIR}/${NAME}.log"
done

echo "python ./stats.py --config $CONFIG_FILE"
python ./stats.py --config "$CONFIG_FILE"

echo "python ./results.py --config $CONFIG_FILE"
python ./results.py --config $CONFIG_FILE