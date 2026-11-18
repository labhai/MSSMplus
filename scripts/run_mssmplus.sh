#!/usr/bin/env bash

set -euo pipefail

usage() {
  printf 'Usage: bash scripts/run_mssmplus.sh --config <config.yaml> [--split <name>]\n'
}

CONFIG=""
SPLIT=""
while (( $# > 0 )); do
  case "$1" in
    --config|-c)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      CONFIG="$2"
      shift 2
      ;;
    --split|-s)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      SPLIT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "$CONFIG" ]] || { usage >&2; exit 2; }
[[ -f "$CONFIG" ]] || { printf 'Config not found: %s\n' "$CONFIG" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
COMMAND=("$PYTHON_BIN" "$SCRIPT_DIR/pls_age_correction.py" --config "$CONFIG")
if [[ -n "$SPLIT" ]]; then
  COMMAND+=(--split "$SPLIT")
fi
exec "${COMMAND[@]}"
