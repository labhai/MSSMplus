#!/usr/bin/env bash

# Map the native FreeSurfer sulc and curv morphometry files of one recon-all
# subject to the fsaverage surface.  This intentionally does not process
# jacobian_white, pial_lgi, thickness, or MSSM GWC maps.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  surf2fsaverage.sh -r <recon-all subject directory> [-f]

Options:
  -r, --recon-dir DIR  Subject directory produced by recon-all. It must
                       contain surf/lh.sphere.reg and surf/rh.sphere.reg.
  -f, --force          Overwrite outputs that already exist.
  -h, --help           Show this help.

Example:
  bash scripts/surf2fsaverage.sh -r /data/recons/sub-001
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

RECON_DIR=""
FORCE=0

while (( $# > 0 )); do
  case "$1" in
    -r|--recon-dir)
      (( $# >= 2 )) || die "$1 requires a directory argument."
      RECON_DIR="$2"
      shift 2
      ;;
    -f|--force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$RECON_DIR" ]] || { usage >&2; die "--recon-dir is required."; }
command -v mri_surf2surf >/dev/null 2>&1 || die "mri_surf2surf is not on PATH. Source FreeSurfer's SetUpFreeSurfer.sh first."

RECON_DIR="$(readlink -f "$RECON_DIR")"
[[ -d "$RECON_DIR/surf" ]] || die "missing surf directory: $RECON_DIR/surf"

SUBJECTS_DIR="$(dirname "$RECON_DIR")"
SUBJECT="$(basename "$RECON_DIR")"
export SUBJECTS_DIR

# mri_surf2surf resolves --trgsubject fsaverage below SUBJECTS_DIR.  Reuse an
# existing local copy/link, or link the template supplied by FreeSurfer.
if [[ -L "$SUBJECTS_DIR/fsaverage" && ! -e "$SUBJECTS_DIR/fsaverage" ]]; then
  die "fsaverage is a dangling symbolic link: $SUBJECTS_DIR/fsaverage"
fi

if [[ ! -e "$SUBJECTS_DIR/fsaverage" ]]; then
  [[ -n "${FREESURFER_HOME:-}" ]] || die "SUBJECTS_DIR/fsaverage is absent and FREESURFER_HOME is not set."
  FSAVERAGE_SOURCE="$FREESURFER_HOME/subjects/fsaverage"
  [[ -d "$FSAVERAGE_SOURCE" ]] || die "fsaverage template not found: $FSAVERAGE_SOURCE"
  ln -s "$FSAVERAGE_SOURCE" "$SUBJECTS_DIR/fsaverage"
  printf 'Linked fsaverage: %s -> %s\n' "$SUBJECTS_DIR/fsaverage" "$FSAVERAGE_SOURCE"
fi
[[ -d "$SUBJECTS_DIR/fsaverage" ]] || die "fsaverage target is not a directory: $SUBJECTS_DIR/fsaverage"

HEMIS=(lh rh)
FEATURES=(sulc curv)

for hemi in "${HEMIS[@]}"; do
  [[ -f "$RECON_DIR/surf/$hemi.sphere.reg" ]] || die "missing registration surface: $RECON_DIR/surf/$hemi.sphere.reg"
  for feature in "${FEATURES[@]}"; do
    [[ -f "$RECON_DIR/surf/$hemi.$feature" ]] || die "missing source morphometry: $RECON_DIR/surf/$hemi.$feature"
  done
done

printf 'Mapping subject %s to fsaverage (SUBJECTS_DIR=%s)\n' "$SUBJECT" "$SUBJECTS_DIR"

for hemi in "${HEMIS[@]}"; do
  for feature in "${FEATURES[@]}"; do
    src="$RECON_DIR/surf/$hemi.$feature"
    trg="$RECON_DIR/surf/$hemi.fsaverage.$feature"

    if [[ -s "$trg" && "$FORCE" -eq 0 ]]; then
      printf 'Skip existing: %s\n' "$trg"
      continue
    fi

    printf 'Map %s -> %s\n' "$src" "$trg"
    mri_surf2surf \
      --hemi "$hemi" \
      --srcsubject "$SUBJECT" \
      --trgsubject fsaverage \
      --srcsurfval "$src" \
      --trgsurfval "$trg" \
      --src_type curv \
      --trg_type curv \
      --surfreg sphere.reg
  done
done

printf 'Done: %s\n' "$RECON_DIR"
