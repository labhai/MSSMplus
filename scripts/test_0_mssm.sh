#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# set default yaml path
CONFIG_FILE="${SCRIPT_DIR}/../configs/config.yaml"
mssm_cand=""

while getopts ":c:m:" opt; do
  case ${opt} in
    c )
      
      CONFIG_FILE="$OPTARG"
      ;;
    m )
      mssm_cand="$OPTARG"
      ;;
    \? )
      echo "Usage: cmd [-c config_file] [-m mssm_cand]"
      exit 1
      ;;
  esac
done

# config check
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file $CONFIG_FILE not found!"
    exit 1
fi

if [[ -z "$mssm_cand" ]]; then
    echo "Error: mssm_cand value is missing. Please provide it with -m option."
    exit 1
fi

DATA_DIR=$(yq '.DATA_DIR' "$CONFIG_FILE")
RESULTS_DIR=$(yq '.RESULTS_DIR' "$CONFIG_FILE")
SPLITS=($(yq '.SPLITS[]' "$CONFIG_FILE"))
HEMIS=($(yq '.HEMIS[]' "$CONFIG_FILE"))
FWHM=$(yq '.FWHM' "$CONFIG_FILE")
BG=$(yq '.BG // 20' "$CONFIG_FILE")
THRESHOLD=$(yq '.THRESHOLD' "$CONFIG_FILE")
NAME=$(basename "$CONFIG_FILE" .yaml)

for split in "${SPLITS[@]}"; do
    python ./age_correction_pls.py --config "$CONFIG_FILE" --mssm_cand "$mssm_cand" --split "$split"
    subject_ids=$(awk '/Input/ {print $2}' "$RESULTS_DIR/${split}/${split}.fsgd")
    export SUBJECTS_DIR="${RESULTS_DIR}/${split}/${mssm_cand}"
    if [[ "$SUBJECTS_DIR" == "$FREESURFER_HOME/subjects"* ]]; then
        echo "Error: SUBJECTS_DIR points to FREESURFER_HOME/subjects! Aborting."
        exit 1
    fi
    mkdir -p $SUBJECTS_DIR
    ln -s $FREESURFER_HOME/subjects/fsaverage $SUBJECTS_DIR/fsaverage
    for hemi in "${HEMIS[@]}"; do

        mkdir -p "$SUBJECTS_DIR/${hemi}"
        mkdir -p "$SUBJECTS_DIR/${hemi}_smoothed"

        export SUBJECTS_DIR RESULTS_DIR FWHM hemi split

        awk '/Input/ {print $2}' "$RESULTS_DIR/${split}/${split}.fsgd" \
        | parallel -j "$BG" --will-cite \
            'mri_surf2surf \
              --srcsubject fsaverage \
              --sval "${SUBJECTS_DIR}/${hemi}/${hemi}.{1}.mssm" --sfmt curv \
              --trgsubject fsaverage \
              --tval "${SUBJECTS_DIR}/${hemi}_smoothed/${hemi}.{1}.mssm" --tfmt curv \
              --fwhm "${FWHM}" --hemi "${hemi}" --cortex'

        files=""
        for subject in $subject_ids; do
        file="${SUBJECTS_DIR}/${hemi}_smoothed/${hemi}.${subject}.mssm"
        if [ -f "$file" ]; then
            files="$files $file"
        else
            echo "Warning: $file not found"
        fi
        done

        mri_concat $files --o "${SUBJECTS_DIR}/${hemi}_concat_smoothed.mgh"

        if [ "$FWHM" -ne 0 ]; then
            rm -r "$SUBJECTS_DIR/${hemi}"
        fi
        rm -r "$SUBJECTS_DIR/${hemi}_smoothed"

        mri_glmfit --y "$SUBJECTS_DIR/${hemi}_concat_smoothed.mgh" \
                --fsgd "$RESULTS_DIR/${split}/${split}.fsgd" dods \
                --C $DATA_DIR/contrast.mtx \
                --surf fsaverage $hemi \
                --fwhm 0 --var-fwhm 0 \
                --seed 42 \
                --cortex --glmdir "$SUBJECTS_DIR/${hemi}_glm" --eres-save
        
        mri_glmfit-sim --glmdir "$SUBJECTS_DIR/${hemi}_glm" \
                    --perm 10000 "${THRESHOLD}" abs \
                    --cwp 0.05 \
                    --2spaces \
                    --bg "${BG}"
    done

done

echo "python ./cand_stats.py --config $CONFIG_FILE --candidate $mssm_cand --script_dir $SCRIPT_DIR --name $NAME"
python ./cand_stats.py --config "$CONFIG_FILE" --candidate "$mssm_cand" --script_dir "$SCRIPT_DIR" --name "$NAME"
