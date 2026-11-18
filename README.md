# MSSM\+
Improved Multiscale Structural Mapping with Supervertex Vision Transformer for the Detection of Alzheimer's Disease Neurodegeneration Repository

This repository contains the MSSM+ generation, SSVM preprocessing, and Supervertex Vision Transformer (SV-ViT) training pipeline.

---

## Repository structure

- `configs/`
  - `MSSMp.yaml`: configuration for MSSM+ map generation and optional age correction.
  - `sv-vit.yaml`: configuration for training/evaluation.
- `models/`
  - `sit.py`: cited SiT backbone implementation (Dahan et al., 2022).
  - `svvit.py`: SV-ViT wrapper that validates the SSVM input shape.
- `data/`
  - `ssvm_ico7_sub_ico3.npy`: bilateral SSVM indices with shape `(2560, 147, 3)`.
- `scripts/`
  - `pls_age_correction.py`: perform the MSSM+ procedure with optional age correction.
  - `run_mssmplus.sh`: fixed MSSM+ command-line entry point.
  - `surf2fsaverage.sh`: map FreeSurfer `sulc` and `curv` to `fsaverage`.
  - `train-sv-vit.py`: train and evaluate SV-ViT.
- `requirements.txt`: minimal Python dependencies.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Citation

```bibtex
@article{baek2026mssmplus,
  title={Improved Multiscale Structural Mapping with Supervertex Vision Transformer for the Detection of Alzheimer's Disease Neurodegeneration},
  author={Baek, Geonwoo and Salat, David H. and Jang, Ikbeom and {Alzheimer's Disease Neuroimaging Initiative}},
  journal={Human Brain Mapping},
  volume={47},
  number={8},
  pages={e70548},
  year={2026},
  publisher={Wiley Online Library},
  doi={10.1002/hbm.70548}
}
```

---

## Map `sulc` and `curv` to `fsaverage`

MSSM+ uses input combination: **MSSM + sulc + curv**. FreeSurfer's `recon-all` produces `sulc` and `curv` in each subject's native surface space, so they must be resampled to the same `fsaverage` space used by MSSM before running MSSM+.

[`scripts/surf2fsaverage.sh`](scripts/surf2fsaverage.sh) performs only this resampling. It maps:

- `lh.sulc` and `rh.sulc`
- `lh.curv` and `rh.curv`

It does not modify thickness or GWC files.

### Requirements

- A completed FreeSurfer `recon-all` subject.
- `surf/{lh,rh}.sphere.reg`, `surf/{lh,rh}.sulc`, and `surf/{lh,rh}.curv` in the subject directory.
- FreeSurfer initialized so that `mri_surf2surf` is on `PATH` and `FREESURFER_HOME` is set.
- The subject directory must be an immediate child of its FreeSurfer `SUBJECTS_DIR`.

### Run one subject

```bash
export FREESURFER_HOME=/path/to/freesurfer
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"

bash scripts/surf2fsaverage.sh \
  -r /data/recons/sub-001
```

The script derives `SUBJECTS_DIR` from the parent of `-r`. If `SUBJECTS_DIR/fsaverage` does not exist, it links `$FREESURFER_HOME/subjects/fsaverage`. Existing non-empty outputs are skipped; pass `--force` to regenerate them.

### Run multiple subjects

```bash
SUBJECTS_DIR=/data/recons
for recon_dir in "$SUBJECTS_DIR"/sub-*; do
  bash scripts/surf2fsaverage.sh -r "$recon_dir"
done
```

With GNU Parallel:

```bash
find /data/recons -mindepth 1 -maxdepth 1 -type d ! -name fsaverage -print0 \
  | parallel -0 -j 8 bash scripts/surf2fsaverage.sh -r {}
```

### Outputs

```text
/data/recons/sub-001/surf/
├── lh.fsaverage.sulc
├── rh.fsaverage.sulc
├── lh.fsaverage.curv
└── rh.fsaverage.curv
```

Quick check:

```bash
for hemi in lh rh; do
  for feature in sulc curv; do
    test -s "/data/recons/sub-001/surf/${hemi}.fsaverage.${feature}" \
      || echo "missing: ${hemi}.fsaverage.${feature}"
  done
done
```

## End-to-end workflow: MSSM to MSSM+

MSSM+ is the next processing stage after MSSM. The final MSSM+ input is always **MSSM + sulc + curv**.

```mermaid
flowchart LR
    A[T1-weighted MRI] --> B[FreeSurfer recon-all]
    B --> C[MSSM preprocessing]
    C --> D[GWC and CT maps on fsaverage]
    B --> E[sulc and curv to fsaverage]
    D --> F[MSSM + sulc + curv integration]
    E --> F
    F --> G{Apply age correction?}
    G -->|yes| H[Age-corrected MSSM+ maps]
    G -->|no| I[Uncorrected MSSM+ maps]
    H --> J[Surface statistics or SV-ViT]
    I --> J
```

The source repositories are [jibikbam/MSSM](https://github.com/jibikbam/MSSM) and [labhai/MSSMplus](https://github.com/labhai/MSSMplus).

### 1. Run FreeSurfer

Use FreeSurfer 7.4.1 and the same `fsaverage` template for every subject.

```bash
export SUBJECTS_DIR=/data/recons
recon-all -sd "$SUBJECTS_DIR" -s sub-001 -i /data/T1/sub-001_T1w.nii.gz -all
```

Complete FreeSurfer QC before continuing. Both hemispheres must contain at least `white`, `pial`, `sphere.reg`, `thickness`, `sulc`, and `curv`.

### 2. Run MSSM

From the MSSM repository's `code/` directory:

```bash
cd /path/to/MSSM/code
bash mssm.sh /data/recons 16 'sub-*' 0
```

The arguments are:

1. FreeSurfer `SUBJECTS_DIR`.
2. Available logical CPU count. The current MSSM shell implementation should be given at least 14.
3. Subject-name shell pattern. Quote it to prevent early expansion by the calling shell.
4. Surface-smoothing FWHM in millimetres.

MSSM generates the GWC-derived inputs and registered thickness used to construct the MSSM representation. The important unsmoothed outputs are:

```text
<subject>/mri/GWC/reg_surf2fsaverage/<GWC files>
<subject>/surf/reg_surf2fsaverage/lh.thickness
<subject>/surf/reg_surf2fsaverage/rh.thickness
```

MSSM also creates `reg_surf2fsaverage_smoothed<FWHM>mm` directories. The same smoothing policy must be used for all subjects, and inputs must not be smoothed twice.

### 3. Map `sulc` and `curv`

Run `scripts/surf2fsaverage.sh` for every recon-all subject. After MSSM and surface mapping, each subject must provide:

```text
<subject>/
├── mri/GWC/reg_surf2fsaverage/
│   ├── lh_subcortwm-0.5divgm.2.nii.gz
│   ├── rh_subcortwm-0.5divgm.2.nii.gz
│   └── ...
└── surf/
    ├── reg_surf2fsaverage/lh.thickness
    ├── reg_surf2fsaverage/rh.thickness
    ├── lh.fsaverage.sulc
    ├── rh.fsaverage.sulc
    ├── lh.fsaverage.curv
    └── rh.fsaverage.curv
```

These inputs are fixed for MSSM+. No additional surface measure is part of this pipeline.

### 4. Arrange the MSSM+ dataset and metadata

The MSSM+ loader resolves subjects using this layout:

```text
<DATA_DIR>/<Split>/<Group>/<Subject>/
```

Symlink the completed recon-all directories instead of duplicating them:

```bash
mkdir -p /datasets/All/AD /datasets/All/CN
ln -s /data/recons/sub-001 /datasets/All/AD/sub-001
ln -s /data/recons/sub-002 /datasets/All/CN/sub-002
```

For this example, set `DATA_DIR: /datasets` and `SPLITS: [All]`. The metadata CSV at `INFO_PATH` must contain at least:

```csv
Subject,Age,Group,Split
sub-001,74,AD,All
sub-002,72,CN,All
```

Subject IDs, group names, and split names must exactly match their directory names and configuration values. With standard `fsaverage`, each hemisphere has 163,842 vertices and the bilateral `N_VERTEX` is 327,684.

The default binary comparison is AD versus CN. To use another binary
comparison, change both `GROUPS` and `LABEL_MAP`.

Use the same `LABEL_MAP` in `configs/sv-vit.yaml` so MSSM+ generation and
SV-ViT assign identical targets.

### 5. Generate fixed MSSM+ maps

Create a cohort-specific copy of `configs/MSSMp.yaml` and set `DATA_DIR`,
`INFO_PATH`, and `MAP_DIR`. The loader always constructs exactly 11 bilateral
features in this order:

```text
8 GWCs + CT + Sulc + Curv
```

The processing sequence is fixed:

```text
MSSM representation + fsaverage sulc + fsaverage curv
  -> vertex-wise MSSM+ integration
  -> optional age correction
  -> subject-level and concatenated MSSM+ surface maps
```

The resulting maps are written as left- and right-hemisphere FreeSurfer surface data. Keep the input order, vertex count, `fsaverage` template, and smoothing policy identical across all training, validation, and test subjects.

Run all configured splits, or one named split:

```bash
bash scripts/run_mssmplus.sh --config configs/MSSMp.yaml
bash scripts/run_mssmplus.sh --config configs/MSSMp.yaml --split All
```

Outputs are written to `<MAP_DIR>/<Split>/MSSMplus/`. Alongside
`lh_concat.mgh` and `rh_concat.mgh`, `subjects.csv` records the exact MGH frame
order. SV-ViT validates this sidecar against the metadata before assigning
labels.

### 6. Age-correction branches

Age correction changes the values of the fixed MSSM+ map; it does not change which inputs are used.

#### With age correction

The reference configuration fits polynomial age models using an independent
set of 150 CN participants. Configure that cohort with the same fixed inputs
and directory layout:

```yaml
CORRECTION: true
AGE_REFERENCE_DATA_DIR: /datasets/age-reference
AGE_REFERENCE_INFO_PATH: /datasets/age-reference/info.csv
AGE_REFERENCE_SPLIT: AgeReference
EXPECTED_AGE_REFERENCE_SIZE: 150
AGE_REFERENCE_STRATIFY_COLUMNS: [Vendor, Sex]
EXPECTED_AGE_REFERENCE_STRATA: 6
EXPECTED_AGE_REFERENCE_PER_STRATUM: 25
POLYNOMIAL_DEGREE: 1
```

When age correction is enabled, an independent CN reference cohort is used to estimate and remove age-related effects from the MSSM+ maps. The
reference cohort must belong to the group mapped to `0` in `LABEL_MAP` and
match the configured count. For the default AD/CN setting, the reference CSV
therefore also needs `Vendor` and `Sex`; the script enforces 25 CN participants
in each of the six vendor-by-sex strata. These subjects must not be reused in
downstream statistics or classification.

#### Without age correction

No reference cohort is loaded. Set:

```yaml
CORRECTION: false
```

The no-correction flow is:

```text
recon-all
  -> MSSM preprocessing and fsaverage mapping
  -> sulc/curv fsaverage mapping
  -> fixed MSSM + sulc + curv integration
  -> uncorrected MSSM+ maps
  -> surface statistics or SV-ViT
```

Keep age in the metadata even when correction is disabled so it remains available as a downstream covariate or for sensitivity analysis.

### 7. Downstream analysis

- Surface statistics: smooth the final MSSM+ maps at the selected FWHM and run the FreeSurfer GLM/permutation workflow.
- SV-ViT classification: use unsmoothed, age-corrected residual maps. Configure `configs/sv-vit.yaml` to read `lh_concat.mgh`, `rh_concat.mgh`, the metadata CSV, and the SSVM NPY.

All downstream analyses must use the same fixed MSSM + sulc + curv definition. Do not mix maps generated with different FreeSurfer templates, surface registration, smoothing, or age-correction settings.

For classification, PLS should be fitted separately within each fold.

### 8. SSVM preprocessing and SV-ViT

The SSVM contains 1,280 ROI-constrained, edge-connected supervertices per hemisphere. The supplied bilateral index array has shape:

```text
(2560 supervertices, 147 faces, 3 vertex positions)
```

The last two dimensions are flattened without spatial averaging. The resulting model input is:

```text
LH MSSM+ residual map ─┐
                      ├─ SSVM gather ─> (subjects, 1, 2560, 441)
RH MSSM+ residual map ─┘
```

The repeated vertex indices created by adjacent faces are intentional: the input preserves the three vertex positions of every retained face. Do not apply a patch-wise mean, max, or other hand-crafted pooling before the learned linear embedding.

SV-ViT extends SiT by replacing its surface patching with SSVM. Accordingly,
[`models/sit.py`](models/sit.py)
contains the SiT backbone and [`models/svvit.py`](models/svvit.py) is a thin
wrapper that validates the SSVM tensor before calling `SiT`:

```text
441 values per supervertex
  -> learned linear embedding
  -> CLS token + learned positional embedding
  -> stacked Transformer encoders
  -> CLS token
  -> LayerNorm + binary classification head
```

The matching model settings are fixed in `configs/sv-vit.yaml`:

```yaml
MODEL_CONFIG:
  pool: "cls"
  num_patches: 2560
  num_channels: 1
  num_vertices: 441
```

Set the cohort paths in the configuration and run from the repository root:

```bash
python scripts/train-sv-vit.py --config configs/sv-vit.yaml
```

Use (age-corrected,) unsmoothed residual maps for SV-ViT classification.
