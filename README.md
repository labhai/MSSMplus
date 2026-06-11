# MSSM\+
Improved Multiscale Structural Mapping with Surface Vision Transformer for the Detection of Alzheimer's Disease Neurodegeneration Repository

This repository contains code for the Supervertex Vision Transformer (SV-ViT) and patches map partitioning cortical surface.

---

## Repository structure

- `configs/`
  - `MSSMp.yaml`: configuration for cortical surface partitioning and CSV-map generation.
  - `sv-vit.yaml`: configuration for training/evaluation.
- `data/`
  - `triangle_indices_ico_7_sub_ico_3.csv`: SV maps (concatenated lh and rh hemisphere).
- `scripts/`
  - `pls_age_correction.py`: perform MSSM+ procedure (with pls and age correction)
  - `test.sh` `test_0_mssm.sh`: MSSM+ pipeline with group analysis
  - `train_age_correction_model.py`: code for training age correction models
  - `train-sv-vit.py`: train and evaluate SV-ViT
- `requirements.txt`: minimal Python dependencies.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 📖 BibTeX (to cite our paper)

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
