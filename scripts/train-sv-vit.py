import os
import sys
import argparse
import yaml
from pathlib import Path

from copy import deepcopy

# Set PYTHONPATH
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Configuration path, for example configs/sv-vit.yaml",
)
args = parser.parse_args()


def resolve_config_path(config_arg):
    requested = Path(config_arg).expanduser()
    names = [requested]
    if requested.suffix not in {".yaml", ".yml"}:
        names.extend([requested.with_suffix(".yaml"), requested.with_suffix(".yml")])

    search_paths = []
    for name in names:
        if name.is_absolute():
            search_paths.append(name)
            continue
        search_paths.extend(
            [Path.cwd() / name, project_root / name, project_root / "configs" / name]
        )
    for path in search_paths:
        if path.is_file():
            return path.resolve()
    searched = "\n  ".join(str(path) for path in search_paths)
    raise FileNotFoundError(f"Configuration file not found. Searched:\n  {searched}")


def resolve_project_path(path_value):
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else project_root / path


config_path = resolve_config_path(args.config)
print(f"Config: {config_path}")

# load config
with open(config_path, "r") as f:
    CONFIG = yaml.safe_load(f)


def binary_label_map(config):
    """Return a validated binary group-to-label mapping."""
    raw_mapping = config.get("LABEL_MAP", {"CN": 0, "AD": 1})
    if not isinstance(raw_mapping, dict) or len(raw_mapping) != 2:
        raise ValueError("LABEL_MAP must map exactly two group names to 0 and 1.")
    if any(type(label) is not int for label in raw_mapping.values()):
        raise ValueError("LABEL_MAP values must be the integers 0 and 1.")
    mapping = {str(group): label for group, label in raw_mapping.items()}
    if set(mapping.values()) != {0, 1}:
        raise ValueError("LABEL_MAP must contain each binary label exactly once: 0 and 1.")
    return mapping


LABEL_MAP = binary_label_map(CONFIG)

os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.get("CUDA_DEVICE", 0))
os.environ.setdefault("WANDB_MODE", str(CONFIG.get("WANDB_MODE", "disabled")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import gc
import random
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from models.svvit import SVViT


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CONFIG.get("SEED", 42))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

base_path = CONFIG["BASE_PATH"]
result_path = CONFIG["RESULT_PATH"]
info_path = CONFIG["INFO_PATH"]
datasets = CONFIG["DATASETS"]
mssmplus_dirname = str(CONFIG.get("MSSMPLUS_DIRNAME", "MSSMplus"))
ssvm_index_path = resolve_project_path(CONFIG["SSVM_INDEX_PATH"])
hemisphere_vertices = int(CONFIG.get("HEMISPHERE_VERTICES", 163842))


def load_concatenated_mgh(file_path, num_subjects):
    """Load an MGH surface stack as (subjects, vertices)."""
    data = np.asarray(nib.load(file_path).get_fdata(), dtype=np.float32)
    if data.ndim == 4 and data.shape[1:3] == (1, 1):
        data = data[:, 0, 0, :].T
    else:
        data = np.squeeze(data)
        if data.ndim != 2:
            raise ValueError(f"Unsupported MGH shape {data.shape}: {file_path}")
        if data.shape[0] != num_subjects and data.shape[1] == num_subjects:
            data = data.T

    expected = (num_subjects, hemisphere_vertices)
    if data.shape != expected:
        raise ValueError(f"Expected MGH shape {expected}, got {data.shape}: {file_path}")
    return data


def load_ssvm_indices(index_path):
    """Load SSVM indices and flatten (face, vertex) positions per patch."""
    indices = np.load(index_path)
    if indices.ndim == 3:
        if indices.shape[-1] != 3:
            raise ValueError(
                f"Expected SSVM shape (patches, faces, 3), got {indices.shape}."
            )
        indices = indices.reshape(indices.shape[0], -1)
    elif indices.ndim != 2:
        raise ValueError(
            f"Expected SSVM shape (patches, values) or (patches, faces, 3), got {indices.shape}."
        )

    indices = indices.astype(np.int64, copy=False)
    expected = (
        int(CONFIG["MODEL_CONFIG"]["num_patches"]),
        int(CONFIG["MODEL_CONFIG"]["num_vertices"]),
    )
    if indices.shape != expected:
        raise ValueError(
            f"SSVM index shape must match MODEL_CONFIG {expected}, got {indices.shape}."
        )
    if indices.shape[0] % 2:
        raise ValueError(
            "The bilateral SSVM must contain an equal number of LH/RH patches."
        )

    half = indices.shape[0] // 2
    if indices[:half].min() < 0 or indices[:half].max() >= hemisphere_vertices:
        raise ValueError("LH SSVM indices are outside the fsaverage hemisphere range.")
    if (
        indices[half:].min() < hemisphere_vertices
        or indices[half:].max() >= 2 * hemisphere_vertices
    ):
        raise ValueError("RH SSVM indices must use the bilateral fsaverage offset.")
    return indices


def load_and_preprocess_surface_data(dataset):
    map_dir = os.path.join(result_path, dataset, mssmplus_dirname)
    sidecar_path = os.path.join(map_dir, "subjects.csv")
    if not os.path.isfile(sidecar_path):
        raise FileNotFoundError(
            f"Missing MSSM+ subject-order sidecar: {sidecar_path}. "
            "Regenerate the maps with scripts/pls_age_correction.py."
        )
    sidecar = pd.read_csv(sidecar_path)
    required_sidecar = {"row_index", "Subject", "Group", "Split"}
    missing = sorted(required_sidecar.difference(sidecar.columns))
    if missing:
        raise ValueError(f"MSSM+ subjects.csv is missing columns: {missing}")
    expected_rows = np.arange(len(sidecar))
    if not np.array_equal(sidecar["row_index"].to_numpy(), expected_rows):
        raise ValueError("MSSM+ subjects.csv row_index is not contiguous from zero.")
    if not (sidecar["Split"].astype(str) == str(dataset)).all():
        raise ValueError(f"subjects.csv contains rows outside split {dataset!r}.")
    if sidecar["Subject"].duplicated().any():
        raise ValueError("MSSM+ subjects.csv contains duplicate subject IDs.")

    metadata = pd.read_csv(info_path)
    metadata = metadata[metadata["Split"].astype(str) == str(dataset)].copy()
    if metadata["Subject"].duplicated().any():
        raise ValueError(f"Metadata contains duplicate subjects in split {dataset!r}.")
    metadata = metadata.set_index("Subject")
    missing_subjects = sorted(set(sidecar["Subject"]) - set(metadata.index))
    if missing_subjects:
        raise ValueError(
            f"subjects.csv contains subjects absent from metadata: {missing_subjects[:5]}"
        )
    aligned_groups = metadata.loc[sidecar["Subject"], "Group"].to_numpy()
    if not np.array_equal(aligned_groups, sidecar["Group"].to_numpy()):
        raise ValueError("Group labels disagree between subjects.csv and metadata.")
    labels = sidecar["Group"].map(LABEL_MAP)
    if labels.isna().any():
        unknown_groups = sorted(
            set(sidecar.loc[labels.isna(), "Group"].astype(str))
        )
        raise ValueError(
            f"Groups are absent from LABEL_MAP: {unknown_groups}; "
            f"configured groups are {list(LABEL_MAP)}."
        )
    labels = labels.to_numpy(dtype=np.int64)
    subject_ids = sidecar["Subject"].astype(str).to_numpy()
    filename = "_concat"
    data_lh = load_concatenated_mgh(
        os.path.join(map_dir, f"lh{filename}.mgh"),
        len(labels),
    )
    data_rh = load_concatenated_mgh(
        os.path.join(map_dir, f"rh{filename}.mgh"),
        len(labels),
    )

    # SSVM contains 1,280 patches per hemisphere. Keep every
    # vertex occurrence in each face; no patch-wise mean is applied.
    indices = load_ssvm_indices(ssvm_index_path)
    half = indices.shape[0] // 2
    data_lh = data_lh[:, indices[:half]]
    data_rh = data_rh[:, indices[half:] - hemisphere_vertices]
    data_concat = np.concatenate([data_lh, data_rh], axis=1)
    data_concat = data_concat[:, np.newaxis, :, :].astype(np.float32, copy=False)

    expected = (
        len(labels),
        int(CONFIG["MODEL_CONFIG"]["num_channels"]),
        int(CONFIG["MODEL_CONFIG"]["num_patches"]),
        int(CONFIG["MODEL_CONFIG"]["num_vertices"]),
    )
    if data_concat.shape != expected:
        raise ValueError(
            f"Expected SV-ViT input shape {expected}, got {data_concat.shape}."
        )
    print(f"[SV-ViT] dataset={dataset}, input shape={data_concat.shape}")
    return data_concat, labels, subject_ids


class EarlyStopping:
    def __init__(self, patience=CONFIG.get("PATIENCE", 10)):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = deepcopy(model.state_dict())
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = deepcopy(model.state_dict())
            self.counter = 0


def train_svvit_model(X_train, y_train, X_valid, y_valid):
    batch_size = CONFIG["DATALOADER_CONFIG"]["batch_size"]
    epochs = CONFIG["EPOCHS"]
    lr = float(CONFIG["OPTIMIZER_CONFIG"]["lr"])
    weight_decay = float(CONFIG["OPTIMIZER_CONFIG"]["weight_decay"])

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), y_train_tensor
    )
    valid_dataset = TensorDataset(
        torch.tensor(X_valid, dtype=torch.float32), y_valid_tensor
    )

    num_workers = int(CONFIG["DATALOADER_CONFIG"].get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = SVViT(**CONFIG["MODEL_CONFIG"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    early_stopping = EarlyStopping(patience=CONFIG.get("PATIENCE", 10))

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            out = model(batch_X).view(-1)
            loss = criterion(out, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in valid_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                out = model(batch_X).view(-1)
                loss = criterion(out, batch_y)
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)

        scheduler.step()

        wandb.log({"train_loss": train_loss, "valid_loss": valid_loss, "epoch": epoch})

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"[Epoch {epoch}/{epochs}] Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}"
            )

        early_stopping(val_loss=valid_loss, model=model)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    if early_stopping.best_model is not None:
        model.load_state_dict(early_stopping.best_model)
    return model


def evaluate_svvit(X, y, model, batch_size=48):
    if len(y) == 0 or len(X) == 0:
        print("Warning: No samples provided for SV-ViT evaluation.")
        return 0.0, 0.0, np.array([])
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    num_workers = int(CONFIG["DATALOADER_CONFIG"].get("num_workers", 4))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    model.eval()
    y_scores = []

    with torch.no_grad():
        for batch in loader:
            batch_data = batch[0].to(device)
            out = model(batch_data)
            score = torch.sigmoid(out).view(-1).cpu().numpy()
            y_scores.extend(score)

    y_scores = np.array(y_scores)
    if len(y) == 0 or len(y_scores) == 0:
        print(
            "Warning: No predictions produced during SV-ViT evaluation."
        )
        return 0.0, 0.0, np.array([])
    try:
        auroc = roc_auc_score(y, y_scores)
    except ValueError:
        print("Warning: AUROC requires both classes.")
        auroc = 0.0
    try:
        auprc = average_precision_score(y, y_scores)
    except ValueError:
        print("Warning: AUPRC requires both classes.")
        auprc = 0.0
    return auroc, auprc, y_scores


def generate_fold_assignment(labels, cv_folds, seed=42):
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_assignment = np.zeros(len(labels), dtype=int)

    for fold_idx, (_, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        fold_assignment[val_idx] = fold_idx

    return fold_assignment


def process_mssmplus(dataset):
    from sklearn.model_selection import train_test_split

    data_concat, labels, subject_ids = load_and_preprocess_surface_data(dataset)
    labels = np.asarray(labels)
    output_dir = os.path.join(base_path, dataset, "MSSMplus")
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = os.path.join(base_path, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    cv_folds = CONFIG["TRAIN_SPLIT"]["cv_folds"]
    if not isinstance(cv_folds, int) or cv_folds <= 1:
        raise ValueError(
            f"Invalid cv_folds={cv_folds} in config. Must be an integer > 1."
        )
    valid_size = CONFIG["TRAIN_SPLIT"]["valid_size"]
    seed = CONFIG.get("SEED", 42)

    fold_file = os.path.join(dataset_dir, "fold_assignment.npy")
    fold_assignment = generate_fold_assignment(labels, cv_folds, seed=CONFIG["SEED"])
    np.save(fold_file, fold_assignment)

    fold_metrics = []
    pred_all = np.full(len(labels), np.nan)
    plot_data = {"train": [], "valid": [], "test": []}
    heldout_predictions = []
    for fold in range(cv_folds):
        test_idx = np.where(fold_assignment == fold)[0]
        trainvalid_idx = np.where(fold_assignment != fold)[0]
        y_trainval = labels[trainvalid_idx]
        train_idx_sub, valid_idx_sub = train_test_split(
            np.arange(len(trainvalid_idx)),
            test_size=valid_size,
            stratify=y_trainval,
            random_state=seed + fold,  # ensure reproducibility but different per fold
        )
        train_idx = trainvalid_idx[train_idx_sub]
        valid_idx = trainvalid_idx[valid_idx_sub]
        X_train, y_train = data_concat[train_idx], labels[train_idx]
        X_valid, y_valid = data_concat[valid_idx], labels[valid_idx]
        X_test, y_test = data_concat[test_idx], labels[test_idx]
        wandb.init(
            project=CONFIG.get("WANDB_PROJECT", "MSSMplus-SVViT"),
            name=f"{dataset}-MSSMplus-fold{fold}",
            config={
                "epochs": CONFIG["EPOCHS"],
                "batch_size": CONFIG["DATALOADER_CONFIG"]["batch_size"],
                "fold": fold,
                "dataset": dataset,
                "input": "MSSM+ (GWCs+CT+Sulc+Curv)",
            },
        )

        model = train_svvit_model(X_train, y_train, X_valid, y_valid)

        # Train/Valid predictions
        auroc_train, auprc_train, y_scores_train = evaluate_svvit(
            X_train,
            y_train,
            model,
            batch_size=CONFIG["DATALOADER_CONFIG"]["batch_size"],
        )
        auroc_valid, auprc_valid, y_scores_valid = evaluate_svvit(
            X_valid,
            y_valid,
            model,
            batch_size=CONFIG["DATALOADER_CONFIG"]["batch_size"],
        )
        # Test predictions (current fold's test set)
        auroc_test, auprc_test, y_scores_test = evaluate_svvit(
            X_test,
            y_test,
            model,
            batch_size=CONFIG["DATALOADER_CONFIG"]["batch_size"],
        )
        # Save validation predictions; held-out predictions replace them below.
        pred_all[valid_idx] = y_scores_valid
        heldout_predictions.append((test_idx, y_scores_test))

        # Save metrics for this fold
        fold_metrics.append(
            {
                "fold": fold,
                "train_AUROC": round(float(auroc_train), 5),
                "train_AUPRC": round(float(auprc_train), 5),
                "valid_AUROC": round(float(auroc_valid), 5),
                "valid_AUPRC": round(float(auprc_valid), 5),
                "test_AUROC": round(float(auroc_test), 5),
                "test_AUPRC": round(float(auprc_test), 5),
            }
        )

        print(
            f"Fold {fold}: [Train] AUROC {auroc_train:.3f}, AUPRC {auprc_train:.3f} | [Valid] AUROC {auroc_valid:.3f}, AUPRC {auprc_valid:.3f} | [Test] AUROC {auroc_test:.3f}, AUPRC {auprc_test:.3f}"
        )

        wandb.log(
            {
                "train_AUROC": auroc_train,
                "train_AUPRC": auprc_train,
                "valid_AUROC": auroc_valid,
                "valid_AUPRC": auprc_valid,
                "test_AUROC": auroc_test,
                "test_AUPRC": auprc_test,
            }
        )

        # Save model
        model_save_path = os.path.join(output_dir, f"svvit_fold{fold}.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model for fold {fold}: {model_save_path}")

        # --- ROC/PR curve visualization
        fpr_train, tpr_train, _ = roc_curve(y_train, y_scores_train)
        fpr_valid, tpr_valid, _ = roc_curve(y_valid, y_scores_valid)
        fpr_test, tpr_test, _ = roc_curve(y_test, y_scores_test)
        precision_train, recall_train, _ = precision_recall_curve(
            y_train, y_scores_train
        )
        precision_valid, recall_valid, _ = precision_recall_curve(
            y_valid, y_scores_valid
        )
        precision_test, recall_test, _ = precision_recall_curve(y_test, y_scores_test)

        # Fold-wise plot (ROC/PR)
        _, axes = plt.subplots(1, 2, figsize=(10, 5))
        # ROC
        axes[0].plot(
            fpr_train,
            tpr_train,
            color="blue",
            lw=2,
            label=f"Train AUROC = {auroc_train:.3f}",
        )
        axes[0].plot(
            fpr_valid,
            tpr_valid,
            color="orange",
            lw=2,
            label=f"Valid AUROC = {auroc_valid:.3f}",
        )
        axes[0].plot(
            fpr_test,
            tpr_test,
            color="green",
            lw=2,
            label=f"Test AUROC = {auroc_test:.3f}",
        )
        axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
        axes[0].set_xlabel("FPR")
        axes[0].set_ylabel("TPR")
        axes[0].set_title(f"Fold {fold} ROC curve")
        axes[0].legend(loc="lower right")
        # PR (오른쪽)
        axes[1].plot(
            recall_train,
            precision_train,
            color="blue",
            lw=2,
            label=f"Train AUPRC = {auprc_train:.3f}",
        )
        axes[1].plot(
            recall_valid,
            precision_valid,
            color="orange",
            lw=2,
            label=f"Valid AUPRC = {auprc_valid:.3f}",
        )
        axes[1].plot(
            recall_test,
            precision_test,
            color="green",
            lw=2,
            label=f"Test AUPRC = {auprc_test:.3f}",
        )
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title(f"Fold {fold} PR curve")
        axes[1].legend(loc="lower left")
        plt.tight_layout()
        fold_graph_file = os.path.join(output_dir, f"SVViT_fold{fold}_AUROC_AUPRC.png")
        plt.savefig(fold_graph_file)
        plt.close()
        print(f"Saved fold {fold} ROC/PR figure: {fold_graph_file}")

        # Instead of per-fold individual plots, accumulate data per split for all folds
        plot_data["train"].append(
            (
                fpr_train,
                tpr_train,
                recall_train,
                precision_train,
                auroc_train,
                auprc_train,
                fold,
            )
        )
        plot_data["valid"].append(
            (
                fpr_valid,
                tpr_valid,
                recall_valid,
                precision_valid,
                auroc_valid,
                auprc_valid,
                fold,
            )
        )
        plot_data["test"].append(
            (
                fpr_test,
                tpr_test,
                recall_test,
                precision_test,
                auroc_test,
                auprc_test,
                fold,
            )
        )

        wandb.finish()
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # After all folds: plot 1 figure per split (train/valid/test) with all folds as lines
    for split in ["train", "valid", "test"]:
        _, axes = plt.subplots(1, 2, figsize=(10, 5))
        # ROC curve
        for fpr, tpr, recall, precision, auroc, auprc, fold in plot_data[split]:
            axes[0].plot(
                fpr,
                tpr,
                lw=2,
                label=f"Fold {fold} AUROC = {auroc:.3f}",
            )
        axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
        axes[0].set_xlabel("FPR")
        axes[0].set_ylabel("TPR")
        axes[0].set_title(f"All Folds {split.capitalize()} ROC curve")
        axes[0].legend(loc="lower right", fontsize=8)
        # PR curve
        for fpr, tpr, recall, precision, auroc, auprc, fold in plot_data[split]:
            axes[1].plot(
                recall,
                precision,
                lw=2,
                label=f"Fold {fold} AUPRC = {auprc:.3f}",
            )
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title(f"All Folds {split.capitalize()} PR curve")
        axes[1].legend(loc="lower left", fontsize=8)
        plt.tight_layout()
        allfold_file = os.path.join(output_dir, f"SVViT_allfold_{split}.png")
        plt.savefig(allfold_file)
        plt.close()
        print(f"Saved all folds {split} ROC/PR figure: {allfold_file}")

    # Save fold_metrics.csv
    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_csv = os.path.join(output_dir, "fold_metrics.csv")
    fold_metrics_df.to_csv(fold_metrics_csv, index=False)
    print(f"Saved fold metrics to {fold_metrics_csv}")

    # Save metric_summary.csv (mean/std of each metric)
    metric_summary = {}
    for metric in [
        "train_AUROC",
        "train_AUPRC",
        "valid_AUROC",
        "valid_AUPRC",
        "test_AUROC",
        "test_AUPRC",
    ]:
        vals = fold_metrics_df[metric].values.astype(float)
        metric_summary[f"{metric}_mean"] = round(np.mean(vals), 5)
        metric_summary[f"{metric}_std"] = round(np.std(vals), 5)
    metric_summary_df = pd.DataFrame([metric_summary])
    metric_summary_csv = os.path.join(output_dir, "metric_summary.csv")
    metric_summary_df.to_csv(metric_summary_csv, index=False)
    print(f"Saved metric summary to {metric_summary_csv}")

    # Save one held-out prediction per subject.
    prediction_df = pd.DataFrame(
        {
            "Subject": subject_ids,
            "sample_index": np.arange(len(labels)),
            "true_label": labels,
            "predicted_prob": pred_all,
            "predicted_label": (pred_all >= 0.5).astype(float),
            "fold": fold_assignment,
        }
    )
    # For each fold, fill in test predictions for that fold's test indices
    for test_idx, y_scores_test in heldout_predictions:
        prediction_df.loc[test_idx, "predicted_prob"] = y_scores_test
        prediction_df.loc[test_idx, "predicted_label"] = (
            np.array(y_scores_test) >= 0.5
        ).astype(float)
    prediction_df.to_csv(
        os.path.join(output_dir, "SVViT_predictions.csv"),
        index=False,
    )
    print(f"Saved predictions to {os.path.join(output_dir, 'SVViT_predictions.csv')}")

    # Save the fixed MSSM+ result summary.
    result_df = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "input": "MSSMplus",
                "test_AUROC_mean": metric_summary["test_AUROC_mean"],
                "test_AUROC_std": metric_summary["test_AUROC_std"],
                "test_AUPRC_mean": metric_summary["test_AUPRC_mean"],
                "test_AUPRC_std": metric_summary["test_AUPRC_std"],
            }
        ]
    )
    result_df.to_csv(os.path.join(output_dir, "svvit_results.csv"), index=False)
    print(f"Saved result summary to {output_dir}/svvit_results.csv")


def main():
    for dataset in datasets:
        result_file = os.path.join(base_path, dataset, "MSSMplus", "svvit_results.csv")
        if os.path.exists(result_file):
            print(f"{dataset} MSSM+ result exists, skip: {result_file}")
            continue
        print(f"Processing fixed MSSM+ input for dataset {dataset}...")
        process_mssmplus(dataset)

    overall_results = []
    for dataset in datasets:
        result_file = os.path.join(base_path, dataset, "MSSMplus", "svvit_results.csv")
        if os.path.exists(result_file):
            overall_results.append(pd.read_csv(result_file))
    if overall_results:
        overall_df = pd.concat(overall_results, axis=0, ignore_index=True)
        overall_csv = os.path.join(base_path, "overall_results.csv")
        overall_df.to_csv(overall_csv, index=False)
        print(f"\nOverall results saved to {overall_csv}")


if __name__ == "__main__":
    main()
