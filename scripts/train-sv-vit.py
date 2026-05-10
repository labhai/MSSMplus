import os
import sys
import argparse
import yaml

from collections import defaultdict
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, train_test_split

# Set PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

print(f"✅ sys.path: {project_root}")

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Config filename without extension (e.g. All-0-post)")
parser.add_argument(
    "--project",
    type=str,
    default="renewal",
    help="Subdirectory under ../config/ where config.yaml is located (default: re)"
)
args = parser.parse_args()

# config path
config_path = os.path.abspath(f"../config/{args.project}/{args.config}.yaml")
print(f"📄 불러오는 config 파일: {config_path}")

# load config
with open(config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

# Patch offset for triangle indices, default to 0 if not set
patch_offset = int(CONFIG.get("PATCH_OFFSET", 0))

os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.get("CUDA_DEVICE", 0))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import gc
import random
import re
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
from sklearn.model_selection import StratifiedKFold, train_test_split

# SurfViT 관련 import
from models.sit import SiT


# 시드 고정
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
candidates = CONFIG["CANDIDATES"]
triangle_csv_path = CONFIG["TRIANGLE_CSV_PATH"]
is_smoothed = bool(CONFIG["ISSMOOTHED"])


def load_mgh(file_path):
    return nib.load(file_path).get_fdata()


def save_mgh(data_array, file_path, reference_mgh):
    img = nib.MGHImage(data_array, affine=nib.load(reference_mgh).affine)
    nib.save(img, file_path)


# Remove save_dataset_fold_info (now handled in process_candidate)


def load_and_preprocess_surface_data(dataset, candidate):
    df = pd.read_csv(os.path.join(info_path))
    df = df[df["Split"] == dataset]
    labels = (
        df["Group"]
        .map({"CN": 0, "AD": 1})
        .values
    )
    filename = "_concat"
    if is_smoothed:
        filename += "_smoothed"
    data_lh = load_mgh(
        os.path.join(result_path, dataset, candidate, f"lh{filename}.mgh")
    ).reshape(len(labels), -1, 1)
    data_rh = load_mgh(
        os.path.join(result_path, dataset, candidate, f"rh{filename}.mgh")
    ).reshape(len(labels), -1, 1)

    triangle_indices = load_triangle_indices(triangle_csv_path)
    #print('triangle indices shape:', triangle_indices.shape)
    # hemi-wise split (LH/RH)
    num_patches_total = triangle_indices.shape[0]
    half_patches = num_patches_total // 2  # 2560 → 1280

    triangle_indices_lh = triangle_indices[:half_patches, :]
    triangle_indices_rh = triangle_indices[half_patches:, :]   # (1280, 153)

    # LH/RH pooling
    data_lh = reshape_vertex_to_surface(data_lh, triangle_indices_lh, offset=0)
    data_rh = reshape_vertex_to_surface(data_rh, triangle_indices_rh, offset=0)

    # (N, 2, 1280)
    data_concat = np.stack([data_lh.squeeze(-1), data_rh.squeeze(-1)], axis=1)

    return data_concat, labels


def load_triangle_indices(csv_path):
    return pd.read_csv(csv_path).values.astype(
        np.int64
    ).T


def reshape_vertex_to_surface(x, triangle_indices, offset=0):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.ndim == 3:
        x = x.squeeze(-1)
    triangle_indices_tensor = torch.from_numpy(triangle_indices).long()
    if (triangle_indices_tensor >= 163842).sum():
        offset = -163842
    triangle_indices_tensor = triangle_indices_tensor + offset
    x_tri = x[:, triangle_indices_tensor].mean(dim=2)
    x_tri = x_tri.unsqueeze(-1)
    return x_tri.numpy()


class SVViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SiT(**CONFIG["MODEL_CONFIG"])

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, -1, 1)
        return self.backbone(x).view(-1)


class EarlyStopping:
    def __init__(self, patience=CONFIG.get("PATIENCE", 10), verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss전
        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model.state_dict()
            self.counter = 0


def train_sit_model(X_train, y_train, X_valid, y_valid, model_cls):
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

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    model = model_cls().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    early_stopping = EarlyStopping(patience=CONFIG.get("PATIENCE", 10), verbose=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            out = model(batch_X)
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
                out = model(batch_X)
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
            model.load_state_dict(early_stopping.best_model)
            break

    return model


def test_classifier_sit(X, y, Model, batch_size=48):
    # Early check for empty y or y_scores
    if len(y) == 0 or len(X) == 0:
        print(
            "⚠️ Warning: No samples provided for test_classifier_sit. Skipping evaluation."
        )
        return 0.0, 0.0, np.array([])
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    Model.eval()
    y_scores = []

    with torch.no_grad():
        for batch in loader:
            batch_data = batch[0].to(device)
            out = Model(batch_data)
            score = torch.sigmoid(out).view(-1).cpu().numpy()
            y_scores.extend(score)

    y_scores = np.array(y_scores)
    if len(y) == 0 or len(y_scores) == 0:
        print(
            "⚠️ Warning: No samples provided for test_classifier_sit. Skipping evaluation."
        )
        return 0.0, 0.0, np.array([])
    # Try-except for AUROC/AUPRC in case of only one class in y
    try:
        auroc = roc_auc_score(y, y_scores)
    except ValueError:
        print("⚠️ AUROC calculation failed due to only one class present in y.")
        auroc = 0.0
    try:
        auprc = average_precision_score(y, y_scores)
    except ValueError:
        print("⚠️ AUPRC calculation failed due to only one class present in y.")
        auprc = 0.0
    return auroc, auprc, y_scores


# New process_candidate with new split structure and metrics/fold saving
def generate_fold_assignment(labels, cv_folds, seed=42):
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_assignment = np.zeros(len(labels), dtype=int)

    # Use stratification for class balance
    for fold_idx, (_, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        fold_assignment[val_idx] = fold_idx

    return fold_assignment


def process_candidate(dataset, candidate):
    from collections import defaultdict
    from copy import deepcopy
    from sklearn.model_selection import StratifiedKFold, train_test_split

    # Load data and labels
    data_concat, labels = load_and_preprocess_surface_data(dataset, candidate)
    labels = np.asarray(labels)
    candidate_dir = os.path.join(base_path, dataset, candidate)
    os.makedirs(candidate_dir, exist_ok=True)
    dataset_dir = os.path.join(base_path, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    cv_folds = CONFIG["TRAIN_SPLIT"]["cv_folds"]
    # Validate cv_folds value
    if not isinstance(cv_folds, int) or cv_folds <= 1:
        raise ValueError(
            f"Invalid cv_folds={cv_folds} in config. Must be an integer > 1."
        )
    valid_size = CONFIG["TRAIN_SPLIT"]["valid_size"]
    seed = CONFIG.get("SEED", 42)

    # Always generate fold_assignment.npy using the new function
    fold_file = os.path.join(dataset_dir, "fold_assignment.npy")
    fold_assignment = generate_fold_assignment(labels, cv_folds, seed=CONFIG["SEED"])
    np.save(fold_file, fold_assignment)

    # For summary/metrics
    fold_metrics = []
    pred_all = np.full(len(labels), np.nan)
    # For test predictions, will collect per fold and average at the end
    test_preds_accum = []

    # For accumulating ROC/PR data for all folds, per split
    plot_data = {"train": [], "valid": [], "test": []}
    # To accumulate test predictions for each fold for ensemble (mean)
    test_pred_folds = []
    # For each fold (cross-validation)
    for fold in range(cv_folds):
        # For this fold: test is fold_assignment == fold
        test_idx = np.where(fold_assignment == fold)[0]
        trainvalid_idx = np.where(fold_assignment != fold)[0]
        # Split trainvalid_idx into train and valid, stratified
        X_trainval = data_concat[trainvalid_idx]
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
        # For logging
        wandb.init(
            project=f"hypo2_ad_classification-{args.config}",
            name=f"{dataset}-{candidate}-fold{fold}",
            config={
                "epochs": CONFIG["EPOCHS"],
                "batch_size": CONFIG["DATALOADER_CONFIG"]["batch_size"],
                "fold": fold,
                "dataset": dataset,
                "candidate": candidate,
            },
        )

        model = train_sit_model(X_train, y_train, X_valid, y_valid, SVViT)

        # Train/Valid predictions
        auroc_train, auprc_train, y_scores_train = test_classifier_sit(
            X_train,
            y_train,
            model,
            batch_size=CONFIG["DATALOADER_CONFIG"]["batch_size"],
        )
        auroc_valid, auprc_valid, y_scores_valid = test_classifier_sit(
            X_valid,
            y_valid,
            model,
            batch_size=CONFIG["DATALOADER_CONFIG"]["batch_size"],
        )
        # Test predictions (current fold's test set)
        auroc_test, auprc_test, y_scores_test = test_classifier_sit(
            X_test,
            y_test,
            model,
            batch_size=CONFIG["DATALOADER_CONFIG"]["batch_size"],
        )
        # Save predictions for valid (for SiT_predictions.csv)
        pred_all[valid_idx] = y_scores_valid
        # For test set, accumulate predictions for ensemble
        test_pred_folds.append((test_idx, y_scores_test))

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
        model_save_path = os.path.join(candidate_dir, f"model_fold{fold}.pt")
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
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
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
        # PR 대각선 기준선 (1-x)
        axes[1].plot([0, 1], [1, 0], color="gray", linestyle="--", label="Random")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title(f"Fold {fold} PR curve")
        axes[1].legend(loc="lower left")
        plt.tight_layout()
        fold_graph_file = os.path.join(candidate_dir, f"SiT_fold{fold}_AUROC_AUPRC.png")
        plt.savefig(fold_graph_file)
        plt.close()
        print(f"Saved fold {fold} ROC/PR 통합 그림: {fold_graph_file}")

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
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # ROC curve
        for i, (fpr, tpr, recall, precision, auroc, auprc, fold) in enumerate(
            plot_data[split]
        ):
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
        for i, (fpr, tpr, recall, precision, auroc, auprc, fold) in enumerate(
            plot_data[split]
        ):
            axes[1].plot(
                recall,
                precision,
                lw=2,
                label=f"Fold {fold} AUPRC = {auprc:.3f}",
            )
        axes[1].plot([0, 1], [1, 0], color="gray", linestyle="--", label="Random")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title(f"All Folds {split.capitalize()} PR curve")
        axes[1].legend(loc="lower left", fontsize=8)
        plt.tight_layout()
        allfold_file = os.path.join(candidate_dir, f"SiT_allfold_{split}.png")
        plt.savefig(allfold_file)
        plt.close()
        print(f"Saved all folds {split} ROC/PR figure: {allfold_file}")

    # Save fold_metrics.csv
    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_csv = os.path.join(candidate_dir, "fold_metrics.csv")
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
    metric_summary_csv = os.path.join(candidate_dir, "metric_summary.csv")
    metric_summary_df.to_csv(metric_summary_csv, index=False)
    print(f"Saved metric summary to {metric_summary_csv}")

    # Save SiT_predictions.csv (for all samples: valid predictions for train, test predictions for test)
    sit_pred_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(labels)),
            "true_label": labels,
            "predicted_prob": pred_all,
            "predicted_label": (pred_all >= 0.5).astype(float),
            "fold": fold_assignment,
        }
    )
    # For each fold, fill in test predictions for that fold's test indices
    for test_idx, y_scores_test in test_pred_folds:
        sit_pred_df.loc[test_idx, "predicted_prob"] = y_scores_test
        sit_pred_df.loc[test_idx, "predicted_label"] = (
            np.array(y_scores_test) >= 0.5
        ).astype(float)
    sit_pred_df.to_csv(
        os.path.join(candidate_dir, "SiT_predictions.csv"),
        index=False,
    )
    print(f"Saved predictions to {os.path.join(candidate_dir, 'SiT_predictions.csv')}")

    # Save candidate_results.csv (mean test AUROC/AUPRC)
    result_df = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "candidate": candidate,
                "test_AUROC_mean": metric_summary["test_AUROC_mean"],
                "test_AUROC_std": metric_summary["test_AUROC_std"],
                "test_AUPRC_mean": metric_summary["test_AUPRC_mean"],
                "test_AUPRC_std": metric_summary["test_AUPRC_std"],
            }
        ]
    )
    result_df.to_csv(os.path.join(candidate_dir, "candidate_results.csv"), index=False)
    print(f"Saved result summary to {candidate_dir}/candidate_results.csv")


def main():
    for dataset in datasets:
        for candidate in candidates:
            if os.path.exists(os.path.join(base_path, dataset, candidate)):
                print(f"{dataset} {candidate} exist, skip.")
                continue
            print(f"Processing candidate {candidate} for dataset {dataset}...")
            process_candidate(dataset, candidate)

    overall_results = []
    for dataset in datasets:
        for candidate in candidates:
            candidate_dir = os.path.join(base_path, dataset, candidate)
            result_file = os.path.join(candidate_dir, "candidate_results.csv")
            if os.path.exists(result_file):
                df = pd.read_csv(result_file)
                # Ensure candidate column is zero-padded string
                df["candidate"] = df["candidate"].astype(str).str.zfill(6)
                overall_results.append(df)
    if overall_results:
        overall_df = pd.concat(overall_results, axis=0, ignore_index=True)
        overall_csv = os.path.join(base_path, "overall_results.csv")
        overall_df.to_csv(overall_csv, index=False)
        print(f"\nOverall results saved to {overall_csv}")


if __name__ == "__main__":
    main()
