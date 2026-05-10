import numpy as np
import pandas as pd
import nibabel as nib
import os
import yaml
import argparse
import pickle
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures, LabelBinarizer, MinMaxScaler

from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cross_decomposition import PLSRegression
import re

import gc

# Set seed
SEED = 42
np.random.seed(SEED)

# data - Split - AD,CN - subjects


def load_config(yaml_path):
    with open(yaml_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


parser = argparse.ArgumentParser(
    description="Train the model with parameters from a yaml file."
)
parser.add_argument("--config", required=True, help="Path to the yaml config file")
parser.add_argument("--mssm_cand", required=True, help="Candidate usage")
parser.add_argument("--split", required=True, help="Split")
args = parser.parse_args()
config = load_config(args.config)
CORRECTION = config.get("CORRECTION", False)
SAVE = config.get("SAVE", False)
MSSM_CAND = str(args.mssm_cand)
SPLITS = [str(args.split)]

WEIGHTS_DIR = config["WEIGHTS_DIR"]
DATA_DIR = config["DATA_DIR"]
# NTH (polynomial degree index) — default 1 if missing
NTH = int(config.get("NTH", 1))
INFO_DIR = config["INFO_DIR"]
INFO_PATH = config["INFO_PATH"]
DATASET_PATH = config["DATASET_PATH"]
WEIGHTS_PATHS = config["WEIGHTS_PATHS"]
DATASET_FILE = config["DATASET_FILE"]
CORRECTION_FILE = config["CORRECTION_FILE"]
N_FEATURES = config["N_FEATURES"]
N_VERTEX = config["N_VERTEX"]
ADD = float(config["ADD"])
SCALER_THRESHOLD = config["SCALER_THRESHOLD"]
MAX_WORKERS = config["MAX_WORKERS"]
FEATURE_EXTRACTION_METHOD = config["FEATURE_EXTRACTION_METHOD"]
REDUCTION_FILE = config["REDUCTION_FILE"]
FEATURE_EXTRACTION_DIR = config["FEATURE_EXTRACTION_DIR"]
RESULT_TYPE = config["RESULT_TYPE"]
THICK_PREFIX = config["THICK_PREFIX"]
GWR_PREFIX = config["GWR_PREFIX"]
CAND_PREFIX = config["CAND_PREFIX"]
STAT_FILE = config["STAT_FILE"]
CT_TYPES = config["CT_TYPES"]
GWR_TYPES = config["GWR_TYPES"]
CAND_TYPES = config["CAND_TYPES"]
CORRECTION_DIMENSIONS = np.array(config["CORRECTION_DIMENSIONS"], dtype=int)
FWHM = int(config["FWHM"])
RESULTS_ROOT = config["RESULTS_ROOT"]
RESULTS_DIR  = os.path.join(RESULTS_ROOT, str(FWHM))
MAP_DIR = config.get("MAP_DIR", os.path.join(RESULTS_ROOT, "map"))
GROUPS = config["GROUPS"]
HEMIS = config["HEMIS"]
USAGE = np.array([MSSM_CAND[0]] * 8 + list(MSSM_CAND)[1:], dtype=int).astype(bool)
MEDIALWALL_MASKS = config["MEDIALWALL_MASKS"]
AFFINES = config.get("AFFINES", [f"fsaverage/surf/{hemi}.orig.avg.area.mgh" for hemi in HEMIS])
FREESURFER_HOME = os.environ.get("FREESURFER_HOME")
SUBJECTS_DIR = os.path.join(FREESURFER_HOME, "subjects")

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data


def save_pkl(results, file_name):
    for split in SPLITS:
        # DATASET_FILE은 공용: MAP_DIR/split
        if file_name == DATASET_FILE:
            os.makedirs(os.path.join(MAP_DIR, split), exist_ok=True)
            pth = os.path.join(MAP_DIR, split, file_name)
            if os.path.exists(pth):
                continue
        else:
            # correction/reduction 등 cand 종속: MAP_DIR/split/cand
            os.makedirs(os.path.join(MAP_DIR, split, MSSM_CAND), exist_ok=True)
            pth = os.path.join(MAP_DIR, split, MSSM_CAND, file_name)

        with open(pth, "wb") as pickle_file:
            pickle.dump(results[split], pickle_file)


def load_masks():
    medialwall = {
        k: nib.freesurfer.io.read_label(os.path.join(SUBJECTS_DIR, v))
        for k, v in zip(list("LR"), MEDIALWALL_MASKS)
    }
    masks = np.ones(N_VERTEX, dtype=bool)
    masks[medialwall["L"]] = False
    masks[medialwall["R"] + N_VERTEX // 2] = False
    return masks

def load_affines():
    def rule_affine(n_hemi_vertices: int) -> np.ndarray:
        half = n_hemi_vertices / 2.0
        return np.array(
            [[-1.0, 0.0, 0.0, half],   # ← 81921.0000
             [ 0.0, 0.0, 1.0,  -0.5],
             [ 0.0,-1.0, 0.0,   0.5],
             [ 0.0, 0.0, 0.0,   1.0]], dtype=float
        )

    n_hemi = N_VERTEX // 2
    affines = {}
    for hemi, relpath in zip(HEMIS, AFFINES):
        p = os.path.join(SUBJECTS_DIR, relpath)
        try:
            img = nib.load(p)
            A = getattr(img, "affine", None)
            if A is None or A.shape != (4,4):
                raise ValueError("Invalid affine")
            affines[hemi] = A
        except Exception as e:
            print(f"[WARN] affine load failed for {p}: {e}; fallback to tkr rule.")
            affines[hemi] = rule_affine(n_hemi)
    return affines

def prepare_data(data_paths, df):
    dataset = {}
    nib_loaders = [nib.load] * 16 + [nib.freesurfer.io.read_morph_data] * 10

    for split in SPLITS:
        dataset[split] = {"data": [], "subjects": [], "group": [], "age": [], "group_id": []}

        # df['Split'] 값이 현재 split과 일치하는 것만 필터링
        split_df = df[df["Split"] == split]

        for group in GROUPS:
            group_df = split_df[split_df["Group"] == group]
            subjects = group_df["Subject"].tolist()
            ages = group_df["Age"].tolist()

            dataset[split]["subjects"] += subjects
            dataset[split]["age"] += ages
            dataset[split]["group"] += [group] * len(subjects)

            lb = LabelBinarizer()
            lb.fit(GROUPS)
            group_ids = (-lb.transform([group] * len(subjects)) + 1).flatten().tolist()
            dataset[split]["group_id"] += group_ids

            for subject in subjects:
                datum = []
                subject_dir = os.path.join(DATA_DIR, split, group, subject)
                for data_path, nib_loader in zip(data_paths, nib_loaders):
                    full_path = os.path.join(subject_dir, data_path)
                    fs_data = nib_loader(full_path)
                    datum.append(
                        fs_data.flatten()
                        if isinstance(fs_data, np.ndarray)
                        else fs_data.get_fdata().flatten()
                    )
                datum = np.stack(datum).reshape(-1, N_VERTEX)
                dataset[split]["data"].append(datum)

        dataset[split]["group"] = np.array(dataset[split]["group"])
        dataset[split]["data"] = np.stack(
            dataset[split]["data"]
        )  # (n_subjects, n_features, N_VERTEX)
        dataset[split]["age"] = np.array(dataset[split]["age"], dtype=np.float32)
        dataset[split]["group_id"] = np.array(dataset[split]["group_id"], dtype=np.int32)
        dataset[split]["subjects"] = np.array(dataset[split]["subjects"])

    return dataset

def generate_dataset():  # GWC - CT - Sulc - Jacobian - Curv - LGI
    candidate_types = [
        f"{hemi}h.fsaverage.{cand_type}" for cand_type in CAND_TYPES for hemi in "lr"
    ]

    data_paths = [os.path.join(f"{GWR_PREFIX}", gwr_type) for gwr_type in GWR_TYPES]
    data_paths += [
        os.path.join(f"{CAND_PREFIX}", candidate_type)
        for candidate_type in candidate_types
    ]
    data_paths = (
        data_paths[:-4]
        + [os.path.join(f"{THICK_PREFIX}", ct_type) for ct_type in CT_TYPES]
        + data_paths[-4:]
    )

    info_df = pd.read_csv(INFO_PATH)

    dataset = prepare_data(data_paths, info_df)

    return dataset


class ModelFactory:
    def __init__(self, degree):
        self.degree = degree

    def linear_svr(self):
        return SVR(kernel="linear")

    def poly_svr(self):
        return SVR(kernel="poly", degree=self.degree)

    def linear_regression(self):
        return LinearRegression()

    def polynomial_regression(self):
        return make_pipeline(PolynomialFeatures(degree=self.degree), LinearRegression())


def process_age_correction(
    ad_score, cn_score, ad_ages, cn_ages, model_builder, weight_params=None
):
    """
    모델을 사용하여 Age Correction을 수행.
    """
    model = model_builder()

    if weight_params is not None:
        # LinearRegression
        if isinstance(model, LinearRegression):
            model.coef_ = np.array(weight_params["coefficients"])
            model.intercept_ = weight_params["intercept"]

        # SVR
        elif isinstance(model, SVR):
            model.support_ = np.array(weight_params["support_vectors"])
            model.dual_coef_ = np.array(weight_params["dual_coefficients"])
            model.intercept_ = weight_params["intercept"]

        # Pipeline (PolynomialFeatures + LinearRegression)
        elif isinstance(model, Pipeline):
            linear_model = model.named_steps["linearregression"]
            linear_model.coef_ = np.array(weight_params["coefficients"])
            linear_model.intercept_ = weight_params["intercept"]

            # PolynomialFeatures
            poly_params = weight_params["polynomialfeatures"]["params"]
            poly_features = PolynomialFeatures(**poly_params)

            n_input_features = weight_params["polynomialfeatures"]["n_input_features_"]
            dummy_data = np.zeros((1, n_input_features))
            poly_features.fit(dummy_data)

            model = Pipeline(
                [
                    ("polynomialfeatures", poly_features),
                    ("linearregression", linear_model),
                ]
            )

    else:
        cn_Y = cn_score
        model.fit(cn_ages.reshape(-1, 1), cn_Y)

    ad_predictions = model.predict(ad_ages.reshape(-1, 1))
    cn_predictions = model.predict(cn_ages.reshape(-1, 1))

    ad_residuals = ad_score - ad_predictions.flatten()
    cn_residuals = cn_score - cn_predictions.flatten()

    return ad_residuals, cn_residuals


def process_feature_chunk(
    feature_idx, X_chunk, ad_ages, cn_ages, model_builder, weights=None
):
    n_subj, vertices = X_chunk.shape
    ad_res = np.empty((len(ad_ages), vertices))
    cn_res = np.empty((len(cn_ages), vertices))

    """if not weights:
        weight_maps = []"""

    for j in range(vertices):
        ad_residuals, cn_residuals = process_age_correction(
            X_chunk[: len(ad_ages), j],
            X_chunk[len(ad_ages) :, j],
            ad_ages,
            cn_ages,
            model_builder,
            weights[j] if weights is not None else None,
        )
        ad_res[:, j] = ad_residuals
        cn_res[:, j] = cn_residuals

    return feature_idx, ad_res, cn_res


def apply_age_correction_parallel(
    X, ages, n_ad, model_builder, max_workers=8, weights=None
):
    n_subj, _, vertices = X.shape
    ad_res_total = X[:n_ad, 0, :]
    cn_res_total = X[n_ad:, 0, :]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        dim = CORRECTION_DIMENSIONS[0]
        weight = weights[dim] if weights is not None else None

        future = executor.submit(
            process_feature_chunk,
            0,
            X[:, 0, :],
            ages[:n_ad],
            ages[n_ad:],
            model_builder[0],
            weight,
        )
        feature_idx, ad_res, cn_res = future.result()
        ad_res_total = ad_res
        cn_res_total = cn_res

    return np.concatenate((ad_res_total, cn_res_total), axis=0).reshape(
        (n_subj, 1, vertices)
    )


def generate_correction(data_source, masks, weights):
    print("age correction start")

    # Early return if CORRECTION is not enabled
    if not CORRECTION:
        print("Skipping age correction as per configuration.")
        correction = {
            split: {
                "data": (data_source[split]["data"]),
                "subjects": np.array(data_source[split]["subjects"]),
                "group_id": data_source[split]["group_id"],
                "age": data_source[split]["age"]
            }
            for split in SPLITS
        }
        return correction

    models_factory = {
        1: ModelFactory(1).polynomial_regression,
        2: ModelFactory(2).polynomial_regression,
        3: ModelFactory(3).polynomial_regression,
    }

    correction = dict()
    lb = LabelBinarizer()

    model_builder = [models_factory[CORRECTION_DIMENSIONS[0]]]

    for split in SPLITS:
        # reduction 출력 (n_subj, vertices)
        X = data_source[split]["data"]  # shape: (n_subj, vertices)
        subjects = data_source[split]["subjects"]
        age_vals = data_source[split]["age"]
        group_id_orig = data_source[split]["group_id"].flatten()  # 1=AD, 0=CN

        n_ad = int(np.sum(group_id_orig == 1))

        if not (np.all(group_id_orig[:n_ad] == 1) and np.all(group_id_orig[n_ad:] == 0)):
            raise ValueError(
                "Dataset order mismatch: Expected AD-first then CN order. "
                "Check GROUPS order or prepare_data() sorting."
            )

        n_subj, vertices = X.shape
        n_features = 1
        X = X.reshape(n_subj, 1, vertices)

        target = (
            MinMaxScaler()
            .fit_transform(np.array(age_vals).reshape(-1, 1).astype(np.float64))
            .astype(np.float32)
            + ADD
        )

        corrected = apply_age_correction_parallel(
            X.astype(np.float32).reshape(n_subj, 1, vertices),
            target,
            n_ad,
            model_builder,
            max_workers=MAX_WORKERS,
            weights=weights,
        )

        correction[split] = {
            "data": corrected.squeeze(1),                 # (n_subj, vertices)
            "subjects": np.array(subjects),
            "group_id": group_id_orig.reshape(-1, 1),     # (n_subj, 1)
            "age": np.array(age_vals, dtype=np.float32)
        }

    return correction

def shuffle_data(data, targets, seed=None):
    """
    shuffle data and target, preserving order
    """
    np.random.seed(seed)
    indices = np.arange(len(targets))
    np.random.shuffle(indices)
    return data[indices], targets[indices], indices


def restore_order(data, indices):
    """
    restore order
    """
    original_order = np.argsort(indices)
    return data[original_order]


def select_model(model_type, n_components=1):
    if model_type == "PLS":
        return PLSRegression(n_components=n_components, scale=True)
    else:
        raise ValueError("Unsupported model type")


def apply_pls_cpu(vertex_index, X, y, feature_extraction_method):
    try:
        model = select_model(feature_extraction_method)
        if vertex_index >= X.shape[2]:
            raise IndexError(
                f"Vertex index {vertex_index} is out of bounds for array with shape {X.shape}"
            )
        X_onevertex = X[:, :, vertex_index]
        model.fit(X_onevertex, y)
        return np.squeeze(model.x_scores_), model
    except Exception as e:
        raise ValueError(f"PLS ERROR at vertex {vertex_index}: {str(e)}")


def process_chunk(
    X_chunk, y_chunk, feature_extraction_method=FEATURE_EXTRACTION_METHOD
):
    """
    PLS를 특정 데이터 청크에 대해 처리.
    """
    n_subj, n_features, n_vertices = X_chunk.shape
    results_x_scores = np.empty((n_subj, n_vertices), dtype=np.float32)

    models = [None] * n_vertices if SAVE else None

    for vertex_index in range(n_vertices):
        embedding, model = apply_pls_cpu(
            vertex_index, X_chunk, y_chunk, feature_extraction_method
        )
        if embedding is not None:
            results_x_scores[:, vertex_index] = embedding
            if SAVE:
                models[vertex_index] = model

    return results_x_scores, (np.array(models) if SAVE else None)


def parallel_pls_regression(X, y, chunks):
    num_chunks = len(chunks)

    results_ordered = [None] * num_chunks
    models_ordered = ([None] * num_chunks) if SAVE else None

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process_chunk, chunks[i], y, FEATURE_EXTRACTION_METHOD): i
            for i in range(num_chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            chunk_results, chunk_models = future.result()
            results_ordered[idx] = chunk_results
            if SAVE:
                models_ordered[idx] = chunk_models

    results = np.concatenate(results_ordered, axis=-1)
    models = (np.concatenate(models_ordered, axis=0) if SAVE else None)
    return results, models


def generate_reduction(source_data, masks):
    """
    Generate PLS reduction.
    input is original dataset; selects USAGE, then runs PLS
    """
    reduction = {}

    for split in SPLITS:
        reduction[split] = {}

        # meta
        if "group_id" in source_data[split]:
            group_id = source_data[split]["group_id"]  # shape: (n_subj, 1) or (n_subj,)
        elif "group" in source_data[split]:
            group = source_data[split]["group"]
            group_id = -LabelBinarizer().fit_transform(group) + 1
        else:
            raise KeyError(f"source_data[{split}]에 'group_id' 또는 'group' 키가 없습니다.")

        age_vals = source_data[split]["age"] if "age" in source_data[split] else None
        subjects = source_data[split]["subjects"]

        # features (USAGE select)
        data = source_data[split]["data"][:, USAGE, :]  # (n_subj, n_features, vertices)
        n_subj, n_features, _ = data.shape

        reduction[split]["subjects"] = subjects
        reduction[split]["group_id"] = group_id
        if age_vals is not None:
            reduction[split]["age"] = age_vals
        
        if np.sum(USAGE) == 1:
            results = data.reshape(n_subj, -1)
            models = None
        else:
            y_group = np.array(group_id).reshape(-1)

            shuffled_data, shuffled_targets, shuffle_indices = shuffle_data(
                data, y_group, seed=42
            )
            chunks = np.array_split(shuffled_data, MAX_WORKERS, axis=2)
            results, models = parallel_pls_regression(
                shuffled_data, shuffled_targets, chunks
            )
            results = restore_order(results, shuffle_indices)

        reduction[split]["data"] = results
        reduction[split]["models"] = models
        reduction[split]["data"][:, ~masks] = 0

    return reduction


def save_freesurfer_data(reduction):
    
    """affine = np.array(
        [
            [-1.0, 0.0, 0.0, 81921.0],
            [0.0, 0.0, 1.0, -0.5],
            [0.0, -1.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )"""
    affines = load_affines()
    for split in SPLITS:
        save_dir = os.path.join(MAP_DIR, split, MSSM_CAND)
        os.makedirs(save_dir, exist_ok=True)
        for h, hemi in enumerate(HEMIS):
            affine = affines[hemi]
            nib.save(
                nib.freesurfer.MGHImage(
                    reduction[split]["data"][
                        :, h * N_VERTEX // 2 : (h + 1) * N_VERTEX // 2
                    ]
                    .reshape(N_VERTEX // 2, 1, 1, len(reduction[split]["subjects"]))
                    .astype(np.float32),
                    affine,
                ),
                os.path.join(save_dir, f"{hemi}_concat.mgh"),
            )
            
            map_dir = os.path.join(save_dir, hemi)
            os.makedirs(map_dir, exist_ok=True)
            for s, subject in enumerate(reduction[split]["subjects"]):
                nib.freesurfer.io.write_morph_data(
                    os.path.join(map_dir, f"{hemi}.{subject}.mssm"),
                    reduction[split]["data"][
                        s, h * N_VERTEX // 2 : (h + 1) * N_VERTEX // 2
                    ]
                    .flatten()
                    .astype(np.float32),
                )

def write_fsgd(df):
    for split in SPLITS:
        split_df = df[df["Split"] == split]
        split_dir = os.path.join(MAP_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        with open(os.path.join(split_dir, f"{split}.fsgd"), "w") as f:
            f.write(f"GroupDescriptorFile 1\nTitle {split}\n\n")
            for g in GROUPS:
                f.write(f"Class {g}\n")
            for group in GROUPS:
                group_df = split_df[split_df["Group"] == group]
                subjects = group_df["Subject"].tolist()
                f.write(f"\n# {group}\n")
                for subject in subjects:
                    f.write(f"Input {subject} {group}\n")

def main():
    info_df = pd.read_csv(INFO_PATH)
    write_fsgd(info_df)
    
    split = SPLITS[0]

    masks = load_masks()

    if DATASET_PATH:
        dataset = load_pkl(DATASET_PATH)
    else:
        dataset = generate_dataset()
        if SAVE:
            save_pkl(dataset, DATASET_FILE)

    weights = {e: None for e in range(1, 4)}

    reduction = generate_reduction(dataset, masks)
    del dataset; gc.collect()
    if WEIGHTS_DIR:
        weight_path = os.path.join(WEIGHTS_DIR, f"polynomial-{NTH}-post-{MSSM_CAND}.pkl")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"weight not found: {weight_path}")
        weights = {NTH: load_pkl(weight_path)}

    correction = generate_correction(reduction, masks, weights)

    if SAVE:
        save_pkl(correction, CORRECTION_FILE)
        save_pkl(reduction, REDUCTION_FILE)

    save_freesurfer_data(correction)

if __name__ == "__main__":
    main()
