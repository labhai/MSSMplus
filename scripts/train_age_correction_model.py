from datetime import datetime
import numpy as np
import pandas as pd
import nibabel as nib
import os
import yaml
import argparse
import glob
import pickle
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, LabelBinarizer, MinMaxScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
import gc
import warnings

# Set seed
SEED = 42
np.random.seed(SEED)

# data - Split - AD,CN - subjects

def load_config(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

parser = argparse.ArgumentParser(description='Train the model with parameters from a yaml file.')
parser.add_argument('--config', required=True, help='Path to the yaml config file')
args = parser.parse_args()
config = load_config(args.config)

DATA_DIR = config['DATA_DIR']
INFO_PATH = config['INFO_PATH']
DATASET_PATH = config['DATASET_PATH']
WEIGHTS_PATH = config['WEIGHTS_PATH']
CORRECTION_MODEL = config['CORRECTION_MODEL']
DATASET_FILE = config['DATASET_FILE']
CORRECTION_FILE = config['CORRECTION_FILE']
WEIGHTS_FILE = config['WEIGHTS_FILE']
N_FEATURES = config['N_FEATURES']
N_VERTEX = config['N_VERTEX']
NTH = config['NTH']
ADD = float(config['ADD'])
MAX_WORKERS = config['MAX_WORKERS']
THICK_PREFIX = config['THICK_PREFIX']
GWR_PREFIX = config['GWR_PREFIX']
CAND_PREFIX = config['CAND_PREFIX']
CT_TYPES = config['CT_TYPES']
GWR_TYPES = config['GWR_TYPES']
CAND_TYPES = config['CAND_TYPES']
MEDIALWALL_MASKS = config['MEDIALWALL_MASKS']
FREESURFER_HOME = os.environ.get('FREESURFER_HOME')

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

def save_pkl(results, file_path):
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(results, pickle_file)

def load_masks():
    medialwall = {k:nib.freesurfer.io.read_label(os.path.join(FREESURFER_HOME, 'subjects', v)) for k,v in zip(list('LR'), MEDIALWALL_MASKS)}
    masks = np.ones(N_VERTEX, dtype=bool)
    masks[medialwall['L']] = False
    masks[medialwall['R']+N_VERTEX//2] = False
    return masks

def prepare_data(data_paths, df):
    dataset = {}
    nib_loaders = [nib.load]*16 + [nib.freesurfer.io.read_morph_data]*10
    dataset = {}
    dataset['data'] = []
    dataset['target'] = []   # age
    dataset['group'] = []    # 'AD'/'CN'
    dataset['group_id'] = [] # 1=AD, 0=CN
    dataset['subjects'] = []
    subpaths = glob.glob(os.path.join(DATA_DIR, '*/'))
    subjects = sorted([os.path.basename(os.path.normpath(subpath)) for subpath in subpaths])
    dataset['subjects'] += subjects

    # align metadata rows to selected subjects in the same order
    meta = df[df['Subject'].isin(subjects)].copy()
    meta = meta.set_index('Subject').loc[subjects]

    ages = meta['Age'].values.astype(np.float32)
    groups = meta['Group'].values
    group_id = (groups == 'AD').astype(np.int32)  # AD=1, CN=0

    dataset['target'] += list(ages)
    dataset['group'] += list(groups)
    dataset['group_id'] += list(group_id)

    for subject in subjects:
        datum = []
        for data_path, nib_loader in zip(data_paths, nib_loaders):
            fs_data = nib_loader(os.path.join(DATA_DIR, subject, data_path))
            datum.append(fs_data.flatten() if type(fs_data) == np.ndarray else fs_data.get_fdata().flatten())
        datum = np.stack(datum).reshape(-1,N_VERTEX)
        dataset['data'].append(datum)
    dataset['data'] = np.stack(dataset['data']) # n_subjects, n_features, N_VERTEX
    dataset['target'] = np.array(dataset['target'], dtype=np.float32)
    dataset['group'] = np.array(dataset['group'])
    dataset['group_id'] = np.array(dataset['group_id'], dtype=np.int32)
    dataset['subjects'] = np.array(dataset['subjects'])
    return dataset

def generate_dataset():
    candidate_types = [f"{hemi}h.fsaverage.{cand_type}" for cand_type in CAND_TYPES for hemi in "lr"]

    data_paths = [os.path.join(f"{GWR_PREFIX}", gwr_type) for gwr_type in GWR_TYPES]
    data_paths += [os.path.join(f"{CAND_PREFIX}", candidate_type) for candidate_type in candidate_types]
    data_paths = data_paths[:-4] + [os.path.join(f"{THICK_PREFIX}", ct_type) for ct_type in CT_TYPES] + data_paths[-4:]

    info_df = pd.read_csv(INFO_PATH)
    
    dataset = prepare_data(data_paths, info_df)

    return dataset


def featurewise_minmax_scale(X):
    n_subj, n_features, n_vertex = X.shape
    X_scaled = np.empty_like(X, dtype=np.float32)
    scalers = []
    for f in range(n_features):
        scaler = MinMaxScaler()
        Xf = X[:, f, :].astype(np.float64)
        Xf_scaled = scaler.fit_transform(Xf)
        X_scaled[:, f, :] = Xf_scaled.astype(np.float32)
        scalers.append(scaler)
    return X_scaled, scalers




# USAGE helper mirroring age_correction_pls.py
def usage_from_cand(cand_bits):
    return np.array([cand_bits[0]] * 8 + list(cand_bits)[1:], dtype=int).astype(bool)


def train_one_candidate(dataset, cand_bits):
    usage_mask = np.array([cand_bits[0]] * 8 + list(cand_bits)[1:], dtype=int).astype(bool)

    X_full = dataset['data']
    X = X_full[:, usage_mask, :]  # (n_subj, n_sel_features, n_vertex)
    y_age = dataset['target'].astype(np.float32)
    y_group = dataset['group_id'].astype(np.float32).reshape(-1)  # AD=1, CN=0

    n_subj, n_features, n_vertex = X.shape
    # Direct PLS scaling (no MinMax scaling; use scale=True in PLSRegression)
    pls_results = np.zeros((n_subj, n_vertex), dtype=np.float32)
    for v in range(n_vertex):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*y residual is constant at iteration.*")
            model = PLSRegression(n_components=1, scale=True)
            model.fit(X[:, :, v], y_group)
        pls_results[:, v] = model.x_scores_.squeeze()

    # 3) vertex-wise age correction (age_scaled → x_score)
    y_scaled = (y_age - y_age.min()) / (y_age.max() - y_age.min()) + ADD

    model_factory = ModelFactory(NTH)
    models_factory = {
        'linear_svr': model_factory.linear_svr,
        'poly_svr': model_factory.poly_svr,
        'linear': model_factory.linear_regression,
        'polynomial': model_factory.polynomial_regression
    }
    model_builder = models_factory[CORRECTION_MODEL]

    weights = np.empty(n_vertex, dtype=object)
    for v in range(n_vertex):
        m = model_builder()
        m.fit(y_scaled.reshape(-1, 1), pls_results[:, v])
        if isinstance(m, LinearRegression):
            weights[v] = {'coefficients': m.coef_, 'intercept': m.intercept_}
        elif isinstance(m, SVR):
            weights[v] = {
                'support_vectors': m.support_,
                'dual_coefficients': m.dual_coef_,
                'intercept': m.intercept_
            }
        elif isinstance(m, Pipeline):
            lin = m.named_steps['linearregression']
            poly = m.named_steps['polynomialfeatures']
            weights[v] = {
                'coefficients': lin.coef_,
                'intercept': lin.intercept_,
                'polynomialfeatures': {
                    'params': poly.get_params(),
                    'n_input_features_': 1,
                    'powers_': poly.powers_
                }
            }
    return weights


class ModelFactory:
    def __init__(self, degree):
        self.degree = degree

    def linear_svr(self):
        return SVR(kernel='linear')

    def poly_svr(self):
        return SVR(kernel='poly', degree=self.degree)

    def linear_regression(self):
        return LinearRegression()

    def polynomial_regression(self):
        return make_pipeline(PolynomialFeatures(degree=self.degree), LinearRegression())

def process_age_correction(score, ages, model_builder, weight_params=None):
    model = model_builder()
    
    Y = score
    model.fit(ages.reshape(-1, 1), Y)

     # Extract weights based on the model type
    if isinstance(model, LinearRegression):
        weights = {
            'coefficients': model.coef_,
            'intercept': model.intercept_
        }
    elif isinstance(model, SVR):
        weights = {
            'support_vectors': model.support_vectors_,
            'dual_coefficients': model.dual_coef_,
            'intercept': model.intercept_
        }
    elif isinstance(model, Pipeline):
        # For pipeline (Polynomial Regression), access the final step
        linear_model = model.named_steps['linearregression']
        poly_features = model.named_steps['polynomialfeatures']
        weights = {
            'coefficients': linear_model.coef_,
            'intercept': linear_model.intercept_,
            'polynomial_features': poly_features.get_feature_names_out()
        }
    else:
        raise ValueError("Unsupported model type for weight extraction")

    return weights

def process_feature_chunk(feature_idx, X_chunk, ages, model_builder):
    """
    각 feature에서 모든 vertex를 처리.
    가중치는 딕셔너리를 ndarray에 저장 가능하도록 처리.
    """
    n_subj, vertices = X_chunk.shape
    weight_maps = np.empty(vertices, dtype=object)

    for v in range(vertices):
        model = model_builder()
        Y = X_chunk[:, v]
        model.fit(ages.reshape(-1, 1), Y)

        if isinstance(model, LinearRegression):
            weight_maps[v] = {
                'coefficients': model.coef_,
                'intercept': model.intercept_
            }
        elif isinstance(model, SVR):
            weight_maps[v] = {
                'support_vectors': model.support_,
                'dual_coefficients': model.dual_coef_,
                'intercept': model.intercept_
            }
        elif isinstance(model, Pipeline):  # Polynomial Regression
            linear_model = model.named_steps['linearregression']
            poly_features = model.named_steps['polynomialfeatures']

            weight_maps[v] = {
                'coefficients': linear_model.coef_,
                'intercept': linear_model.intercept_,
                'polynomialfeatures': {
                    'params': poly_features.get_params(),
                    'n_input_features_': ages.reshape(-1, 1).shape[1],
                    'powers_': poly_features.powers_
                }
            }

    return feature_idx, weight_maps


def apply_age_correction_parallel(X, ages, model_builder, max_workers=8):
    n_subj, n_features, vertices = X.shape
    weight_maps = np.empty((n_features, vertices), dtype=object)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for f in range(n_features):
            futures.append(executor.submit(process_feature_chunk, f, X[:, f, :], ages, model_builder))

        for future in as_completed(futures):
            feature_idx, weight_map = future.result()
            weight_maps[feature_idx, :] = weight_map  # save weights

    return weight_maps


def train_correction_model(dataset, masks):
    print('PLS reduction + age correction start')

    model_factory = ModelFactory(NTH)
    models_factory = {
        'linear_svr': model_factory.linear_svr,
        'poly_svr': model_factory.poly_svr,
        'linear': model_factory.linear_regression,
        'polynomial': model_factory.polynomial_regression
    }
    model_builder = models_factory[CORRECTION_MODEL]

    X = dataset['data']  # (n_subj, n_features, n_vertex)
    y_age = dataset['target'].astype(np.float32)
    y_group = dataset['group_id'].astype(np.float32).reshape(-1)  # AD=1, CN=0
    n_subj, n_features, n_vertex = X.shape

    # age scaling for age-correction model
    y_scaled = (y_age - y_age.min()) / (y_age.max() - y_age.min()) + ADD

    # --- PLS (target = group) vertex-wise ---
    def apply_pls_vertex(v):
        model = PLSRegression(n_components=1)
        # Use all features at this vertex: (n_subj, n_features)
        model.fit(X[:, :, v], y_group)
        return model.x_scores_.squeeze(), model

    pls_results = np.zeros((n_subj, n_vertex), dtype=np.float32)
    pls_models = [None] * n_vertex

    for v in range(n_vertex):
        x_scores, model = apply_pls_vertex(v)
        pls_results[:, v] = x_scores
        pls_models[v] = model

    save_pkl(pls_results, os.path.join(WEIGHTS_FILE, "post-regression.pkl"))

    # --- vertex-wise age correction model on x_scores ---
    weights = np.empty(n_vertex, dtype=object)
    for v in range(n_vertex):
        model = model_builder()
        model.fit(y_scaled.reshape(-1, 1), pls_results[:, v])
        if isinstance(model, LinearRegression):
            weights[v] = {'coefficients': model.coef_, 'intercept': model.intercept_}
        elif isinstance(model, SVR):
            weights[v] = {
                'support_vectors': model.support_,
                'dual_coefficients': model.dual_coef_,
                'intercept': model.intercept_
            }
        elif isinstance(model, Pipeline):
            lin = model.named_steps['linearregression']
            poly = model.named_steps['polynomialfeatures']
            weights[v] = {
                'coefficients': lin.coef_,
                'intercept': lin.intercept_,
                'polynomialfeatures': {
                    'params': poly.get_params(),
                    'n_input_features_': y_scaled.reshape(-1, 1).shape[1],
                    'powers_': poly.powers_
                }
            }

    print('Train end.')
    return weights

def main():
    masks = load_masks()
    if DATASET_PATH:
        dataset = load_pkl(DATASET_PATH)
    else:
        dataset = generate_dataset()
        save_pkl(dataset, DATASET_FILE)

    mssm_cands = config.get('MSSM_CANDIDATES', [])
    if not mssm_cands:
        raise ValueError("MSSM_CANDIDATES is empty. Check yaml.")

    for cand in mssm_cands:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}][TRAIN][POST] candidate={cand} — feature-wise scaling + group-PLS + age-correction")
        weights = train_one_candidate(dataset, cand)
        if weights is None:
            continue
        out_path = os.path.join(WEIGHTS_FILE, f"{CORRECTION_MODEL}-{NTH}-post-{cand}.pkl")
        save_pkl(weights, out_path)
        gc.collect()
    print("All candidates done.")

if __name__ == "__main__":
    main()