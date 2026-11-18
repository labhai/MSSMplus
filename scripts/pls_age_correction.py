"""Generate fixed MSSM+ maps.

Eight GWC maps, cortical thickness, sulcal depth, and cortical curvature are
integrated by vertex-wise PLS-DA. Optional age residualization is fitted only
on an independent healthy-reference cohort and is applied after PLS-DA.
"""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from sklearn.cross_decomposition import PLSRegression


FEATURE_NAMES = [
    "GWC_0.5mm_GM20",
    "GWC_1.0mm_GM20",
    "GWC_0.5mm_GM40",
    "GWC_1.0mm_GM40",
    "GWC_0.5mm_GM60",
    "GWC_1.0mm_GM60",
    "GWC_0.5mm_GM80",
    "GWC_1.0mm_GM80",
    "CT",
    "Sulc",
    "Curv",
]


def binary_label_map(config: dict) -> dict[str, int]:
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


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = ["DATA_DIR", "INFO_PATH", "MAP_DIR", "GWR_TYPES"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Set the following config values: {', '.join(missing)}")
    return config


def ordered_metadata(
    info_path: str,
    split: str,
    groups: list[str],
    expected_group: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(info_path)
    required = {"Subject", "Age", "Group", "Split"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Metadata is missing columns: {missing}")
    frame = frame.loc[frame["Split"].astype(str) == str(split)].copy()
    if expected_group is not None:
        frame = frame.loc[frame["Group"] == expected_group].copy()
        groups = [expected_group]
    unknown = sorted(set(frame["Group"]) - set(groups))
    if unknown:
        raise ValueError(f"Unexpected groups in split {split}: {unknown}")
    if frame.empty:
        raise ValueError(f"No metadata rows found for split {split!r}.")
    if frame["Subject"].duplicated().any():
        duplicates = frame.loc[frame["Subject"].duplicated(), "Subject"].tolist()
        raise ValueError(f"Duplicate subject IDs in metadata: {duplicates[:5]}")
    order = {group: index for index, group in enumerate(groups)}
    frame["_group_order"] = frame["Group"].map(order)
    frame["_source_order"] = np.arange(len(frame))
    return (
        frame.sort_values(["_group_order", "_source_order"], kind="stable")
        .drop(columns=["_group_order", "_source_order"])
        .reset_index(drop=True)
    )


def _load_vector(path: Path, loader: str, expected_vertices: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required surface input is missing: {path}")
    if loader == "image":
        vector = np.asarray(nib.load(path).get_fdata()).reshape(-1)
    else:
        vector = np.asarray(nib.freesurfer.io.read_morph_data(path)).reshape(-1)
    if vector.size != expected_vertices:
        raise ValueError(
            f"Expected {expected_vertices} vertices, got {vector.size}: {path}"
        )
    return vector.astype(np.float32, copy=False)


def fixed_feature_paths(config: dict) -> list[tuple[str, str, str, str]]:
    gwr_types = list(config["GWR_TYPES"])
    if len(gwr_types) != 16:
        raise ValueError("GWR_TYPES must contain eight LH/RH file pairs (16 files).")
    pairs: list[tuple[str, str, str, str]] = []
    gwr_prefix = config.get("GWR_PREFIX", "mri/GWC/reg_surf2fsaverage")
    for index in range(0, 16, 2):
        lh_name, rh_name = gwr_types[index : index + 2]
        if not lh_name.startswith("lh_") or not rh_name.startswith("rh_"):
            raise ValueError(
                "GWR_TYPES must alternate LH then RH for each GWC scale/depth."
            )
        pairs.append(
            (
                FEATURE_NAMES[index // 2],
                str(Path(gwr_prefix) / lh_name),
                str(Path(gwr_prefix) / rh_name),
                "image",
            )
        )
    thick_prefix = config.get("THICK_PREFIX", "surf/reg_surf2fsaverage")
    surface_prefix = config.get("SURFACE_PREFIX", "surf")
    pairs.extend(
        [
            (
                "CT",
                str(Path(thick_prefix) / "lh.thickness"),
                str(Path(thick_prefix) / "rh.thickness"),
                "morph",
            ),
            (
                "Sulc",
                str(Path(surface_prefix) / "lh.fsaverage.sulc"),
                str(Path(surface_prefix) / "rh.fsaverage.sulc"),
                "morph",
            ),
            (
                "Curv",
                str(Path(surface_prefix) / "lh.fsaverage.curv"),
                str(Path(surface_prefix) / "rh.fsaverage.curv"),
                "morph",
            ),
        ]
    )
    return pairs


def load_fixed_mssmplus_features(
    data_dir: str,
    metadata: pd.DataFrame,
    split: str,
    config: dict,
) -> np.ndarray:
    n_vertices = int(config.get("N_VERTEX", 327684))
    if n_vertices % 2:
        raise ValueError("N_VERTEX must contain equal LH/RH vertex counts.")
    n_hemi = n_vertices // 2
    feature_paths = fixed_feature_paths(config)
    if [entry[0] for entry in feature_paths] != FEATURE_NAMES:
        raise RuntimeError("Internal fixed-feature order changed unexpectedly.")
    data = np.empty(
        (len(metadata), len(feature_paths), n_vertices), dtype=np.float32
    )
    for subject_index, row in metadata.iterrows():
        subject_dir = Path(data_dir) / str(split) / str(row.Group) / str(row.Subject)
        if not subject_dir.is_dir():
            raise FileNotFoundError(f"Subject directory is missing: {subject_dir}")
        for feature_index, (_, lh_rel, rh_rel, loader) in enumerate(feature_paths):
            data[subject_index, feature_index, :n_hemi] = _load_vector(
                subject_dir / lh_rel, loader, n_hemi
            )
            data[subject_index, feature_index, n_hemi:] = _load_vector(
                subject_dir / rh_rel, loader, n_hemi
            )
    return data


def load_cortical_mask(config: dict) -> np.ndarray:
    n_vertices = int(config.get("N_VERTEX", 327684))
    masks = np.ones(n_vertices, dtype=bool)
    medial_wall_paths = config.get("MEDIALWALL_MASKS", [])
    if not medial_wall_paths:
        return masks
    if len(medial_wall_paths) != 2:
        raise ValueError("MEDIALWALL_MASKS must contain LH and RH label paths.")
    freesurfer_home = os.environ.get("FREESURFER_HOME")
    if not freesurfer_home:
        raise EnvironmentError(
            "FREESURFER_HOME must be set when MEDIALWALL_MASKS are configured."
        )
    subjects_dir = Path(freesurfer_home) / "subjects"
    n_hemi = n_vertices // 2
    lh_mask = nib.freesurfer.io.read_label(subjects_dir / medial_wall_paths[0])
    rh_mask = nib.freesurfer.io.read_label(subjects_dir / medial_wall_paths[1])
    masks[lh_mask] = False
    masks[rh_mask + n_hemi] = False
    return masks


def _reduce_vertex_block(
    vertex_indices: np.ndarray,
    target_block: np.ndarray,
    labels: np.ndarray,
    target_ages: np.ndarray,
    reference_block: np.ndarray | None,
    reference_ages: np.ndarray | None,
    polynomial_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.empty((target_block.shape[0], target_block.shape[2]), dtype=np.float32)
    target_age_design = np.vander(
        target_ages, N=polynomial_degree + 1, increasing=True
    )
    reference_age_design = None
    if reference_ages is not None:
        reference_age_design = np.vander(
            reference_ages, N=polynomial_degree + 1, increasing=True
        )
    for local_index, global_index in enumerate(vertex_indices):
        model = PLSRegression(n_components=1, scale=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                model.fit(target_block[:, :, local_index], labels)
            except Exception as exc:
                raise RuntimeError(f"PLS-DA failed at vertex {global_index}: {exc}") from exc
        scores = model.x_scores_[:, 0]
        if reference_block is not None:
            reference_scores = model.transform(
                reference_block[:, :, local_index]
            )[:, 0]
            coefficients, _, _, _ = np.linalg.lstsq(
                reference_age_design, reference_scores, rcond=None
            )
            scores = scores - target_age_design @ coefficients
        result[:, local_index] = scores.astype(np.float32, copy=False)
    return vertex_indices, result


def generate_mssmplus_map(
    target_data: np.ndarray,
    labels: np.ndarray,
    target_ages: np.ndarray,
    cortical_mask: np.ndarray,
    config: dict,
    reference_data: np.ndarray | None = None,
    reference_ages: np.ndarray | None = None,
) -> np.ndarray:
    correction = bool(config.get("CORRECTION", True))
    if correction != (reference_data is not None and reference_ages is not None):
        raise ValueError(
            "CORRECTION=true requires the independent healthy age-reference cohort."
        )
    if len(np.unique(labels)) != 2:
        raise ValueError("PLS-DA requires both configured labels in the target cohort.")
    cortical_vertices = np.flatnonzero(cortical_mask)
    block_size = int(config.get("VERTEX_BLOCK_SIZE", 512))
    max_workers = max(1, int(config.get("MAX_WORKERS", 1)))
    polynomial_degree = int(config.get("POLYNOMIAL_DEGREE", 1))
    if polynomial_degree < 1:
        raise ValueError("POLYNOMIAL_DEGREE must be at least 1.")
    output = np.zeros(
        (target_data.shape[0], target_data.shape[2]), dtype=np.float32
    )
    blocks = [
        cortical_vertices[start : start + block_size]
        for start in range(0, len(cortical_vertices), block_size)
    ]

    def arguments(vertices: np.ndarray) -> tuple:
        ref_block = (
            reference_data[:, :, vertices] if reference_data is not None else None
        )
        return (
            vertices,
            target_data[:, :, vertices],
            labels,
            target_ages,
            ref_block,
            reference_ages,
            polynomial_degree,
        )

    if max_workers == 1:
        for block_number, vertices in enumerate(blocks, start=1):
            indices, values = _reduce_vertex_block(*arguments(vertices))
            output[:, indices] = values
            print(f"PLS-DA block {block_number}/{len(blocks)} complete")
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            block_iterator = iter(enumerate(blocks, start=1))
            futures = {}

            def submit_next() -> bool:
                try:
                    number, vertices = next(block_iterator)
                except StopIteration:
                    return False
                future = executor.submit(_reduce_vertex_block, *arguments(vertices))
                futures[future] = number
                return True

            for _ in range(max_workers):
                if not submit_next():
                    break
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    number = futures.pop(future)
                    indices, values = future.result()
                    output[:, indices] = values
                    print(f"PLS-DA block {number}/{len(blocks)} complete")
                    submit_next()
    return output


def _surface_affine(hemi: str, n_hemi: int) -> np.ndarray:
    freesurfer_home = os.environ.get("FREESURFER_HOME")
    if freesurfer_home:
        reference = (
            Path(freesurfer_home)
            / "subjects"
            / "fsaverage"
            / "surf"
            / f"{hemi}.orig.avg.area.mgh"
        )
        if reference.is_file():
            return nib.load(reference).affine
    return np.array(
        [
            [-1.0, 0.0, 0.0, n_hemi / 2.0],
            [0.0, 0.0, 1.0, -0.5],
            [0.0, -1.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def save_outputs(
    values: np.ndarray,
    metadata: pd.DataFrame,
    split: str,
    config: dict,
) -> Path:
    output_name = config.get("OUTPUT_NAME", "MSSMplus")
    output_dir = Path(config["MAP_DIR"]) / str(split) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    n_hemi = values.shape[1] // 2
    for hemi_index, hemi in enumerate(("lh", "rh")):
        hemi_values = values[:, hemi_index * n_hemi : (hemi_index + 1) * n_hemi]
        image_data = hemi_values.T.reshape(n_hemi, 1, 1, len(metadata))
        nib.save(
            nib.freesurfer.MGHImage(
                image_data.astype(np.float32), _surface_affine(hemi, n_hemi)
            ),
            output_dir / f"{hemi}_concat.mgh",
        )
        subject_dir = output_dir / hemi
        subject_dir.mkdir(exist_ok=True)
        for subject_index, subject in enumerate(metadata["Subject"]):
            nib.freesurfer.io.write_morph_data(
                subject_dir / f"{hemi}.{subject}.mssmplus",
                hemi_values[subject_index].astype(np.float32),
            )
    sidecar = metadata[["Subject", "Age", "Group", "Split"]].copy()
    sidecar.insert(0, "row_index", np.arange(len(sidecar)))
    sidecar.to_csv(output_dir / "subjects.csv", index=False)
    fsgd_path = Path(config["MAP_DIR"]) / str(split) / f"{split}.fsgd"
    fsgd_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fsgd_path, "w", encoding="utf-8") as stream:
        stream.write(f"GroupDescriptorFile 1\nTitle {split}\n")
        for group in config.get("GROUPS", ["AD", "CN"]):
            stream.write(f"Class {group}\n")
        for row in metadata.itertuples(index=False):
            stream.write(f"Input {row.Subject} {row.Group}\n")
    return output_dir


def process_split(config: dict, split: str) -> None:
    groups = [str(group) for group in config.get("GROUPS", ["AD", "CN"])]
    label_map = binary_label_map(config)
    if set(groups) != set(label_map):
        raise ValueError(
            "GROUPS and LABEL_MAP must contain the same two group names; "
            f"got GROUPS={groups} and LABEL_MAP={list(label_map)}."
        )
    metadata = ordered_metadata(config["INFO_PATH"], split, groups)
    labels = metadata["Group"].map(label_map).to_numpy(dtype=np.float32)
    ages = metadata["Age"].to_numpy(dtype=np.float64)
    target_data = load_fixed_mssmplus_features(
        config["DATA_DIR"], metadata, split, config
    )
    reference_data = None
    reference_ages = None
    if bool(config.get("CORRECTION", True)):
        reference_data_dir = config.get("AGE_REFERENCE_DATA_DIR", "")
        reference_info_path = config.get("AGE_REFERENCE_INFO_PATH", "")
        if not reference_data_dir or not reference_info_path:
            raise ValueError(
                "CORRECTION=true requires AGE_REFERENCE_DATA_DIR and "
                "AGE_REFERENCE_INFO_PATH."
            )
        reference_split = str(config.get("AGE_REFERENCE_SPLIT", "AgeReference"))
        reference_group = next(
            group for group, label in label_map.items() if label == 0
        )
        reference_metadata = ordered_metadata(
            reference_info_path,
            reference_split,
            [reference_group],
            expected_group=reference_group,
        )
        expected_size = int(config.get("EXPECTED_AGE_REFERENCE_SIZE", 150))
        if expected_size > 0 and len(reference_metadata) != expected_size:
            raise ValueError(
                f"Expected {expected_size} independent {reference_group} "
                "age-reference subjects, "
                f"found {len(reference_metadata)}."
            )
        strata_columns = list(config.get("AGE_REFERENCE_STRATIFY_COLUMNS", []))
        if strata_columns:
            missing_columns = sorted(
                set(strata_columns).difference(reference_metadata.columns)
            )
            if missing_columns:
                raise ValueError(
                    "Age-reference metadata is missing stratification columns: "
                    f"{missing_columns}"
                )
            strata_counts = reference_metadata.groupby(
                strata_columns, dropna=False
            ).size()
            expected_strata = int(config.get("EXPECTED_AGE_REFERENCE_STRATA", 6))
            expected_per_stratum = int(
                config.get("EXPECTED_AGE_REFERENCE_PER_STRATUM", 25)
            )
            if len(strata_counts) != expected_strata or not (
                strata_counts == expected_per_stratum
            ).all():
                raise ValueError(
                    "The configured age-reference cohort requires "
                    f"{expected_strata} strata with {expected_per_stratum} "
                    f"{reference_group} "
                    f"subjects each; found {strata_counts.to_dict()}."
                )
        reference_data = load_fixed_mssmplus_features(
            reference_data_dir, reference_metadata, reference_split, config
        )
        reference_ages = reference_metadata["Age"].to_numpy(dtype=np.float64)
    values = generate_mssmplus_map(
        target_data,
        labels,
        ages,
        load_cortical_mask(config),
        config,
        reference_data,
        reference_ages,
    )
    output_dir = save_outputs(values, metadata, split, config)
    correction_name = "age-corrected" if config.get("CORRECTION", True) else "uncorrected"
    print(
        f"Saved fixed GWC+CT+Sulc+Curv {correction_name} MSSM+ maps: {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="MSSM+ YAML configuration")
    parser.add_argument(
        "--split", action="append", help="Process only this split; may be repeated"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    splits = args.split or [str(split) for split in config.get("SPLITS", ["All"])]
    for split in splits:
        process_split(config, split)


if __name__ == "__main__":
    main()
