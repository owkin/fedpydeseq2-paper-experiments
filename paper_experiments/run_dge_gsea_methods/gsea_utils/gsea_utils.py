import pickle
import shutil
import subprocess
import tempfile
from itertools import product
from pathlib import Path
from typing import Any
from typing import cast

import pandas as pd
from fedpydeseq2_datasets.constants import TCGADatasetNames
from fedpydeseq2_datasets.utils import get_experiment_id
from fedpydeseq2_datasets.utils import get_ground_truth_dds_name
from fedpydeseq2_datasets.utils import get_valid_centers_from_subfolders_file
from loguru import logger

from paper_experiments.run_dge_gsea_methods.meta_analysis_tcga_pipe import (
    get_meta_analysis_id,
)
from paper_experiments.utils.constants import DGE_MODES
from paper_experiments.utils.constants import MetaAnalysisParameter


def get_input_output_paths_pydeseq2_per_center(
    dataset_name: TCGADatasetNames,
    centers_results_path: Path,
    tmp_centers_results_dir: Path,
    results_path: Path,
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    heterogeneity_method: str | None = None,
    heterogeneity_method_param: float | None = None,
    reference_dds_ref_level: tuple[str, str] = ("stage", "Advanced"),
    **pydeseq2_kwargs: Any,
) -> tuple[list, list]:
    """Get input and output paths for the GSEA for each center.

    Parameters
    ----------
    dataset_name : TCAGDatasetNames
        The dataset names to use

    centers_results_path : Path
        The path to the centers results

    tmp_centers_results_dir : Path
        The path to the temporary directory, where the csv files will be saved.

    results_path : Path
        The path to save the results

    small_samples : bool
        If True, use a small number of samples

    small_genes : bool
        If True, use a small number of genes

    only_two_centers : bool
        If True, use only two centers

    design_factors : str or list[str]
        The design factors to use

    continuous_factors : list[str] or None
        The continuous factors to use

    heterogeneity_method : str or None
        The heterogeneity method to use

    heterogeneity_method_param : float or None
        The parameter for the heterogeneity method

    reference_dds_ref_level : tuple[str, str]
        The reference level of the design factor

    **pydeseq2_kwargs : Any
        Additional arguments to pass to the pydeseq2 function.

    Returns
    -------
    tuple[list,list]
        The input and output paths
    """
    centers_results_path = Path(centers_results_path)
    refit_cooks = pydeseq2_kwargs.get("refit_cooks", True)
    experiment_id = get_experiment_id(
        dataset_name=dataset_name,
        small_samples=small_samples,
        small_genes=small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        heterogeneity_method=heterogeneity_method,
        heterogeneity_method_param=heterogeneity_method_param,
        **pydeseq2_kwargs,
    )

    center_dds_name = get_ground_truth_dds_name(
        reference_dds_ref_level=reference_dds_ref_level,
        refit_cooks=refit_cooks,
        pooled=False,
    )

    experiments_results_path = centers_results_path / experiment_id

    _, existing_centers = get_valid_centers_from_subfolders_file(
        experiments_results_path, f"{center_dds_name}_stats_res.pkl", pkl=True
    )

    # See which centers exist

    input_pkl_paths = [
        (
            experiments_results_path
            / f"center_{center_id}"
            / f"{center_dds_name}_stats_res.pkl"
        )
        for center_id in existing_centers
    ]

    input_paths = []
    for input_pkl_path, center_id in zip(
        input_pkl_paths, existing_centers, strict=True
    ):
        with open(input_pkl_path, "rb") as f:
            stats_res = pickle.load(f)
            tmp_centers_results_dir.mkdir(parents=True, exist_ok=True)
            input_path = (
                tmp_centers_results_dir
                / experiment_id
                / f"center_{center_id}"
                / f"{center_dds_name}_stats_res.csv"
            )
            input_paths.append(input_path)
            stats_res["results_df"].dropna().to_csv(input_path)

    output_paths = [
        results_path
        / experiment_id
        / f"center_{center_id}"
        / f"{center_dds_name}_gsea_results.csv"
        for center_id in existing_centers
    ]

    return input_paths, output_paths


def get_input_output_paths_pydeseq2_largest(
    dataset_name: TCGADatasetNames,
    centers_results_path: Path,
    tmp_centers_results_dir: Path,
    results_path: Path,
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    heterogeneity_method: str | None = None,
    heterogeneity_method_param: float | None = None,
    reference_dds_ref_level: tuple[str, str] = ("stage", "Advanced"),
    **pydeseq2_kwargs: Any,
) -> tuple[list, list]:
    """Get input and output paths for the GSEA for the largest center.

    Parameters
    ----------
    dataset_name : TCAGDatasetNames
        The dataset names to use

    centers_results_path : Path
        The path to the centers results

    tmp_centers_results_dir : Path
        The path to the temporary directory, where the csv files will be saved.

    results_path : Path
        The path to save the results

    small_samples : bool
        If True, use a small number of samples

    small_genes : bool
        If True, use a small number of genes

    only_two_centers : bool
        If True, use only two centers

    design_factors : str or list[str]
        The design factors to use

    continuous_factors : list[str] or None
        The continuous factors to use

    heterogeneity_method : str or None
        The heterogeneity method to use

    heterogeneity_method_param : float or None
        The parameter for the heterogeneity method

    reference_dds_ref_level : tuple[str, str]
        The reference level of the design factor

    **pydeseq2_kwargs : Any
        Additional arguments to pass to the pydeseq2 function.

    Returns
    -------
    tuple[list,list]
        The input and output paths
    """
    centers_results_path = Path(centers_results_path)
    refit_cooks = pydeseq2_kwargs.get("refit_cooks", True)
    experiment_id = get_experiment_id(
        dataset_name=dataset_name,
        small_samples=small_samples,
        small_genes=small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        heterogeneity_method=heterogeneity_method,
        heterogeneity_method_param=heterogeneity_method_param,
        **pydeseq2_kwargs,
    )

    center_dds_name = get_ground_truth_dds_name(
        reference_dds_ref_level=reference_dds_ref_level,
        refit_cooks=refit_cooks,
        pooled=False,
    )

    experiments_results_path = centers_results_path / experiment_id

    _, existing_centers = get_valid_centers_from_subfolders_file(
        experiments_results_path, f"{center_dds_name}_stats_res.pkl", pkl=True
    )

    # See which is the largest center
    center_sizes = []
    for center_id in existing_centers:
        with open(
            experiments_results_path
            / f"center_{center_id}"
            / f"{center_dds_name}_stats_res.pkl",
            "rb",
        ) as f:
            center_sizes.append(pickle.load(f)["n_obs"])

    largest_center_id = existing_centers[center_sizes.index(max(center_sizes))]

    results_df = pickle.load(
        open(
            experiments_results_path
            / f"center_{largest_center_id}"
            / f"{center_dds_name}_stats_res.pkl",
            "rb",
        )
    )["results_df"].dropna()

    # Save it to the temporary directory
    tmp_dir = Path(
        tmp_centers_results_dir, experiment_id, f"center_{largest_center_id}"
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tmp_dir / f"{center_dds_name}_stats_res.csv")

    input_paths = [tmp_dir / f"{center_dds_name}_stats_res.csv"]

    output_paths = [
        results_path / experiment_id / f"{center_dds_name}_gsea_results.csv"
    ]

    return input_paths, output_paths


def get_input_output_paths_pydeseq2_pooled(
    dataset_name: TCGADatasetNames,
    pooled_results_path: Path,
    tmp_pooled_results_dir: Path,
    results_path: Path,
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    heterogeneity_method: str | None = None,
    heterogeneity_method_param: float | None = None,
    reference_dds_ref_level: tuple[str, str] = ("stage", "Advanced"),
    **pydeseq2_kwargs: Any,
) -> tuple[Path, Path]:
    """Get input and output paths for the GSEA for the pooled results.

    Parameters
    ----------
    dataset_name : TCAGDatasetNames
        The dataset names to use

    pooled_results_path : Path
        The path to the pooled results

    tmp_pooled_results_dir : Path
        The path to the temporary directory, where the csv files will be saved.

    results_path : Path
        The path to save the results

    small_samples : bool
        If True, use a small number of samples

    small_genes : bool
        If True, use a small number of genes

    only_two_centers : bool
        If True, use only two centers

    design_factors : str or list[str]
        The design factors to use

    continuous_factors : list[str] or None
        The continuous factors to use

    heterogeneity_method : str or None
        The heterogeneity method to use

    heterogeneity_method_param : float or None
        The parameter for the heterogeneity method

    reference_dds_ref_level : tuple[str, str]
        The reference level of the design factor

    **pydeseq2_kwargs : Any
        Additional arguments to pass to the pydeseq2 function.

    Returns
    -------
    tuple[Path,Path]
        The input and output paths
    """
    pooled_results_path = Path(pooled_results_path)
    refit_cooks = pydeseq2_kwargs.get("refit_cooks", True)
    experiment_id = get_experiment_id(
        dataset_name=dataset_name,
        small_samples=small_samples,
        small_genes=small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        heterogeneity_method=heterogeneity_method,
        heterogeneity_method_param=heterogeneity_method_param,
        **pydeseq2_kwargs,
    )

    center_dds_name = get_ground_truth_dds_name(
        reference_dds_ref_level=reference_dds_ref_level,
        refit_cooks=refit_cooks,
        pooled=True,
    )

    input_path = (
        tmp_pooled_results_dir / experiment_id / f"{center_dds_name}_stats_res.csv"
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)

    pkl_path = pooled_results_path / experiment_id / f"{center_dds_name}_stats_res.pkl"
    with open(pkl_path, "rb") as f:
        stats_res = pickle.load(f)
        stats_res["results_df"].dropna().to_csv(input_path)

    output_path = results_path / experiment_id / f"{center_dds_name}_gsea_results.csv"

    return input_path, output_path


def save_fl_result_as_csv(
    dataset_name: TCGADatasetNames,
    fl_results_path: Path,
    tmp_dir: Path,
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    heterogeneity_method: str | None = None,
    heterogeneity_method_param: float | None = None,
    force: bool = False,
    **pydeseq2_kwargs: Any,
):
    """Convert the fl result to csv in order to perform the gsea analysis.

    Parameters
    ----------
    dataset_name : TCAGDatasetNames
        The dataset name to use.

    fl_results_path : Path
        The path to the fl results.

    tmp_dir : Path
        The path to the temporary directory.

    small_samples : bool
        If True, use a small number of samples

    small_genes : bool
        If True, use a small number of genes

    only_two_centers : bool
        If True, use only two centers

    design_factors : str or list[str]
        The design factors to use

    continuous_factors : list[str] or None
        The continuous factors to use

    heterogeneity_method : str or None
        The heterogeneity method to use

    heterogeneity_method_param : float or None
        The parameter for the heterogeneity method

    force : bool
        If True, force the conversion to csv.

    **pydeseq2_kwargs : Any
        Additional arguments to pass to the pydeseq2 function.
    """
    experiment_id = get_experiment_id(
        dataset_name=dataset_name,
        small_samples=small_samples,
        small_genes=small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        heterogeneity_method=heterogeneity_method,
        heterogeneity_method_param=heterogeneity_method_param,
        **pydeseq2_kwargs,
    )

    pkl_path = fl_results_path / experiment_id / "fl_result.pkl"
    csv_path = tmp_dir / experiment_id / "fl_result.csv"

    if not csv_path.exists() or force:
        logger.info(f"Converting {dataset_name}, {experiment_id} fl result to csv.")
        with open(pkl_path, "rb") as f:
            fl_r = pickle.load(f)

        df_fl = pd.DataFrame(
            {
                "stat": fl_r["wald_statistics"],
                "pvalue": fl_r["p_values"],
                "padj": fl_r["padj"].values,
            },
            index=fl_r["gene_names"],
        )

        df_fl = df_fl.dropna()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_fl.to_csv(csv_path, index_label="gene")

        logger.success(f"Converted {dataset_name}, {experiment_id} fl result to csv.")


def get_input_output_paths_fedpydeseq2(
    dataset_name: TCGADatasetNames,
    fl_results_path: Path,
    tmp_fl_results_dir: Path,
    results_path: Path,
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    heterogeneity_method: str | None = None,
    heterogeneity_method_param: float | None = None,
    **pydeseq2_kwargs: Any,
) -> tuple[Path, Path]:
    """Get input and output paths for the GSEA for the fedpydeseq2 results.

    Parameters
    ----------
    dataset_name : TCAGDatasetNames
        The dataset names to use

    fl_results_path : Path
        The path to the fl results. This is before the experiment id.

    tmp_fl_results_dir : Path
        The path to the temporary directory, where the csv files will be saved.

    results_path : Path
        The path to save the results

    small_samples : bool
        If True, use a small number of samples

    small_genes : bool
        If True, use a small number of genes

    only_two_centers : bool
        If True, use only two centers

    design_factors : str or list[str]
        The design factors to use

    continuous_factors : list[str] or None
        The continuous factors to use

    heterogeneity_method : str or None
        The heterogeneity method to use

    heterogeneity_method_param : float or None
        The parameter for the heterogeneity method

    **pydeseq2_kwargs : Any
        Additional arguments to pass to the fedpydeseq2 function.


    Returns
    -------
    tuple[Path,Path]
        The input and output paths
    """
    fl_results_path = Path(fl_results_path)
    tmp_fl_results_dir = Path(tmp_fl_results_dir)

    # Start by creating the fedpydeseq2 csv files
    save_fl_result_as_csv(
        dataset_name=dataset_name,
        fl_results_path=fl_results_path,
        tmp_dir=tmp_fl_results_dir,
        small_samples=small_samples,
        small_genes=small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        heterogeneity_method=heterogeneity_method,
        heterogeneity_method_param=heterogeneity_method_param,
        **pydeseq2_kwargs,
    )

    experiment_id = get_experiment_id(
        dataset_name=dataset_name,
        small_samples=small_samples,
        small_genes=small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        heterogeneity_method=heterogeneity_method,
        heterogeneity_method_param=heterogeneity_method_param,
        **pydeseq2_kwargs,
    )

    # See which centers exist

    input_path = tmp_fl_results_dir / experiment_id / "fl_result.csv"

    output_path = results_path / experiment_id / "gsea_results.csv"

    return input_path, output_path


def get_input_output_paths_meta_analysis(
    dataset_name: TCGADatasetNames,
    meta_analysis_results_path: Path,
    results_path: Path,
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    heterogeneity_method: str | None = None,
    heterogeneity_method_param: float | None = None,
    meta_analysis_type: str = "random_effect",
    method_random_effects: str | None = "dl",
    method_combination: str | None = "stouffer",
    stats_clip_value: float | None = None,
    **pydeseq2_kwargs: Any,
) -> tuple[Path, Path]:
    """Get input and output paths for the GSEA for the fedpydeseq2 results.

    Parameters
    ----------
    dataset_name : TCAGDatasetNames
        The dataset names to use

    meta_analysis_results_path : Path
        The path to the fl results. This is before the experiment id.

    results_path : Path
        The path to save the results

    small_samples : bool
        If True, use a small number of samples

    small_genes : bool
        If True, use a small number of genes

    only_two_centers : bool
        If True, use only two centers

    design_factors : str or list[str]
        The design factors to use

    continuous_factors : list[str] or None
        The continuous factors to use

    heterogeneity_method : str or None
        The heterogeneity method to use

    heterogeneity_method_param : float or None
        The parameter for the heterogeneity method

    meta_analysis_type : str, optional
        The type of meta-analysis to use, by default "random_effect".
        Can be in ["pvalue_combination", "fixed_effect", "random_effect"].

    method_random_effects : Optional[str], optional
        The method for random effects, by default "dl".
        Can be in ["dl", "iterated", "chi2"].

    method_combination : Optional[str], optional
        The method for combination, by default "stouffer".
        Can be in ["stouffer", "fisher"].

    stats_clip_value : Optional[float], optional
        The value to clip the statistics to, by default None.

    **pydeseq2_kwargs : Any
        Additional arguments to pass to the fedpydeseq2 function.

    Returns
    -------
    tuple[Path,Path]
        The input and output paths
    """
    meta_analysis_results_path = Path(meta_analysis_results_path)

    experiment_id = get_experiment_id(
        dataset_name=dataset_name,
        small_samples=small_samples,
        small_genes=small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        heterogeneity_method=heterogeneity_method,
        heterogeneity_method_param=heterogeneity_method_param,
        **pydeseq2_kwargs,
    )

    # Generate meta analysis id
    meta_analysis_id = get_meta_analysis_id(
        meta_analysis_type, method_random_effects, method_combination, stats_clip_value
    )
    # See which centers exist

    input_path = meta_analysis_results_path / experiment_id / f"{meta_analysis_id}.csv"

    output_path = results_path / experiment_id / f"{meta_analysis_id}_gsea_results.csv"

    return input_path, output_path


def run_gsea_method(
    dataset_names: list[TCGADatasetNames],
    dge_results_paths: dict[str, Path],
    gsea_results_paths: dict[str, Path],
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    heterogeneity_method: str | None = None,
    heterogeneity_method_params: list[float | None] | None = None,
    reference_dds_ref_level: tuple[str, str] | None = ("stage", "Advanced"),
    meta_analysis_parameters: list[MetaAnalysisParameter] | None = None,
    conda_activate_path: str | Path | None = None,
    **pydeseq2_kwargs: Any,
):
    """Run a bash script to run the GSEA analysis.

    Parameters
    ----------
    dataset_names : list[TCGADatasetNames]
        The dataset names to use

    dge_results_paths : dict[str, Path]
        The path to the centers results, with keys DGE modes.

    gsea_results_paths : dict[str, Path]
        The path to save the results, with keys DGE modes.

    small_samples : bool
        If True, use a small number of samples

    small_genes : bool
        If True, use a small number of genes

    only_two_centers : bool
        If True, use only two centers

    design_factors : str or list[str]
        The design factors to use

    continuous_factors : list[str] or None
        The continuous factors to use

    heterogeneity_method : str or None
        The heterogeneity method to use

    heterogeneity_method_params : list[Optional[float]] or None
        The parameters for the heterogeneity method. Will be
        applied to all dge methods except pydeseq2.

    reference_dds_ref_level : tuple[str, str] or None
        The reference level of the design factor. Must be
        a tuple for pydeseq2 modes.

    meta_analysis_parameters : list[MetaAnalysisParameter] or None
        The meta analysis parameters to use. Must be a list of tuples
        with the meta analysis type, the method for random effects and
        the method for combination.

    conda_activate_path : str or Path or None
        The path to the conda activate script. If None, use the default
        conda activate script.

    **pydeseq2_kwargs : Any
        Additional arguments to pass to the pydeseq2 function.
    """
    # Check the keys of both dictionaries
    assert set(dge_results_paths.keys()) == set(gsea_results_paths.keys())
    assert set(dge_results_paths.keys()).issubset(set(DGE_MODES))

    if heterogeneity_method_params is None:
        heterogeneity_method_params = [None]

    # Create an environment in a temporary directory
    temp_dir = tempfile.mkdtemp()

    envs_dir = Path(temp_dir, "envs")
    envs_dir.mkdir(parents=True, exist_ok=True)
    tmp_fl_simulated_results_dir = Path(temp_dir, "fl_results")
    tmp_fl_simulated_results_dir.mkdir(parents=True, exist_ok=True)
    tmp_fl_remote_results_dir = Path(temp_dir, "fl_results_remote")
    tmp_fl_remote_results_dir.mkdir(parents=True, exist_ok=True)
    tmp_pooled_results_dir = Path(temp_dir, "pooled_results")
    tmp_pooled_results_dir.mkdir(parents=True, exist_ok=True)
    tmp_centers_results_dir = Path(temp_dir, "centers_results")
    tmp_centers_results_dir.mkdir(parents=True, exist_ok=True)
    env_prefix = Path(envs_dir, "r_gsea")
    env_file = Path(__file__).parent / "environment.yml"
    create_env_cmd = f"yes | conda env create --prefix {env_prefix} --file {env_file}"
    try:
        subprocess.run(create_env_cmd, shell=True, check=True, executable="/bin/bash")
    except Exception as e:
        logger.error(f"Error creating the conda environment: {e}")
        shutil.rmtree(temp_dir)
        raise e

    access_conda_command = (
        """
        conda init bash
        if [ -f ~/.bashrc ]; then
            . ~/.bashrc
        fi
        if [ -f ~/.bash_profile ]; then
            . ~/.bash_profile
        fi
        """
        if conda_activate_path is None
        else f"""
        . {conda_activate_path}
        """
    )

    # Add first lines needed to activate the r_gsea conda environment
    command = f"{access_conda_command} \n conda activate {env_prefix}  \n"
    gsea_r_script = Path(__file__).parent / "gsea.R"
    all_input_files = []
    all_output_files = []
    all_descriptions: list[str] = []

    # For pydeseq2, we do not add the heterogeneity method
    for dataset_name in dataset_names:
        dataset_name = cast(TCGADatasetNames, dataset_name)

        if "pydeseq2" in dge_results_paths:
            dge_results_path = dge_results_paths["pydeseq2"]
            gsea_results_path = gsea_results_paths["pydeseq2"]
            assert reference_dds_ref_level is not None
            input_file, output_file = get_input_output_paths_pydeseq2_pooled(
                dataset_name=dataset_name,
                pooled_results_path=dge_results_path,
                tmp_pooled_results_dir=tmp_pooled_results_dir,
                results_path=gsea_results_path,
                small_samples=small_samples,
                small_genes=small_genes,
                only_two_centers=only_two_centers,
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                heterogeneity_method=None,
                heterogeneity_method_param=None,
                reference_dds_ref_level=reference_dds_ref_level,
                **pydeseq2_kwargs,
            )
            all_input_files.append(input_file)
            all_output_files.append(output_file)
            all_descriptions.append(
                f"Running GSEA for dataset {dataset_name}, {input_file}"
            )

    for dataset_name, heterogeneity_method_param in product(
        dataset_names, heterogeneity_method_params
    ):
        dataset_name = cast(TCGADatasetNames, dataset_name)
        if "pydeseq2_per_center" in dge_results_paths:
            dge_results_path = dge_results_paths["pydeseq2_per_center"]
            gsea_results_path = gsea_results_paths["pydeseq2_per_center"]
            assert reference_dds_ref_level is not None
            input_files, output_files = get_input_output_paths_pydeseq2_per_center(
                dataset_name=dataset_name,
                centers_results_path=dge_results_path,
                tmp_centers_results_dir=tmp_centers_results_dir,
                results_path=gsea_results_path,
                small_samples=small_samples,
                small_genes=small_genes,
                only_two_centers=only_two_centers,
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                heterogeneity_method=heterogeneity_method,
                heterogeneity_method_param=heterogeneity_method_param,
                reference_dds_ref_level=reference_dds_ref_level,
                **pydeseq2_kwargs,
            )
            all_input_files.extend(input_files)
            all_output_files.extend(output_files)
            all_descriptions.extend(
                [
                    f"Running GSEA for dataset {dataset_name}, {input_file}"
                    for input_file in input_files
                ]
            )

        if "pydeseq2_largest" in dge_results_paths:
            dge_results_path = dge_results_paths["pydeseq2_largest"]
            gsea_results_path = gsea_results_paths["pydeseq2_largest"]
            assert reference_dds_ref_level is not None
            input_files, output_files = get_input_output_paths_pydeseq2_largest(
                dataset_name=dataset_name,
                centers_results_path=dge_results_path,
                tmp_centers_results_dir=tmp_centers_results_dir,
                results_path=gsea_results_path,
                small_samples=small_samples,
                small_genes=small_genes,
                only_two_centers=only_two_centers,
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                heterogeneity_method=heterogeneity_method,
                heterogeneity_method_param=heterogeneity_method_param,
                reference_dds_ref_level=reference_dds_ref_level,
                **pydeseq2_kwargs,
            )
            all_input_files.extend(input_files)
            all_output_files.extend(output_files)
            all_descriptions.extend(
                [
                    f"Running GSEA for dataset {dataset_name}, {input_file}"
                    for input_file in input_files
                ]
            )

        if "fedpydeseq2_simulated" in dge_results_paths:
            dge_results_path = dge_results_paths["fedpydeseq2_simulated"]
            gsea_results_path = gsea_results_paths["fedpydeseq2_simulated"]
            input_file, output_file = get_input_output_paths_fedpydeseq2(
                dataset_name=dataset_name,
                fl_results_path=dge_results_path,
                tmp_fl_results_dir=tmp_fl_simulated_results_dir,
                results_path=gsea_results_path,
                small_samples=small_samples,
                small_genes=small_genes,
                only_two_centers=only_two_centers,
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                heterogeneity_method=heterogeneity_method,
                heterogeneity_method_param=heterogeneity_method_param,
                **pydeseq2_kwargs,
            )
            all_input_files.append(input_file)
            all_output_files.append(output_file)
            all_descriptions.append(
                f"Running GSEA for dataset {dataset_name}, {input_file}"
            )
        if "fedpydeseq2_remote" in dge_results_paths:
            dge_results_path = dge_results_paths["fedpydeseq2_remote"]
            gsea_results_path = gsea_results_paths["fedpydeseq2_remote"]
            input_file, output_file = get_input_output_paths_fedpydeseq2(
                dataset_name=dataset_name,
                fl_results_path=dge_results_path,
                tmp_fl_results_dir=tmp_fl_remote_results_dir,
                results_path=gsea_results_path,
                small_samples=small_samples,
                small_genes=small_genes,
                only_two_centers=only_two_centers,
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                heterogeneity_method=heterogeneity_method,
                heterogeneity_method_param=heterogeneity_method_param,
                **pydeseq2_kwargs,
            )
            all_input_files.append(input_file)
            all_output_files.append(output_file)
            all_descriptions.append(
                f"Running GSEA for dataset {dataset_name}, {input_file}"
            )

        if "meta_analysis" in dge_results_paths:
            dge_results_path = dge_results_paths["meta_analysis"]
            gsea_results_path = gsea_results_paths["meta_analysis"]
            assert meta_analysis_parameters is not None
            for meta_analysis_parameter in meta_analysis_parameters:
                if len(meta_analysis_parameter) == 3:
                    (
                        meta_analysis_type,
                        method_random_effects,
                        method_combination,
                    ) = meta_analysis_parameter
                    stats_clip_value = None
                elif len(meta_analysis_parameter) == 4:
                    (
                        meta_analysis_type,
                        method_random_effects,
                        method_combination,
                        stats_clip_value,
                    ) = meta_analysis_parameter
                else:
                    raise ValueError(
                        "Meta analysis parameter must have 3 or 4 elements."
                    )
                input_file, output_file = get_input_output_paths_meta_analysis(
                    dataset_name=dataset_name,
                    meta_analysis_results_path=dge_results_path,
                    results_path=gsea_results_path,
                    small_samples=small_samples,
                    small_genes=small_genes,
                    only_two_centers=only_two_centers,
                    design_factors=design_factors,
                    continuous_factors=continuous_factors,
                    heterogeneity_method=heterogeneity_method,
                    heterogeneity_method_param=heterogeneity_method_param,
                    meta_analysis_type=meta_analysis_type,
                    method_random_effects=method_random_effects,
                    method_combination=method_combination,
                    stats_clip_value=stats_clip_value,
                    **pydeseq2_kwargs,
                )
                all_input_files.append(input_file)
                all_output_files.append(output_file)
                all_descriptions.append(
                    f"Running GSEA for dataset {dataset_name},input "
                    f"{input_file}, output {output_file}"
                )

    for input_file, output_file, description in zip(
        all_input_files, all_output_files, all_descriptions, strict=False
    ):
        # create output file directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        # Echo what we are doing
        command += f"echo '{description}'\n"
        # Run the R script
        command += (
            f"Rscript {gsea_r_script} --input_file {input_file} "
            f"--output_file {output_file} \n"
        )

    try:
        logger.info("Running GSEA analysis.")
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")
        logger.success("GSEA analysis completed.")
    except Exception as e:
        logger.error(f"Error running GSEA analysis: {e}")
        raise e
    finally:
        shutil.rmtree(temp_dir)  # delete the temporary directory
