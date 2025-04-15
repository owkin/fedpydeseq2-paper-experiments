"""Run PyDESeq2."""

import os
import pickle
import tempfile
from inspect import signature
from pathlib import Path
from typing import Any

import numpy as np
from fedpydeseq2_datasets.constants import TCGADatasetNames
from fedpydeseq2_datasets.create_reference_dds import setup_tcga_ground_truth_dds
from fedpydeseq2_datasets.process_and_split_data import setup_tcga_dataset
from fedpydeseq2_datasets.utils import get_experiment_id
from fedpydeseq2_datasets.utils import get_ground_truth_dds_name
from loguru import logger
from pydeseq2.ds import DeseqStats


def run_tcga_pooled_experiments(
    dataset_name: TCGADatasetNames,
    raw_data_path: Path,
    results_path: Path,
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    heterogeneity_method: str | None = None,
    heterogeneity_method_param: float | None = None,
    **pydeseq2_kwargs: Any,
):
    """Run the TCGA experiments with pooled data.

    Parameters
    ----------
    dataset_name : TCGADatasetNames
        The name of the dataset to use

    raw_data_path : Path
        The path to the raw data
        It is assumed that the raw data is in the following structure:
        ```
        <raw_data_path>
        ├── tcga
        │   ├── COHORT
        │   │   ├── Counts_raw.parquet
        │   │   └── recount3_metadata.tsv.gz
        │   ├── centers.csv
        │   ├── tumor_purity_metadata.csv
        │   └── cleaned_clinical_metadata.csv
        ```

    results_path : Path
        The path to save the results
        It is assumed that the results will be saved in the following structure:
        ```
        <results_path>
        ├── EXPERIMENT_ID
        │   ├── GROUND_TRUTH_DDS_NAME_stats_res.pkl
        └── ...
        ```

    small_samples : bool
        If True, use a small number of samples

    small_genes : bool
        If True, use a small number of genes

    only_two_centers : bool
        If True, use only two centers

    design_factors : str or list[str]
        The design factors to use

    continuous_factors : list[str]
        The continuous factors to use

    heterogeneity_method : str or None
        The method to used to define the heterogeneity
        of the center's attribution.
        For now, only 'binomial' is supported.
        It can be used only with two centers.

    heterogeneity_method_param : float or None
        The parameter of the heterogeneity method.
        If it is 0., the data is heterogenous.
        If it is 1., the data is homogenous.

    **pydeseq2_kwargs : Any
        Any other keyword arguments to pass to the PyDESeq2 strategy
    """
    with tempfile.TemporaryDirectory() as processed_data_path_str:
        processed_data_path = Path(processed_data_path_str)

        setup_tcga_dataset(
            raw_data_path,
            processed_data_path,
            dataset_name=dataset_name,
            small_samples=small_samples,
            small_genes=small_genes,
            only_two_centers=only_two_centers,
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            heterogeneity_method=heterogeneity_method,
            heterogeneity_method_param=heterogeneity_method_param,
            force=True,
            **pydeseq2_kwargs,
        )
        setup_tcga_ground_truth_dds(
            processed_data_path,
            dataset_name=dataset_name,
            small_samples=small_samples,
            small_genes=small_genes,
            only_two_centers=only_two_centers,
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            reference_dds_ref_level=("stage", "Advanced"),
            default_refit_cooks=True,
            force=True,
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
        refit_cooks = pydeseq2_kwargs.get("refit_cooks", True)

        ground_truth_dds_name = get_ground_truth_dds_name(
            reference_dds_ref_level=("stage", "Advanced"),
            refit_cooks=refit_cooks,
            pooled=True,
        )

        os.makedirs(Path(results_path, experiment_id), exist_ok=True)
        dds_filepath = Path(
            processed_data_path,
            "pooled_data",
            "tcga",
            experiment_id,
            ground_truth_dds_name + ".pkl",
        )

        stats_res_file = Path(
            results_path,
            experiment_id,
            ground_truth_dds_name + "_stats_res.pkl",
        )

        create_and_save_pydeseq2_stats_results(
            dds_filepath,
            pydeseq2_kwargs,
            stats_res_file,
            center_id=None,
        )


def create_and_save_pydeseq2_stats_results(
    dds_filepath: Path,
    pydeseq2_kwargs: dict[str, Any],
    stats_res_file: Path,
    center_id: int | None = None,
):
    """Create and save the PyDESeq2 stats results.

    We save the results in a dictionary with the following
    keys:
    - results_df: The results DataFrame
    - n_obs: The number of observations

    The results_df is rescaled to be in natural log scale.
    it contains the following columns:
    - baseMean
    - lfc
    - lfcSE
    - stat
    - pvalue
    - padj
    and is indexed by the gene.


    Parameters
    ----------
    dds_filepath : Path
        The path to the DeseqDataSet object

    pydeseq2_kwargs : dict[str, Any]
        The keyword arguments to pass to the DeseqStats object

    stats_res_file : Path
        The path to save the stats results

    center_id : int or None
        The center id
        For logging purposes
    """
    with open(dds_filepath, "rb") as f:
        dds = pickle.load(f)

    stats_res_file.parent.mkdir(parents=True, exist_ok=True)

    if dds is None:
        if center_id is None:
            raise ValueError("DDS is not provided.")
        logger.warning(
            f"Center {center_id} has no results, because did not "
            "have a full design matrix. Skipping."
        )
        # Save None in the stats_res_file
        with open(stats_res_file, "wb") as f:
            pickle.dump(None, f)
        return

    ds_kwargs = {
        k: v
        for k, v in pydeseq2_kwargs.items()
        if k in signature(DeseqStats).parameters
    }
    stats_res = DeseqStats(dds, **ds_kwargs)
    stats_res.summary()

    results_df = stats_res.results_df
    # Now we rescale the results_df to be in log scale
    results_df["lfc"] = np.log(2) * results_df["log2FoldChange"]
    # delte the log2FoldChange column
    results_df.drop(columns=["log2FoldChange"], inplace=True)
    # rescale the lfcSE column
    results_df.loc[:, "lfcSE"] = np.log(2) * results_df["lfcSE"]

    # Set index label as gene
    results_df.index.name = "gene"

    # Get the number of observations
    n_obs = stats_res.dds.n_obs
    # Save the results in a dictionary
    stats_to_save = {
        "results_df": results_df,
        "n_obs": n_obs,
    }

    # Save the stats results
    with open(stats_res_file, "wb") as f:
        pickle.dump(stats_to_save, f)

    logger.success(f"Stats results saved in {stats_res_file}")
