import math
import pickle
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
import pandas as pd
import seaborn as sns
from fedpydeseq2.core.utils.stat_utils import build_contrast_vector
from fedpydeseq2_datasets.constants import TCGADatasetNames
from fedpydeseq2_datasets.utils import get_experiment_id
from fedpydeseq2_datasets.utils import get_ground_truth_dds_name
from fedpydeseq2_datasets.utils import get_valid_centers_from_subfolders_file
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from paper_experiments.run_dge_gsea_methods.meta_analysis_tcga_pipe import (
    get_meta_analysis_id,
)
from paper_experiments.utils.constants import MetaAnalysisParameter

NAME_MAPPING = {
    "pydeseq2_per_center": "PyDESeq2 (per center)",
    "pydeseq2": "PyDESeq2",
    "fedpydeseq2_simulated": "FedPyDESeq2 (simulated)",
    "fedpydeseq2_remote": "FedPyDESeq2",
    "pydeseq2_largest": "PyDESeq2 (largest cohort)",
    "meta_analysis": "Meta-analysis",
    "fixed_effect": "Fixed effects",
    "random_effect_dl": "Random effects\n(DerSimonian-Laird)",
    "random_effect_iterated": "Random effects\n(iterated)",
    "pydeseq2_largest": "PyDESeq2\n(largest)",
    "pvalue_combination_fisher": "Fisher",
    "pvalue_combination_stouffer": "Stouffer",
}


def get_padj_lfc_fedpydeseq2(
    fedpydeseq2_results_path: str | Path,
    experiment_id: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Get the adjusted p-values and log-fold changes from a fedpydeseq2 result file.

    Parameters
    ----------
    fedpydeseq2_results_path : Union[str, Path]
        The path to the fedpydeseq2 results.

    experiment_id : str
        The experiment id.

    Returns
    -------
    fl_padj : pd.Series
        The adjusted p-values.

    fl_LFC : pd.Series
        The log-fold changes, *in natural scale*.

    """
    result_file_path = Path(fedpydeseq2_results_path, experiment_id, "fl_result.pkl")

    with open(result_file_path, "rb") as f:
        fl_result = pickle.load(f)

    fl_padj = fl_result["padj"]
    contrast = fl_result["contrast"]
    fl_all_LFC = fl_result["LFC"]
    contrast_vector, _ = build_contrast_vector(contrast, LFC_columns=fl_all_LFC.columns)

    fl_LFC = fl_all_LFC @ contrast_vector

    return fl_padj, fl_LFC


def get_padj_lfc_pydeseq2(
    pydeseq2_results_path: str | Path,
    experiment_id: str,
    ground_truth_dds_name: str,
    center: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Get the adjusted p-values and log-fold changes from a pydeseq2 result file.

    Parameters
    ----------
    pydeseq2_results_path : Union[str, Path]
        The path to the pydeseq2 results. Must contain the experiment_id folder.

    experiment_id : str
        The experiment id.

    ground_truth_dds_name : str
        The ground truth dds name, which identifies the result file.

    center : int or None
        The center to use. If None, the pooled results are used.

    Returns
    -------
    padj : pd.Series
        The adjusted p-values.

    LFC : pd.Series
        The log-fold changes, *in natural scale*.

    """
    pydeseq2_results_path = Path(pydeseq2_results_path)
    if center is None:
        stats_file = (
            pydeseq2_results_path
            / experiment_id
            / f"{ground_truth_dds_name}_stats_res.pkl"
        )
    else:
        stats_file = (
            pydeseq2_results_path
            / experiment_id
            / f"center_{center}"
            / f"{ground_truth_dds_name}_stats_res.pkl"
        )

    with open(stats_file, "rb") as f:
        stats_res = pickle.load(f)

    results_df = stats_res["results_df"]

    return results_df["padj"], results_df["lfc"]


def get_padj_lfc_meta_analysis(
    meta_analysis_results_path: str | Path,
    experiment_id: str,
    meta_analysis_id: str,
):
    """
    Get the adjusted p-values and log-fold changes from a meta-analysis result file.

    Parameters
    ----------
    meta_analysis_results_path : Union[str, Path]
        The path to the meta-analysis results. Must contain the experiment_id folder.

    experiment_id : str
        The experiment id.

    meta_analysis_id : str
        The meta-analysis id.

    Returns
    -------
    meta_padj : pd.Series
        The adjusted p-values.

    meta_LFC : pd.Series
        The log-fold changes, *in natural scale*.

    """
    meta_analysis_results_path = Path(meta_analysis_results_path)
    stats_result_path = (
        meta_analysis_results_path / experiment_id / f"{meta_analysis_id}.csv"
    )

    meta_analysis_results = pd.read_csv(stats_result_path, index_col=0)

    meta_padj = meta_analysis_results["padj"]
    meta_LFC = meta_analysis_results["lfc"]
    return meta_padj, meta_LFC


def get_padj_lfc_from_method(
    dge_method: str,
    dge_method_results_path: str | Path,
    experiment_id: str,
    refit_cooks: bool = True,
    reference_dds_ref_level: tuple[str, str] | None = ("stage", "Advanced"),
    meta_analysis_parameters: list[MetaAnalysisParameter] | None = None,
) -> tuple[pd.Series | dict[str, pd.Series], pd.Series | dict[str, pd.Series]]:
    """
    Get the adjusted p-values and log-fold changes from a DGE method result file.

    Parameters
    ----------
    dge_method : str
        The DGE method to use.

    dge_method_results_path : Union[str, Path]
        The path to the DGE method results.

    experiment_id : str
        The experiment id.

    refit_cooks : bool
        Whether to refit cooks distance.

    reference_dds_ref_level : tuple[str, str] or None
        The reference dds ref level to use.
        Necessary if the DGE method is PyDESeq2 per center or pooled.

    meta_analysis_parameters : list[MetaAnalysisParameter] or None
        The meta-analysis parameters to use.
        Necessary if the DGE method is Meta-analysis.

    Returns
    -------
    padj : pd.Series or dict[str, pd.Series]
        The adjusted p-values for each center if the method is per center.
        The adjusted p-values for each meta-analysis if the method is Meta-analysis.
        The adjusted p-values otherwise.

    LFC : pd.Series or dict[str, pd.Series]
        The log-fold changes for each center if the method is per center.
        The log-fold changes for each meta-analysis if the method is Meta-analysis.
        The log-fold changes otherwise.
        All LFC are *in natural scale*.

    Raises
    ------
    ValueError
        If the DGE method is unknown.

    """
    dge_method_results_path = Path(dge_method_results_path)
    if dge_method.startswith("fedpydeseq2"):
        return get_padj_lfc_fedpydeseq2(dge_method_results_path, experiment_id)
    elif dge_method == "pydeseq2":
        ground_truth_dds_name = get_ground_truth_dds_name(
            reference_dds_ref_level=reference_dds_ref_level,
            refit_cooks=refit_cooks,
            pooled=True,
        )
        return get_padj_lfc_pydeseq2(
            dge_method_results_path, experiment_id, ground_truth_dds_name
        )
    elif dge_method == "pydeseq2_largest":
        ground_truth_dds_name = get_ground_truth_dds_name(
            reference_dds_ref_level=reference_dds_ref_level,
            refit_cooks=refit_cooks,
            pooled=False,
        )
        center_sizes = []
        _, existing_centers = get_valid_centers_from_subfolders_file(
            dge_method_results_path / experiment_id,
            f"{ground_truth_dds_name}_stats_res.pkl",
            pkl=True,
        )
        for center_id in existing_centers:
            with open(
                dge_method_results_path
                / experiment_id
                / f"center_{center_id}"
                / f"{ground_truth_dds_name}_stats_res.pkl",
                "rb",
            ) as f:
                center_sizes.append(pickle.load(f)["n_obs"])

        largest_center_id = existing_centers[center_sizes.index(max(center_sizes))]
        return get_padj_lfc_pydeseq2(
            dge_method_results_path,
            experiment_id,
            ground_truth_dds_name,
            center=largest_center_id,
        )
    elif dge_method == "pydeseq2_per_center":
        ground_truth_dds_name = get_ground_truth_dds_name(
            reference_dds_ref_level=reference_dds_ref_level,
            refit_cooks=refit_cooks,
            pooled=False,
        )
        # Now get available centers
        _, existing_centers = get_valid_centers_from_subfolders_file(
            dge_method_results_path / experiment_id,
            f"{ground_truth_dds_name}_stats_res.pkl",
            pkl=True,
        )
        result_dict = {
            f"PyDESeq2, center {center_id}": get_padj_lfc_pydeseq2(
                dge_method_results_path,
                experiment_id,
                ground_truth_dds_name,
                center=center_id,
            )
            for center_id in existing_centers
        }
        return {method_id: padj for method_id, (padj, _) in result_dict.items()}, {
            method_id: LFC for method_id, (_, LFC) in result_dict.items()
        }
    elif dge_method == "meta_analysis":
        assert meta_analysis_parameters is not None
        all_padj, all_LFC = {}, {}
        for meta_analysis_parameter in meta_analysis_parameters:
            meta_analysis_id = get_meta_analysis_id(*meta_analysis_parameter)
            padj, LFC = get_padj_lfc_meta_analysis(
                dge_method_results_path, experiment_id, meta_analysis_id
            )
            method_id = ", ".join(
                [
                    "Meta-analysis",
                    *[str(subparameter) for subparameter in meta_analysis_parameter],
                ]
            )
            all_padj[method_id] = padj
            all_LFC[method_id] = LFC
        # Sort the results
        sorted_keys = sorted(all_padj.keys())
        all_padj = {key: all_padj[key] for key in sorted_keys}
        all_LFC = {key: all_LFC[key] for key in sorted_keys}
        return all_padj, all_LFC
    else:
        raise ValueError(f"Unknown DGE method: {dge_method}")


def build_pan_cancer_confusion_matrix(
    method_test,
    method_ref,
    method_test_results_path: str | Path,
    method_ref_results_path: str | Path,
    dataset_names: list[TCGADatasetNames],
    save_file_path: str | Path,
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    reference_dds_ref_level: tuple[str, str] | None = ("stage", "Advanced"),
    meta_analysis_parameters: list[MetaAnalysisParameter] | None = None,
    log2fc_threshold: float = 2.0,
    padj_threshold: float = 0.05,
    **pydeseq2_kwargs: Any,
):
    """
    Make a pan-cancer confusion matrix between a test method and a reference method.

    Represents confusion matrices sides by side for each dataset, with a common
    colorbar.

    Parameters
    ----------
    method_test : str
        The tested method.

    method_ref : str
        The reference method.

    method_test_results_path : str or Path
        The path to the tested method results.

    method_ref_results_path : str or Path
        The path to the reference method results.

    dataset_names : list[TCGADatasetNames]
        The list of dataset to include in the figure.

    save_file_path : str or Path
        The path where to save the plot.

    small_samples : bool
        Whether to use small samples.

    small_genes : bool
        Whether to use small genes.

    only_two_centers : bool
        Whether to use only two centers.

    design_factors : str or list[str]
        The design factors used in the experiment.

    continuous_factors : list[str] or None
        The continuous factors used in the experiment.

    reference_dds_ref_level : tuple[str, str] or None
        The reference dds ref level to use.

    meta_analysis_parameters : list[MetaAnalysisParameter] or None
        The meta-analysis parameters to use.

    log2fc_threshold : float
        The log2-fold change threshold to define up and down regulated genes.

    padj_threshold : float
        The adjusted p-value threshold to define differentially expressed genes.

    **pydeseq2_kwargs : Any
        Additional keyword arguments to pass to the PyDESeq2 and FedPyDESeq2
        methods.
    """
    plt.clf()

    n = len(dataset_names)

    fig, axes = plt.subplots(
        2,
        int(np.ceil(n / 2)),
        figsize=(int(np.ceil(n / 2)) * 5, 8),
        constrained_layout=True,
    )
    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.6])

    logger.info(
        f"Building pan-cacer cross table for {method_test} vs " f"{method_ref}."
    )

    for i, dataset_name in enumerate(dataset_names):
        experiment_id = get_experiment_id(
            dataset_name=dataset_name,
            small_samples=small_samples,
            small_genes=small_genes,
            only_two_centers=only_two_centers,
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            **pydeseq2_kwargs,
        )

        refit_cooks = pydeseq2_kwargs.get("refit_cooks", True)

        method_test_padj, method_test_lfc = get_padj_lfc_from_method(
            method_test,
            method_test_results_path,
            experiment_id,
            refit_cooks=refit_cooks,
            reference_dds_ref_level=reference_dds_ref_level,
            meta_analysis_parameters=meta_analysis_parameters,
        )

        method_ref_padj, method_ref_lfc = get_padj_lfc_from_method(
            method_ref,
            method_ref_results_path,
            experiment_id,
            refit_cooks=refit_cooks,
            reference_dds_ref_level=reference_dds_ref_level,
            meta_analysis_parameters=meta_analysis_parameters,
        )

        # This does not handle meta-analysis for now
        # Check that the lfc and padj are Series and not dictionaries
        assert isinstance(method_test_padj, pd.Series), "Meta-analysis not supported"
        assert isinstance(method_test_lfc, pd.Series), "Meta-analysis not supported"
        assert isinstance(method_ref_padj, pd.Series), "Meta-analysis not supported"
        assert isinstance(method_ref_lfc, pd.Series), "Meta-analysis not supported"

        method_test_up_reg_genes = method_test_padj[
            (method_test_padj < padj_threshold)
            & (method_test_lfc > np.log(2) * log2fc_threshold)
        ].index
        method_test_down_reg_genes = method_test_padj[
            (method_test_padj < padj_threshold)
            & (method_test_lfc < -np.log(2) * log2fc_threshold)
        ].index

        method_ref_up_reg_genes = method_ref_padj[
            (method_ref_padj < padj_threshold)
            & (method_ref_lfc > np.log(2) * log2fc_threshold)
        ].index
        method_ref_down_reg_genes = method_ref_padj[
            (method_ref_padj < padj_threshold)
            & (method_ref_lfc < -np.log(2) * log2fc_threshold)
        ].index

        method_ref_all_genes = method_ref_padj.index

        # Check that the method_test are included
        assert set(method_test_padj.index).issubset(method_ref_padj.index)

        ax = axes.flatten()[i]

        confusion_matrix = build_33_confusion_matrix(
            set(method_test_up_reg_genes),
            set(method_test_down_reg_genes),
            set(method_ref_up_reg_genes),
            set(method_ref_down_reg_genes),
            set(method_ref_all_genes),
        )

        heatmap_matrix = build_33_heatmap_matrix(
            set(method_test_up_reg_genes),
            set(method_test_down_reg_genes),
            set(method_ref_up_reg_genes),
            set(method_ref_down_reg_genes),
            set(method_ref_all_genes),
        )
        sns.heatmap(
            heatmap_matrix,
            annot=confusion_matrix,
            fmt="g",
            vmin=0.0,
            vmax=1.0,
            cmap="flare_r",
            linewidths=1.0,
            annot_kws={"size": 14},
            cbar_ax=cbar_ax,
            ax=ax,
        )

        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, 0))

        ax.set_xlabel(NAME_MAPPING[method_ref], fontsize=15)
        ax.set_ylabel(NAME_MAPPING[method_test], fontsize=15)
        ax.set_xticklabels(["up-reg.", "none", "down-reg."], size=12)
        ax.set_yticklabels(["up-reg.", "none", "down-reg."], rotation=0, size=12)
        ax.set_title(f"{dataset_name}", fontsize=16)

    cbar_ax.tick_params(labelsize=14)

    save_file_path = Path(save_file_path)
    save_file_path = (
        save_file_path / "pan_cancer" / f"cross_table_{method_test}_vs_{method_ref}.pdf"
    )

    save_file_path = Path(save_file_path)
    save_file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file_path, bbox_inches="tight", transparent=True)
    plt.close()


def build_test_vs_ref_cross_table(
    method_test: str,
    method_ref: str,
    method_test_results_path: str | Path,
    method_ref_results_path: str | Path,
    cross_table_save_path: str | Path,
    dataset_name: TCGADatasetNames,
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    reference_dds_ref_level: tuple[str, str] | None = ("stage", "Advanced"),
    meta_analysis_parameters: list[MetaAnalysisParameter] | None = None,
    heterogeneity_method: str | None = None,
    heterogeneity_method_param: float | None = None,
    ref_with_heterogeneity: bool = False,
    log2fc_threshold: float = 2.0,
    padj_threshold: float = 0.05,
    **pydeseq2_kwargs: Any,
):
    """
    Build a cross table between a test method and a reference method.

    Parameters
    ----------
    method_test : str
        The test method.

    method_ref : str
        The reference method. Cannot be per center.

    method_test_results_path : Union[str, Path]
        The path to the test method results.

    method_ref_results_path : Union[str, Path]
        The path to the reference method results.

    cross_table_save_path : Union[str, Path]
        The path where to save the cross table.
        The following file structure will be created:
        ```
        <cross_table_save_path>
        └── <experiment_id>
            ├── cross_table_<method_test>_vs_<method_ref>.pdf
        ```
        The cross table will contain the confusion matrix between the test method(s)
        and the reference method.


    dataset_name : TCGADatasetNames
        The dataset name.

    small_samples : bool
        Whether to use small samples.

    small_genes : bool
        Whether to use small genes.

    only_two_centers : bool
        Whether to use only two centers.

    design_factors : str or list[str]
        Design factors to use.

    continuous_factors : list[str] or None
        Continuous factors to use.

    reference_dds_ref_level : tuple[str, str] or None
        The reference dds ref level to use.
        Necessary if the test method is PyDESeq2 per center or pooled.

    meta_analysis_parameters : list[MetaAnalysisParameter] or None
        The meta-analysis parameters to use.
        Necessary if the test method is Meta-analysis.

    heterogeneity_method : str or None
        The heterogeneity method to use.

    heterogeneity_method_param : float or None
        The heterogeneity method parameter to use.

    ref_with_heterogeneity : bool
        Whether the reference method has heterogeneity.

    log2fc_threshold : float
        The log2-fold change threshold to define up and down regulated genes.

    padj_threshold : float
        The adjusted p-value threshold to define differentially expressed genes.

    **pydeseq2_kwargs : Any
        Additional keyword arguments to pass to the PyDESeq2 and FedPyDESeq2
        methods.

    """
    # set experiment id
    test_experiment_id = get_experiment_id(
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
    if ref_with_heterogeneity:
        ref_experiment_id = get_experiment_id(
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
    else:
        ref_experiment_id = get_experiment_id(
            dataset_name=dataset_name,
            small_samples=small_samples,
            small_genes=small_genes,
            only_two_centers=only_two_centers,
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            **pydeseq2_kwargs,
        )
    refit_cooks = pydeseq2_kwargs.get("refit_cooks", True)
    method_test_padj, method_test_lfc = get_padj_lfc_from_method(
        method_test,
        method_test_results_path,
        test_experiment_id,
        refit_cooks=refit_cooks,
        reference_dds_ref_level=reference_dds_ref_level,
        meta_analysis_parameters=meta_analysis_parameters,
    )
    method_ref_padj, method_ref_lfc = get_padj_lfc_from_method(
        method_ref,
        method_ref_results_path,
        ref_experiment_id,
        refit_cooks=refit_cooks,
        reference_dds_ref_level=reference_dds_ref_level,
        meta_analysis_parameters=meta_analysis_parameters,
    )

    assert not (
        isinstance(method_ref_padj, dict)
    ), "Reference method should not be per center nor meta-analysis"
    assert not (
        isinstance(method_ref_lfc, dict)
    ), "Reference method should not be per center nor meta-analysis"

    save_file_path = (
        cross_table_save_path
        / test_experiment_id
        / f"cross_table_{method_test}_vs_{method_ref}.pdf"
    )
    build_cross_table(
        method_test_padj=method_test_padj,
        method_test_LFC=method_test_lfc,
        method_ref_padj=method_ref_padj,
        method_ref_LFC=method_ref_lfc,
        padj_threshold=padj_threshold,
        log2fc_threshold=log2fc_threshold,
        plot_title=(
            f"Cross table for {NAME_MAPPING[method_test]} vs {NAME_MAPPING[method_ref]}"
        ),
        save_file_path=save_file_path,
        method_test_name=NAME_MAPPING[method_test],
        method_ref_name=NAME_MAPPING[method_ref],
    )


def build_test_vs_ref_cross_tables(
    method_pairs: list[tuple[str, str]],
    method_results_paths: dict[str, Path],
    cross_table_save_path: str | Path,
    dataset_names: list[TCGADatasetNames],
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    reference_dds_ref_level: tuple[str, str] | None = ("stage", "Advanced"),
    meta_analysis_parameters: list[MetaAnalysisParameter] | None = None,
    heterogeneity_method: str | None = None,
    heterogeneity_method_param: float | None = None,
    ref_with_heterogeneity: bool | list[bool] = False,
    log2fc_threshold: float = 2.0,
    padj_threshold: float = 0.05,
    **pydeseq2_kwargs: Any,
):
    """
    Build cross tables for a list of test-reference method pairs.

    Parameters
    ----------
    method_pairs : list[tuple[str, str]]
        The list of test-reference method pairs.
        Note that pydeseq2_per_center cannot be a reference.

    method_results_paths : dict[str, Path]
        The dictionary of method results paths, mapping
        the method to the path where the results are stored.

    cross_table_save_path : Union[str, Path]
        The path where to save the cross table.
        The following file structure will be created for each
        experiment
        ```
        <cross_table_save_path>
        └── <experiment_id>
            ├── cross_table_<method_test>_vs_<method_ref>.pdf
        ```

    dataset_names : list[TCGADatasetNames]
        The list of dataset names.

    small_samples : bool
        Whether to use small samples.

    small_genes : bool
        Whether to use small genes.

    only_two_centers : bool
        Whether to use only two centers.

    design_factors : str or list[str]
        Design factors to use.

    continuous_factors : list[str] or None
        Continuous factors to use.

    reference_dds_ref_level : tuple[str, str] or None
        The reference dds ref level to use.
        Necessary if the test method is PyDESeq2 per center or pooled.

    meta_analysis_parameters : list[MetaAnalysisParameter] or None
        The meta-analysis parameters to use.
        Necessary if the test method is Meta-analysis.

    heterogeneity_method : str or None
        The heterogeneity method to use.

    heterogeneity_method_param : float or None
        The heterogeneity method parameter to use.

    ref_with_heterogeneity : bool or list[bool]
        Whether the reference method has heterogeneity.
        If a list, the length must be the same as the number of method pairs.

    log2fc_threshold : float
        The log2-fold change threshold to define up and down regulated genes.

    padj_threshold : float
        The adjusted p-value threshold to define differentially expressed genes.

    **pydeseq2_kwargs : Any
        Additional keyword arguments to pass to the PyDESeq2 and FedPyDESeq2.
        For example the contrast parameter.

    """
    if isinstance(ref_with_heterogeneity, bool):
        ref_with_heterogeneity = [ref_with_heterogeneity] * len(method_pairs)
    for dataset_name in dataset_names:
        for (method_test, method_ref), method_ref_with_heterogeneity in zip(
            method_pairs, ref_with_heterogeneity, strict=True
        ):
            logger.info(
                f"Building cross table for {dataset_name} ({method_test} vs "
                f"{method_ref})."
            )
            build_test_vs_ref_cross_table(
                method_test=method_test,
                method_ref=method_ref,
                method_test_results_path=method_results_paths[method_test],
                method_ref_results_path=method_results_paths[method_ref],
                cross_table_save_path=cross_table_save_path,
                dataset_name=cast(TCGADatasetNames, dataset_name),
                small_samples=small_samples,
                small_genes=small_genes,
                only_two_centers=only_two_centers,
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                reference_dds_ref_level=reference_dds_ref_level,
                meta_analysis_parameters=meta_analysis_parameters,
                heterogeneity_method=heterogeneity_method,
                heterogeneity_method_param=heterogeneity_method_param,
                ref_with_heterogeneity=method_ref_with_heterogeneity,
                log2fc_threshold=log2fc_threshold,
                padj_threshold=padj_threshold,
                **pydeseq2_kwargs,
            )


def build_cross_table(
    method_test_padj: dict[str, pd.Series] | pd.Series,
    method_test_LFC: dict[str, pd.Series] | pd.Series,
    method_ref_padj: pd.Series,
    method_ref_LFC: pd.Series,
    padj_threshold: float,
    log2fc_threshold: float,
    plot_title: str,
    save_file_path: str | Path,
    method_test_name: str | None,
    method_ref_name: str,
):
    """
    Build a 3x3 confusion matrix between a test method and a reference method.

    The confusion matrix is normalized by columns, i.e., w.r.t. the reference method.

    Parameters
    ----------
    method_test_padj : pd.Series or dict[str, pd.Series]
        The adjusted p-values of the tested method(s).

    method_test_LFC : pd.Series or dict[str, pd.Series]
        The log-fold changes of the tested method(s), *in natural scale*.

    method_ref_padj : pd.Series
        The adjusted p-values of the reference method.

    method_ref_LFC : pd.Series
        The log-fold changes of the reference method, *in natural scale*.

    padj_threshold : float
        The adjusted p-value threshold to consider a gene as differentially expressed.

    log2fc_threshold : float
        The log2-fold change threshold to consider a gene as differentially expressed.

    plot_title : str
        The title of the plot.

    save_file_path :str or Path
        The path where to save the plot.

    method_test_name : str or None
        The name of the tested method. Only necessary
        if method_test_padj is a pd.Series.

    method_ref_name : str
        The name of the reference method.


    """
    plt.clf()
    if isinstance(method_test_padj, dict):
        assert isinstance(method_test_LFC, dict)
        assert set(method_test_padj.keys()) == set(method_test_LFC.keys())
        n_methods = len(method_test_padj)
        fig, axes = plt.subplots(1, n_methods, figsize=(n_methods * 8, 8), sharey=True)
        for i, method_id in enumerate(sorted(method_test_padj.keys())):
            # Extract the meta-analysis submethod name in a more readable format
            submethod_name = method_id.split(", ")[1:]
            submethod_name = "_".join(
                [param for param in submethod_name if param != "None"]
            )
            center_padj = method_test_padj[method_id]
            build_cross_table_on_ax(
                method_test_padj=center_padj,
                method_test_LFC=method_test_LFC[method_id],
                method_ref_padj=method_ref_padj,
                method_ref_LFC=method_ref_LFC,
                padj_threshold=padj_threshold,
                log2fc_threshold=log2fc_threshold,
                plot_title=None,
                method_test_name=NAME_MAPPING[submethod_name],
                method_ref_name=method_ref_name,
                ax=axes[i],
            )
    else:
        assert method_test_name is not None
        fig, ax = plt.subplots(figsize=(8, 8))
        build_cross_table_on_ax(
            method_test_padj=method_test_padj,
            method_test_LFC=method_test_LFC,
            method_ref_padj=method_ref_padj,
            method_ref_LFC=method_ref_LFC,
            padj_threshold=padj_threshold,
            log2fc_threshold=log2fc_threshold,
            plot_title=plot_title,
            method_test_name=method_test_name,
            method_ref_name=method_ref_name,
            ax=ax,
        )

    plt.tight_layout()
    save_file_path = Path(save_file_path)
    save_file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file_path, transparent=True)
    plt.close()


def build_cross_table_on_ax(
    method_test_padj: pd.Series,
    method_test_LFC: pd.Series,
    method_ref_padj: pd.Series,
    method_ref_LFC: pd.Series,
    padj_threshold: float,
    log2fc_threshold: float,
    plot_title: str,
    method_test_name: str,
    method_ref_name: str,
    ax: plt.Axes,
):
    """
    Build a 3x3 confusion matrix between a test method and a reference method.

    The confusion matrix is normalized by columns, i.e., w.r.t. the reference method.

    Parameters
    ----------
    method_test_padj : pd.Series
        The adjusted p-values of the tested method.

    method_test_LFC : pd.Series
        The log-fold changes of the tested method, *in natural scale*.

    method_ref_padj : pd.Series
        The adjusted p-values of the reference method.

    method_ref_LFC : pd.Series
        The log-fold changes of the reference method, *in natural scale*.

    padj_threshold : float
        The adjusted p-value threshold to consider a gene as differentially expressed.

    log2fc_threshold : float
        The log2-fold change threshold to consider a gene as differentially expressed.

    plot_title : str
        The title of the plot.

    method_test_name : str
        The name of the tested method.

    method_ref_name : str
        The name of the reference method.

    ax : plt.Axes
        The matplotlib axes where to plot the confusion matrix

    """
    method_test_up_reg_genes = method_test_padj[
        (method_test_padj < padj_threshold)
        & (method_test_LFC > np.log(2) * log2fc_threshold)
    ].index
    method_test_down_reg_genes = method_test_padj[
        (method_test_padj < padj_threshold)
        & (method_test_LFC < -np.log(2) * log2fc_threshold)
    ].index

    # Check that the method_test are included
    assert set(method_test_padj.index).issubset(method_ref_padj.index)

    method_ref_up_reg_genes = method_ref_padj[
        (method_ref_padj < padj_threshold)
        & (method_ref_LFC > np.log(2) * log2fc_threshold)
    ].index
    method_ref_down_reg_genes = method_ref_padj[
        (method_ref_padj < padj_threshold)
        & (method_ref_LFC < -np.log(2) * log2fc_threshold)
    ].index
    method_ref_all_genes = method_ref_padj.index

    matrix = build_33_confusion_matrix(
        set(method_test_up_reg_genes),
        set(method_test_down_reg_genes),
        set(method_ref_up_reg_genes),
        set(method_ref_down_reg_genes),
        set(method_ref_all_genes),
    )

    heatmap_matrix = build_33_heatmap_matrix(
        set(method_test_up_reg_genes),
        set(method_test_down_reg_genes),
        set(method_ref_up_reg_genes),
        set(method_ref_down_reg_genes),
        set(method_ref_all_genes),
    )

    sns.heatmap(
        heatmap_matrix,
        annot=matrix,
        cmap="viridis",
        fmt="g",
        linewidths=1.0,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        annot_kws={"size": 14},
    )
    ax.set_xlabel(method_ref_name, fontsize=16)
    ax.set_ylabel(method_test_name, fontsize=16)
    ax.set_xticklabels(["up-reg.", "none", "down-reg."], size=14)
    ax.set_yticklabels(["up-reg.", "none", "down-reg."], rotation=0, size=14)
    if plot_title is not None:
        ax.set_title(plot_title, fontsize=16)


def build_33_confusion_matrix(
    method_1_up_genes: set,
    method_1_down_genes: set,
    method_2_up_genes: set,
    method_2_down_genes: set,
    all_genes: set,
):
    """
    Build a 3x3 confusion matrix.

    Parameters
    ----------
    method_1_up_genes : set
        The up-regulated genes of the first method.

    method_1_down_genes : set
        The down-regulated genes of the first method.

    method_2_up_genes : set
        The up-regulated genes of the second method.

    method_2_down_genes : set
        The down-regulated genes of the second method.

    all_genes : set
        The set of all genes.

    Returns
    -------
    np.ndarray
        The confusion matrix.
    """
    matrix = np.zeros((3, 3))

    method_1_none_genes = all_genes - method_1_up_genes - method_1_down_genes
    method_2_none_genes = all_genes - method_2_up_genes - method_2_down_genes

    matrix[0, 0] = len(method_1_up_genes.intersection(method_2_up_genes))
    matrix[0, 1] = len(method_1_up_genes.intersection(method_2_none_genes))
    matrix[0, 2] = len(method_1_up_genes.intersection(method_2_down_genes))

    matrix[1, 0] = len(method_1_none_genes.intersection(method_2_up_genes))
    matrix[1, 1] = len(method_1_none_genes.intersection(method_2_none_genes))
    matrix[1, 2] = len(method_1_none_genes.intersection(method_2_down_genes))

    matrix[2, 0] = len(method_1_down_genes.intersection(method_2_up_genes))
    matrix[2, 1] = len(method_1_down_genes.intersection(method_2_none_genes))
    matrix[2, 2] = len(method_1_down_genes.intersection(method_2_down_genes))

    return matrix


def build_33_heatmap_matrix(
    method_test_up_genes: set,
    method_test_down_genes: set,
    method_ref_up_genes: set,
    method_ref_down_genes: set,
    all_genes: set,
):
    """
    Build a 3x3 heatmap matrix.

    This heatmap matrix computest the following formula:
    len(
        intersection(method_test_{up/none/down}_genes, method_ref_{up/none/down}_genes)
        )
    / math.sqrt(
        len(method_ref_{up/none/down}_genes)*len(method_ref_{up/none/down}_genes)
        )

    This is basically computing a form of correlation, with a reference
    method.

    Parameters
    ----------
    method_test_up_genes : set
        The up-regulated genes of the first method.

    method_test_down_genes : set
        The down-regulated genes of the first method.

    method_ref_up_genes : set
        The up-regulated genes of the second method.

    method_ref_down_genes : set
        The down-regulated genes of the second method.

    all_genes : set
        The set of all genes.

    Returns
    -------
    np.ndarray
        The heatmap matrix.
    """
    matrix = build_33_confusion_matrix(
        method_test_up_genes,
        method_test_down_genes,
        method_ref_up_genes,
        method_ref_down_genes,
        all_genes,
    )

    method_ref_none_genes = all_genes - method_ref_up_genes - method_ref_down_genes

    heatmap_matrix = np.zeros((3, 3))
    heatmap_matrix[0, 0] = matrix[0, 0] / len(method_ref_up_genes)
    heatmap_matrix[0, 1] = matrix[0, 1] / math.sqrt(
        len(method_ref_up_genes) * len(method_ref_none_genes)
    )
    heatmap_matrix[0, 2] = matrix[0, 2] / math.sqrt(
        len(method_ref_up_genes) * len(method_ref_down_genes)
    )

    heatmap_matrix[1, 0] = matrix[1, 0] / math.sqrt(
        len(method_ref_up_genes) * len(method_ref_none_genes)
    )
    heatmap_matrix[1, 1] = matrix[1, 1] / len(method_ref_none_genes)
    heatmap_matrix[1, 2] = matrix[1, 2] / math.sqrt(
        len(method_ref_down_genes) * len(method_ref_none_genes)
    )

    heatmap_matrix[2, 0] = matrix[2, 0] / math.sqrt(
        len(method_ref_up_genes) * len(method_ref_down_genes)
    )
    heatmap_matrix[2, 1] = matrix[2, 1] / math.sqrt(
        len(method_ref_none_genes) * len(method_ref_down_genes)
    )
    heatmap_matrix[2, 2] = matrix[2, 2] / len(method_ref_down_genes)

    return heatmap_matrix
