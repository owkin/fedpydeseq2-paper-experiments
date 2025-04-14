import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fedpydeseq2.core.utils.stat_utils import build_contrast_vector
from fedpydeseq2_datasets.utils import get_ground_truth_dds_name
from fedpydeseq2_datasets.utils import get_valid_centers_from_subfolders_file

from paper_experiments.run_dge_gsea_methods.meta_analysis_tcga_pipe import (
    get_meta_analysis_id,
)
from paper_experiments.utils.constants import MetaAnalysisParameter

NAME_MAPPING = {
    "pydeseq2_per_center": "PyDESeq2 (per center)",
    "pydeseq2": "PyDESeq2",
    "fedpydeseq2_simulated": "FedPyDESeq2 (simulated)",
    "fedpydeseq2_remote": "FedPyDESeq2",
    "meta_analysis": "Meta-analysis",
    "fixed_effect": "Fixed effects",
    "random_effect_dl": "Random effects\n(DerSimonian-Laird)",
    "random_effect_iterated": "Random effects\n(iterated)",
    "pydeseq2_largest": "PyDESeq2\n(largest cohort)",
    "pvalue_combination_fisher": "Fisher",
    "pvalue_combination_stouffer": "Stouffer",
}

SCORING_FUNCTIONS_YLABELS: dict[str, str] = {
    "precision_0.05_2.0": "Precision",
    "sensitivity_0.05_2.0": "Recall",
    "f1_score_0.05_2.0": "F1 score",
    "pearson_correlation_pvalues": "Pearson correlation of -log10(p-values)",
    "pearson_correlation_lfcs": "Pearson correlation of log fold changes",
    "pearson_correlation_pvalues_0.05": (
        "Pearson correlation of -log10(p-values) \n (padj < 0.05)"
    ),
    "pearson_correlation_lfcs_0.05": (
        "Pearson correlation of log fold changes \n (padj < 0.05)"
    ),
    "pearson_correlation_pvalues_7": (
        "Pearson correlation of -log10(p-values) \n (padj clipped to 1e-7)"
    ),
    "pearson_correlation_pvalues_10": (
        "Pearson correlation of -log10(p-values) \n (padj clipped to 1e-10)"
    ),
    "pearson_correlation_pvalues_12": (
        "Pearson correlation of -log10(p-values) \n (padj clipped to 1e-12)"
    ),
    "pearson_correlation_pvalues_15": (
        "Pearson correlation of -log10(p-values) \n (padj clipped to 1e-15)"
    ),
}


def process_method_name(method_name: str) -> str:
    """
    Make method name more readable.

    Parameters
    ----------
    method_name : str
        The method test name.

    Returns
    -------
    str
        The processed method name.

    """
    if method_name.startswith("Meta-analysis"):
        # Extract the meta-analysis submethod name in a more readable format
        submethod_name = "_".join(
            [param for param in method_name.split(", ")[1:] if param != "None"]
        )
        return NAME_MAPPING[submethod_name]
    else:
        return NAME_MAPPING[method_name]


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


def get_de_genes(
    method_padj: pd.Series,
    method_lfc: pd.Series,
    padj_threshold: float | None,
    log2fc_threshold: float | None,
) -> pd.Index:
    """
    Get the differentially expressed genes.

    We define the differentially expressed genes as the genes with an adjusted p-value
    below a certain threshold and an absolute log fold change above a certain threshold.

    Parameters
    ----------
    method_padj : pd.Series
        The adjusted p-values, indexed by gene names.

    method_lfc : pd.Series
        The log fold changes, indexed by gene names, *in natural scale*.

    padj_threshold : float or None
        The adjusted p-value threshold.

    log2fc_threshold : float or None
        The log2 fold change threshold.

    Returns
    -------
    pd.Index
        The differentially expressed genes.

    """
    # Initialize a boolean series to True
    condition = pd.Series(True, index=method_padj.index)
    if padj_threshold is not None:
        condition &= method_padj < padj_threshold

    if log2fc_threshold is not None:
        condition &= np.abs(method_lfc) > np.log(2) * log2fc_threshold
    method_diff_genes = method_padj[condition].index
    return method_diff_genes
