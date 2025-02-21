from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from fedpydeseq2_datasets.constants import TCGADatasetNames
from fedpydeseq2_datasets.utils import get_experiment_id
from loguru import logger
from matplotlib import pyplot as plt

from paper_experiments.figures.utils import SCORING_FUNCTIONS_YLABELS
from paper_experiments.figures.utils import get_de_genes
from paper_experiments.figures.utils import get_padj_lfc_from_method
from paper_experiments.figures.utils import process_method_name
from paper_experiments.utils.constants import MetaAnalysisParameter


def sensitivity(
    method_test_padj: pd.Series,
    method_test_lfc: pd.Series,
    method_ref_padj: pd.Series,
    method_ref_lfc: pd.Series,
    padj_threshold: float | None,
    log2fc_threshold: float | None,
) -> float:
    """
    Compute the number of recovered positives out of all true positives. A.k.a. recall.

    By recovered positives, we mean the fraction of differentially expressed genes
    found by the test method that are also found by the reference method, w.r.t. the
    differentially expressed genes found by the reference method.

    We define the differentially expressed genes as the genes with an adjusted p-value
    below a certain threshold and an absolute log fold change above a certain threshold.

    Parameters
    ----------
    method_test_padj : pd.Series
        The adjusted p-values of the test method, indexed by gene names.

    method_test_lfc : pd.Series
        The log fold changes of the test method, indexed by gene names,
        *in natural scale*.

    method_ref_padj : pd.Series
        The adjusted p-values of the reference method, indexed by gene names.

    method_ref_lfc : pd.Series
        The log fold changes of the reference method, indexed by gene names,
        *in natural scale*.

    padj_threshold : float or None
        The adjusted p-value threshold.

    log2fc_threshold : float or None
        The log2 fold change threshold.

    Returns
    -------
    float
        The sensitivity score, also known as recall.

    """
    method_test_diff_genes = get_de_genes(
        method_test_padj, method_test_lfc, padj_threshold, log2fc_threshold
    )
    method_ref_diff_genes = get_de_genes(
        method_ref_padj, method_ref_lfc, padj_threshold, log2fc_threshold
    )

    true_positives = len(set(method_test_diff_genes) & set(method_ref_diff_genes))
    false_negatives = len(set(method_ref_diff_genes) - set(method_test_diff_genes))

    return true_positives / (true_positives + false_negatives)


def precision(
    method_test_padj: pd.Series,
    method_test_lfc: pd.Series,
    method_ref_padj: pd.Series,
    method_ref_lfc: pd.Series,
    padj_threshold: float | None,
    log2fc_threshold: float | None,
) -> float:
    """
    Compute the number of true positives out of all predicted positives.

    By true positives, we mean the fraction of differentially expressed genes
    found by the test method that are also found by the reference method, w.r.t. the
    differentially expressed genes found by the reference method.

    We define the differentially expressed genes as the genes with an adjusted p-value
    below a certain threshold and an absolute log fold change above a certain threshold.

    Parameters
    ----------
    method_test_padj : pd.Series
        The adjusted p-values of the test method, indexed by gene names.

    method_test_lfc : pd.Series
        The log fold changes of the test method, indexed by gene names,
        *in natural scale*.

    method_ref_padj : pd.Series
        The adjusted p-values of the reference method, indexed by gene names.

    method_ref_lfc : pd.Series
        The log fold changes of the reference method, indexed by gene names,
        *in natural scale*.

    padj_threshold : float or None
        The adjusted p-value threshold.

    log2fc_threshold : float or None
        The log2 fold change threshold.

    Returns
    -------
    float
        The precision score.

    """
    method_test_diff_genes = get_de_genes(
        method_test_padj, method_test_lfc, padj_threshold, log2fc_threshold
    )
    method_ref_diff_genes = get_de_genes(
        method_ref_padj, method_ref_lfc, padj_threshold, log2fc_threshold
    )

    true_positives = len(set(method_test_diff_genes) & set(method_ref_diff_genes))
    false_positives = len(set(method_test_diff_genes) - set(method_ref_diff_genes))

    return true_positives / (true_positives + false_positives)


def f1_score(
    method_test_padj: pd.Series,
    method_test_lfc: pd.Series,
    method_ref_padj: pd.Series,
    method_ref_lfc: pd.Series,
    padj_threshold: float | None,
    log2fc_threshold: float | None,
):
    """
    Compute the F1 score.

    The F1 score is the harmonic mean of the precision and recall.

    Parameters
    ----------
    method_test_padj : pd.Series
        The adjusted p-values of the test method, indexed by gene names.

    method_test_lfc : pd.Series
        The log fold changes of the test method, indexed by gene names,
        *in natural scale*.

    method_ref_padj : pd.Series
        The adjusted p-values of the reference method, indexed by gene names.

    method_ref_lfc : pd.Series
        The log fold changes of the reference method, indexed by gene names,
        *in natural scale*.

    padj_threshold : float or None
        The adjusted p-value threshold.

    log2fc_threshold : float or None
        The log2 fold change threshold.

    Returns
    -------
    float
        The F1 score.

    """
    method_test_diff_genes = get_de_genes(
        method_test_padj, method_test_lfc, padj_threshold, log2fc_threshold
    )
    method_ref_diff_genes = get_de_genes(
        method_ref_padj, method_ref_lfc, padj_threshold, log2fc_threshold
    )

    true_positives = len(set(method_test_diff_genes) & set(method_ref_diff_genes))
    false_positives = len(set(method_test_diff_genes) - set(method_ref_diff_genes))
    false_negatives = len(set(method_ref_diff_genes) - set(method_test_diff_genes))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return 2 * (precision * recall) / (precision + recall)


def pearson_correlation_pvalues(
    method_test_padj: pd.Series,
    method_test_lfc: pd.Series,
    method_ref_padj: pd.Series,
    method_ref_lfc: pd.Series,
    padj_threshold: float | None = None,
    log2fc_threshold: float | None = None,
    padj_lower_bound: float = 1e-10,
) -> float:
    """
    Compute the Pearson correlation of the -log10(p-values).

    Parameters
    ----------
    method_test_padj : pd.Series
        The adjusted p-values of the test method, indexed by gene names.

    method_test_lfc : pd.Series
        The log fold changes of the test method, indexed by gene names,
        *in natural scale*.

    method_ref_padj : pd.Series
        The adjusted p-values of the reference method, indexed by gene names.

    method_ref_lfc : pd.Series
        The log fold changes of the reference method, indexed by gene names,
        *in natural scale*.

    padj_threshold : float or None
        The adjusted p-value threshold. If not None, we
        will restrict the genes to those whose
        reference p-value is below this threshold.

    log2fc_threshold : float or None
        The log2 fold change threshold. If not None, we
        will restrict the genes to those whose reference
        log2 fold change is above this threshold.

    padj_lower_bound : float
        The lower bound for the adjusted p-values. If a p-value is below this
        threshold, we will clip it to this value.

    Returns
    -------
    float
        The Pearson correlation of the -log10(p-values).

    """
    reference_genes = get_de_genes(
        method_ref_padj, method_ref_lfc, padj_threshold, log2fc_threshold
    )
    # Drop nan values
    common_genes = (
        method_ref_padj.loc[reference_genes]
        .dropna()
        .index.intersection(method_test_padj.dropna().index)
    )
    # Get the arrays of pvalues
    pvalues_test = method_test_padj.loc[common_genes].to_numpy().reshape(1, -1)
    pvalues_ref = method_ref_padj.loc[common_genes].to_numpy().reshape(1, -1)
    # Clip pvalues to avoid log(0)
    where_too_small_test = pvalues_test < padj_lower_bound
    where_too_small_ref = pvalues_ref < padj_lower_bound
    if where_too_small_test.any():
        logger.warning(
            f"Found {where_too_small_test.sum()} p-values below {padj_lower_bound} "
            "for the test method.\n"
            f"Clipping them to {padj_lower_bound}."
        )
    if where_too_small_ref.any():
        logger.warning(
            f"Found {where_too_small_ref.sum()} p-values below {padj_lower_bound} "
            "for the reference method.\n"
            f"Clipping them to {padj_lower_bound}."
        )
    pvalues_test = np.clip(pvalues_test, padj_lower_bound, 1)
    pvalues_ref = np.clip(pvalues_ref, padj_lower_bound, 1)
    # transform as -log10(pvalue)
    pvalues_test = -np.log10(pvalues_test)
    pvalues_ref = -np.log10(pvalues_ref)
    # Compute the correlation
    return np.corrcoef(pvalues_test, pvalues_ref)[0, 1]


def pearson_correlation_lfcs(
    method_test_padj: pd.Series,
    method_test_lfc: pd.Series,
    method_ref_padj: pd.Series,
    method_ref_lfc: pd.Series,
    padj_threshold: float | None = None,
    log2fc_threshold: float | None = None,
) -> float:
    """
    Compute the Pearson correlation of the log fold changes.

    Parameters
    ----------
    method_test_padj : pd.Series
        The adjusted p-values of the test method, indexed by gene names.

    method_test_lfc : pd.Series
        The log fold changes of the test method, indexed by gene names,
        *in natural scale*.

    method_ref_padj : pd.Series
        The adjusted p-values of the reference method, indexed by gene names.

    method_ref_lfc : pd.Series
        The log fold changes of the reference method, indexed by gene names,
        *in natural scale*.

    padj_threshold : float or None
        The adjusted p-value threshold. If not None, we
        will restrict the genes to those whose
        reference p-value is below this threshold.

    log2fc_threshold : float or None
        The log2 fold change threshold. If not None, we
        will restrict the genes to those whose reference
        log2 fold change is above this threshold.

    Returns
    -------
    float
        The Pearson correlation of the LFC.

    """
    # Compute common genes where the pvalue is not NaN
    reference_genes = get_de_genes(
        method_ref_padj, method_ref_lfc, padj_threshold, log2fc_threshold
    )
    common_genes = method_test_lfc.dropna().index.intersection(
        reference_genes.intersection(method_ref_lfc.dropna().index)
    )

    # Get the arrays of lfcs
    lfcs_test = method_test_lfc.loc[common_genes].to_numpy().reshape(1, -1)
    lfcs_ref = method_ref_lfc.loc[common_genes].to_numpy().reshape(1, -1)
    # Compute the correlation
    return np.corrcoef(lfcs_test, lfcs_ref)[0, 1]


SCORING_FUNCTIONS: dict[str, Callable] = {
    "sensitivity_0.05_2.0": partial(
        sensitivity, padj_threshold=0.05, log2fc_threshold=2.0
    ),
    "precision_0.05_2.0": partial(precision, padj_threshold=0.05, log2fc_threshold=2.0),
    "f1_score_0.05_2.0": partial(f1_score, padj_threshold=0.05, log2fc_threshold=2.0),
    "pearson_correlation_pvalues": pearson_correlation_pvalues,
    "pearson_correlation_pvalues_7": partial(
        pearson_correlation_pvalues, padj_lower_bound=1e-7
    ),
    "pearson_correlation_pvalues_10": partial(
        pearson_correlation_pvalues, padj_lower_bound=1e-10
    ),
    "pearson_correlation_pvalues_12": partial(
        pearson_correlation_pvalues, padj_lower_bound=1e-12
    ),
    "pearson_correlation_pvalues_15": partial(
        pearson_correlation_pvalues, padj_lower_bound=1e-15
    ),
    "pearson_correlation_lfcs": pearson_correlation_lfcs,
    "pearson_correlation_pvalues_0.05": partial(
        pearson_correlation_pvalues, padj_threshold=0.05
    ),
    "pearson_correlation_lfcs_0.05": partial(
        pearson_correlation_lfcs, padj_threshold=0.05
    ),
}


def build_heterogeneity_grid_plot(
    methods_test: list[str],
    method_ref: str,
    methods_test_results_paths: dict[str, str | Path],
    method_ref_results_path: str | Path,
    save_file_path: str | Path,
    dataset_names: list[TCGADatasetNames],
    heterogeneity_method_params: list[float],
    scoring_function_names: list[str],
    heterogeneity_method: str = "binomial",
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    reference_dds_ref_level: tuple[str, str] | None = ("stage", "Advanced"),
    meta_analysis_parameters: list[MetaAnalysisParameter] | None = None,
    **pydeseq2_kwargs: Any,
):
    """
    Make a grid of barplots to summarize heterogeneity experiments.

    Represents barplots side by side for each dataset and each scoring function, with a
    common legend.

    Parameters
    ----------
    methods_test : str
        The tested method.

    method_ref : str
        The reference method.

    methods_test_results_paths : str or Path
        The path to the tested method results.

    method_ref_results_path : str or Path
        The path to the reference method results.

    save_file_path : str or Path
        The path where to save the plot.

    dataset_names : list[TCGADatasetNames]
        The list of dataset to include in the figure.

    heterogeneity_method_params : list[float]
        The heterogeneity method parameters to use.

    scoring_function_names : list[str]
        The scoring functions to plot. Must be keys in SCORING_FUNCTIONS.

    heterogeneity_method : str
        The heterogeneity method to use.

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

    **pydeseq2_kwargs : Any
        Additional keyword arguments to pass to the PyDESeq2 and FedPyDESeq2
        methods.
    """
    plt.clf()
    sns.set_theme()
    sns.set_style("whitegrid")

    num_datasets, num_scores = len(dataset_names), len(scoring_function_names)

    fig, axes = plt.subplots(
        num_scores,
        num_datasets,
        figsize=(6 * num_scores, 6 * num_datasets),
        constrained_layout=True,
    )

    logger.info("Building heterogeneity plot grid.")

    for i, dataset_name in enumerate(dataset_names):
        experiment_ids = [
            get_experiment_id(
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
            for heterogeneity_method_param in heterogeneity_method_params
        ]

        global_experiment_id = get_experiment_id(
            dataset_name=dataset_name,
            small_samples=small_samples,
            small_genes=small_genes,
            only_two_centers=only_two_centers,
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            heterogeneity_method=heterogeneity_method,
            heterogeneity_method_param=None,
            **pydeseq2_kwargs,
        )

        reference_experiment_id = get_experiment_id(
            dataset_name=dataset_name,
            small_samples=small_samples,
            small_genes=small_genes,
            only_two_centers=only_two_centers,
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            **pydeseq2_kwargs,
        )

        refit_cooks = pydeseq2_kwargs.get("refit_cooks", True)

        methods_test_padj_lfc: dict[str, list[tuple[pd.Series, pd.Series]]] = {}

        for heterogeneity_id in range(len(experiment_ids)):
            for method_test in methods_test:
                method_test_padj, method_test_lfc = get_padj_lfc_from_method(
                    method_test,
                    methods_test_results_paths[method_test],
                    experiment_ids[heterogeneity_id],
                    refit_cooks=refit_cooks,
                    reference_dds_ref_level=reference_dds_ref_level,
                    meta_analysis_parameters=meta_analysis_parameters,
                )

                if isinstance(method_test_padj, dict) and isinstance(
                    method_test_lfc, dict
                ):
                    if heterogeneity_id == 0:
                        for method_test_str in method_test_padj.keys():
                            methods_test_padj_lfc[method_test_str] = []
                    for method_test_str in method_test_padj.keys():
                        methods_test_padj_lfc[method_test_str].append(
                            (
                                method_test_padj[method_test_str],
                                method_test_lfc[method_test_str],
                            )
                        )
                else:
                    assert isinstance(method_test_padj, pd.Series)
                    assert isinstance(method_test_lfc, pd.Series)
                    method_test_str = method_test
                    if heterogeneity_id == 0:
                        methods_test_padj_lfc[method_test_str] = []
                    methods_test_padj_lfc[method_test_str].append(
                        (method_test_padj, method_test_lfc)
                    )

        method_ref_padj, method_ref_lfc = get_padj_lfc_from_method(
            method_ref,
            method_ref_results_path,
            reference_experiment_id,
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

        for j, scoring_function_name in enumerate(scoring_function_names):
            # Compute scores
            scoring_function = SCORING_FUNCTIONS[scoring_function_name]
            scores: dict[str, list[float]] = {
                method_test_str: [
                    scoring_function(
                        method_test_padj,
                        method_test_lfc,
                        method_ref_padj,
                        method_ref_lfc,
                    )
                    for method_test_padj, method_test_lfc in methods_test_padj_lfc[
                        method_test_str
                    ]
                ]
                for method_test_str in methods_test_padj_lfc
            }

            ax = axes[j, i]
            lines = []

            for method_test, scores_list in scores.items():
                for k, score in enumerate(scores_list):
                    lines.append(
                        {
                            "method_test_name": process_method_name(method_test),
                            # Here we invert the heterogeneity level
                            "heterogeneity level": 1.0 - heterogeneity_method_params[k],
                            "score": score,
                        }
                    )
            df = pd.DataFrame(lines)
            # Type the columns
            df = df.astype(
                {
                    "method_test_name": "string",
                    "heterogeneity level": "float",
                    "score": "float",
                }
            )

            sns.barplot(
                df,
                x="method_test_name",
                y="score",
                hue="heterogeneity level",
                palette="viridis",
                ax=ax,
            )

            ax.set_title(f"{dataset_name}", fontsize=16)
            ax.set_ylabel(
                f"{SCORING_FUNCTIONS_YLABELS[scoring_function_name]}", fontsize=15
            )
            # Set limits for the y axis
            ax.set_ylim(0, 1)
            ax.set_xlabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            # Increase y ticks labels font size
            ax.tick_params(axis="y", labelsize=12)

    # Create a common legend
    lines, labels = axes[0, 0].get_legend_handles_labels()
    # Remove subplot legends
    for ax in axes.flat:
        ax.get_legend().remove()

    legend = fig.legend(
        lines,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, 1.05),
        title="Heterogeneity level",
        fontsize=15,
        title_fontsize=15,
        ncol=5,
    )

    plt.setp(legend.get_title(), fontweight="bold")

    save_file_path = Path(save_file_path)
    save_file_path = Path(
        save_file_path / global_experiment_id / "heterogeneity_grid_plot.pdf"
    )

    save_file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file_path, bbox_inches="tight")
    plt.close()


def build_test_vs_ref_heterogeneity_plot(
    methods_test: list[str],
    method_ref: str,
    methods_results_path: dict[str, str | Path],
    heterogeneity_plot_save_path: str | Path,
    plot_title: str,
    dataset_name: TCGADatasetNames,
    heterogeneity_method_params: list[float],
    scoring_function_name: str,
    heterogeneity_method: str = "binomial",
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    reference_dds_ref_level: tuple[str, str] | None = ("stage", "Advanced"),
    meta_analysis_parameters: list[MetaAnalysisParameter] | None = None,
    **pydeseq2_kwargs: Any,
):
    """
    Build a cross table between a test method and a reference method.

    Parameters
    ----------
    methods_test : list[str]
        The test methods.

    method_ref : str
        The reference method. Cannot be per center.

    methods_results_path : dict[str, Union[str, Path]]
        The path to the results of the methods.

    heterogeneity_plot_save_path : Union[str, Path]
        The path where to save the heterogeneity plot.
        The following file structure will be created:
        ```
        <heterogeneity_plot_save_path>
        └── <global_experiment_id>
            ├── heterogeneity_plot_<scoring_function_name>
                _test_<methods_test>_ref_<method_ref>.pdf
        ```

    plot_title : str
        The plot title.

    dataset_name : TCGADatasetNames
        The dataset name.

    heterogeneity_method : str
        The heterogeneity method to use.

    heterogeneity_method_params : list[float]
        The heterogeneity method parameters to use.

    scoring_function_name : str
        The scoring function name.
        Is a key in SCORING_FUNCTIONS.

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

    **pydeseq2_kwargs : Any
        Additional keyword arguments to pass to the PyDESeq2 and FedPyDESeq2
        methods.

    """
    # Get experiment id for each heterogeneity method param
    experiment_ids = [
        get_experiment_id(
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
        for heterogeneity_method_param in heterogeneity_method_params
    ]

    global_experiment_id = get_experiment_id(
        dataset_name=dataset_name,
        small_samples=small_samples,
        small_genes=small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        heterogeneity_method=heterogeneity_method,
        heterogeneity_method_param=None,
        **pydeseq2_kwargs,
    )

    reference_experiment_id = get_experiment_id(
        dataset_name=dataset_name,
        small_samples=small_samples,
        small_genes=small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        **pydeseq2_kwargs,
    )

    refit_cooks = pydeseq2_kwargs.get("refit_cooks", True)

    methods_test_padj_lfc: dict[str, list[tuple[pd.Series, pd.Series]]] = {}

    for heterogeneity_id in range(len(experiment_ids)):
        for method_test in methods_test:
            method_test_padj, method_test_lfc = get_padj_lfc_from_method(
                method_test,
                methods_results_path[method_test],
                experiment_ids[heterogeneity_id],
                refit_cooks=refit_cooks,
                reference_dds_ref_level=reference_dds_ref_level,
                meta_analysis_parameters=meta_analysis_parameters,
            )

            if isinstance(method_test_padj, dict) and isinstance(method_test_lfc, dict):
                if heterogeneity_id == 0:
                    for method_test_str in method_test_padj.keys():
                        methods_test_padj_lfc[method_test_str] = []
                for method_test_str in method_test_padj.keys():
                    methods_test_padj_lfc[method_test_str].append(
                        (
                            method_test_padj[method_test_str],
                            method_test_lfc[method_test_str],
                        )
                    )
            else:
                assert isinstance(method_test_padj, pd.Series)
                assert isinstance(method_test_lfc, pd.Series)
                method_test_str = method_test
                if heterogeneity_id == 0:
                    methods_test_padj_lfc[method_test_str] = []
                methods_test_padj_lfc[method_test_str].append(
                    (method_test_padj, method_test_lfc)
                )
    logger.info(f"methods_test_padj_lfc keys: {methods_test_padj_lfc.keys()}")

    method_ref_padj, method_ref_lfc = get_padj_lfc_from_method(
        method_ref,
        methods_results_path[method_ref],
        reference_experiment_id,
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

    heterogeneity_plot_name = get_heterogeneity_plot_name(
        methods_test=methods_test,
        method_ref=method_ref,
        scoring_function_name=scoring_function_name,
    )

    save_file_path = (
        heterogeneity_plot_save_path
        / global_experiment_id
        / f"{heterogeneity_plot_name}.pdf"
    )

    # Compute scores
    scoring_function = SCORING_FUNCTIONS[scoring_function_name]
    scores: dict[str, list[float]] = {
        method_test_str: [
            scoring_function(
                method_test_padj, method_test_lfc, method_ref_padj, method_ref_lfc
            )
            for method_test_padj, method_test_lfc in methods_test_padj_lfc[
                method_test_str
            ]
        ]
        for method_test_str in methods_test_padj_lfc
    }

    logger.info(f"scores keys: {scores.keys()}")

    make_heterogeneity_plot(
        scores=scores,
        heterogeneity_method_params=heterogeneity_method_params,
        scoring_function_name=scoring_function_name,
        plot_title=plot_title,
        heterogeneity_plot_save_path=save_file_path,
    )


def make_heterogeneity_plot(
    scores: dict[str, list[float]],
    heterogeneity_method_params: list[float],
    scoring_function_name: str,
    plot_title: str,
    heterogeneity_plot_save_path: str | Path,
    heterogeneity_method_params_names: dict[int, str] | None = None,
):
    """
    Make a heterogeneity plot.

    Parameters
    ----------
    scores : dict[str, list[float]]
        The scores.

    heterogeneity_method_params : list[float]
        The heterogeneity method parameters.

    scoring_function_name : str
        The scoring function name.

    plot_title : str
        The plot title.

    heterogeneity_plot_save_path : Union[str, Path]
        The path where to save the heterogeneity plot.

    heterogeneity_method_params_names : dict[int, str] or None
        The heterogeneity method parameters names.

    """
    sns.set_style("whitegrid")
    # Create a dataframe with all scores
    lines = []
    for method_test, scores_list in scores.items():
        for i, score in enumerate(scores_list):
            lines.append(
                {
                    "method_test_name": process_method_name(method_test),
                    # Here we invert the heterogeneity level
                    "heterogeneity level": heterogeneity_method_params_names[i]
                    if heterogeneity_method_params_names is not None
                    else 1.0 - heterogeneity_method_params[i],
                    "score": score,
                }
            )
    df = pd.DataFrame(lines)
    # Type the columns
    df = df.astype(
        {
            "method_test_name": "string",
            "heterogeneity level": "float",
            "score": "float",
        }
    )
    ax = sns.barplot(
        df,
        x="method_test_name",
        y="score",
        hue="heterogeneity level",
        palette="viridis",
    )
    ax.set_title(plot_title)
    ax.set_ylabel(f"{SCORING_FUNCTIONS_YLABELS[scoring_function_name]}")
    # Set limits for the y axis
    ax.set_ylim(0, 1)
    ax.set_xlabel("Method")
    _, xlabels = plt.xticks()
    xticks_loc = ax.get_xticks()
    ax.set_xticks(xticks_loc)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # create the save directory
    heterogeneity_plot_save_path = Path(heterogeneity_plot_save_path)
    heterogeneity_plot_save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(heterogeneity_plot_save_path, bbox_inches="tight", transparent=True)
    plt.close()


def get_heterogeneity_plot_name(
    methods_test: list[str],
    method_ref: str,
    scoring_function_name: str,
) -> str:
    """
    Get the heterogeneity plot name.

    Parameters
    ----------
    methods_test : list[str]
        The test methods.

    method_ref : str
        The reference method.

    scoring_function_name : str
        The scoring function name.

    Returns
    -------
    str
        The heterogeneity plot name.

    """
    methods_test_str = "-".join(methods_test)
    return (
        "heterogeneity_plot"
        f"_{scoring_function_name}_"
        f"test_{methods_test_str}_"
        f"ref_{method_ref}"
    )
