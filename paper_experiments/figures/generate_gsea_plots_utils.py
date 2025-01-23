from pathlib import Path
from typing import Any
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fedpydeseq2_datasets.constants import TCGADatasetNames
from fedpydeseq2_datasets.utils import get_experiment_id
from fedpydeseq2_datasets.utils import get_ground_truth_dds_name
from fedpydeseq2_datasets.utils import get_valid_centers_from_subfolders_file
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize

from paper_experiments.run_dge_gsea_methods.meta_analysis_tcga_pipe import (
    get_meta_analysis_id,
)
from paper_experiments.utils.constants import DGE_MODES
from paper_experiments.utils.constants import MetaAnalysisParameter


def build_pathways_dfs(
    method_gsea_dicts: dict[str, dict[TCGADatasetNames, pd.DataFrame]],
    top_n: int = 10,
):
    """Build summary dataframes with the top n pathways of each method.

    The input  are dictionaries, where keys are method names and values
    are a dictionary where keys are dataset names and values are pd.DataFrames
    containing the GSEA results for that dataset.

    The datasets must be the same for all methods.

    This function first identifies the top 10 pathways for each dataset for each method,
    and aggregates them as a global set of pathways.

    This function then creates two dictionaries, one containing
    the NES and one containing
    the p-values for each pathway and each dataset, for each method.

    Parameters
    ----------
    method_gsea_dicts : dict[str, dict[TCGADatasetNames, pd.DataFrame]]
        A dictionary where each key is a method name, and each value is
        a dictionary where the keys are dataset names and the values are
        pd.DataFrames containing the GSEA results for that dataset.
        The datasets must be the same for all methods.

    top_n : int
        Number of top pathways to consider, for each dataset and method.
        Default is 10.

    Returns
    -------
    method_nes_dict : dict[str, pd.DataFrame]
        A dictionary where each key is a method name, and each value is a
        pd.DataFrame containing the NES indexed by pathways and with columns
        corresponding to the datasets.

    method_pvalues_dict : dict[str, pd.DataFrame]
        A dictionary where each key is a method name, and each value is a
        pd.DataFrame containing the p-values indexed by pathways and with columns
        corresponding to the datasets.
    """
    # Start by checking that the datasets match
    method_0_gsea_dict = method_gsea_dicts[list(method_gsea_dicts.keys())[0]]
    datasets = list(method_0_gsea_dict.keys())
    for method_gsea_dict in method_gsea_dicts.values():
        assert set(method_gsea_dict.keys()) == set(datasets)

    # Remove the 'REACTOME_' prefix from pathway names
    for dataset in datasets:
        for method_gsea_dict in method_gsea_dicts.values():
            method_gsea_dict[dataset].index = pd.Index(
                [
                    pathway[pathway.startswith("REACTOME_") and len("REACTOME_") :]
                    for pathway in method_gsea_dict[dataset].index
                ]
            )

    # Get the top n pathways for each dataset and method
    all_pathways: set[str] = set()
    for method_gsea_dict in method_gsea_dicts.values():
        for method_dataset_df in method_gsea_dict.values():
            all_pathways = all_pathways.union(set(method_dataset_df[:top_n].index))

    method_nes_dict = {
        method_name: pd.DataFrame(
            index=list(all_pathways), columns=datasets, dtype=float
        )
        for method_name in method_gsea_dicts.keys()
    }
    method_pvalues_dict = {
        method_name: pd.DataFrame(
            index=list(all_pathways), columns=datasets, dtype=float
        )
        for method_name in method_gsea_dicts.keys()
    }

    for method_name, method_gsea_dict in method_gsea_dicts.items():
        for dataset, method_dataset_df in method_gsea_dict.items():
            method_nes_dict[method_name][dataset] = method_dataset_df["NES"]
            method_pvalues_dict[method_name][dataset] = method_dataset_df["padj"]

    for method_name, method_nes_df in method_nes_dict.items():
        method_nes_df.sort_index(inplace=True, key=lambda x: x.str[1:].astype(int))
        method_pvalues_dict[method_name].sort_index(
            inplace=True, key=lambda x: x.str[1:].astype(int)
        )

    return (
        method_nes_dict,
        method_pvalues_dict,
    )


def make_single_plot(
    df: pd.DataFrame,
    ax: plt.Axes,
    ax_title: str,
    cbar_ax: plt.Axes | None,
    cbar_label: str,
    cmap: str | LinearSegmentedColormap,
    custom_cmap: bool = False,
    logscale: bool = False,
    **custom_cmap_kwargs: Any,
):
    """Make a single heatmap plot.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to plot.

    ax : plt.Axes
        Axis to plot on.

    ax_title : str
        Title of the axis.

    cbar_ax : plt.Axes or None
        Axis for the colorbar.

    cbar_label  : str
        Label of the colorbar.

    cmap : str
        Colormap to use.

    custom_cmap : bool
        If True, use a custom colormap.

    logscale : bool
        If True, use a log scale.

    **custom_cmap_kwargs : Any
        Additional arguments for the custom colormap.
    """
    if logscale:
        new_df = df.fillna(1.0)
    else:
        new_df = df.fillna(0.0)

    if custom_cmap:
        assert isinstance(cmap, str)
        cmap = create_cmap_from_diverging_cmap(
            cmap_name=cmap, logspace=logscale, **custom_cmap_kwargs
        )

    res = sns.heatmap(
        new_df,
        cmap=cmap,
        linewidth=0.5,
        linecolor="grey",
        cbar_ax=cbar_ax,
        cbar_kws={"label": cbar_label},
        center=0,
        ax=ax,
        norm=(
            LogNorm(
                vmin=custom_cmap_kwargs.get("min_value", None),
                vmax=custom_cmap_kwargs.get("max_value", None),
            )
            if logscale
            else Normalize(
                vmin=custom_cmap_kwargs.get("min_value", None),
                vmax=custom_cmap_kwargs.get("max_value", None),
            )
        ),
    )

    ax.tick_params(axis="x", labelrotation=90)
    ax.set_title(ax_title, fontsize=16, y=1.01)

    # Drawing the frame
    for _, spine in res.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("grey")


def make_gsea_plot(
    method_nes_dict: dict[str, pd.DataFrame],
    method_pvalues_dict: dict[str, pd.DataFrame],
    save_path: str | Path,
    reference_method_name: str | None = None,
    with_diff: bool = False,
    padj_threshold: float | None = 0.05,
    with_pvalues: bool = False,
    padj_threshold_pvalues: float = 0.05,
    padj_max_pvalues: float = 0.1,
    padj_min_pvalues: float = 1e-3,
):
    """Make a heatmap of the GSEA results for method 1 and method 2.

    Parameters
    ----------
    method_nes_dict : dict[str, pd.DataFrame]
        A dictionary where each key is a method name, and each value is a
        pd.DataFrame containing the NES indexed by pathways and with columns
        corresponding to the datasets.

    method_pvalues_dict : dict[str, pd.DataFrame]
        A dictionary where each key is a method name, and each value is a
        pd.DataFrame containing the p-values indexed by pathways and with columns
        corresponding to the datasets.

    save_path : Union[str, Path]
        Path where to save the figure.

    reference_method_name : str or None
        Must be specified if with_diff is True. The name of the method to use
        as reference.

    with_diff : bool
        If True, plot the difference between the NES of all methods and the reference
        method.

    padj_threshold : float or None
        If not None, The NES values with corresponding p-values above this threshold
        are set to NaN.

    with_pvalues : bool
        If True, plot the p-values.

    padj_threshold_pvalues : float
        The p-values threshold for the p-values heatmap.

    padj_max_pvalues : float
        The maximum p-value to consider for the p-values heatmap.

    padj_min_pvalues : float
        The minimum p-value to consider for the p-values heatmap.
    """
    n_methods = len(method_nes_dict)
    assert set(method_nes_dict.keys()) == set(method_pvalues_dict.keys())

    # Get the datasets
    datasets = list(method_nes_dict[list(method_nes_dict.keys())[0]].columns)
    # Check that all methods have the same datasets and reorder
    for method_name, method_nes_df in method_nes_dict.items():
        assert set(method_nes_df.columns) == set(datasets)
        method_nes_dict[method_name] = method_nes_df[datasets]
    for method_name, method_pvalues_df in method_pvalues_dict.items():
        assert set(method_pvalues_df.columns) == set(datasets)
        method_pvalues_dict[method_name] = method_pvalues_df[datasets]

    # Check that if with_diff is True, reference_method_name is not None
    if with_diff:
        assert reference_method_name is not None
        assert reference_method_name in method_nes_dict.keys()

    n_plots = n_methods
    if with_pvalues:
        n_plots += n_methods

    if with_diff:
        n_plots += n_methods - 1

    n_cbars = 2 if with_pvalues else 1

    plt.clf()
    fig, axes = plt.subplots(
        1, n_plots + n_cbars, figsize=((n_plots + n_cbars) * 3, 15), sharey=True
    )

    if with_pvalues:
        # Get left, bottom, width, height of second to last ax
        left, _, width, _ = axes[-2].get_position().bounds
        axes[-2].axis("off")
        cbar_ax = fig.add_axes((left, 0.3, width / 3, 0.4))

        # Same for pvalues
        left, _, width, _ = axes[-1].get_position().bounds
        axes[-1].axis("off")
        cbar_ax_pvalues = fig.add_axes((left, 0.3, width / 3, 0.4))
    else:
        left, _, width, _ = axes[-1].get_position().bounds
        axes[-1].axis("off")
        cbar_ax = fig.add_axes((left, 0.3, width / 3, 0.4))
        cbar_ax_pvalues = None

    if padj_threshold is not None:
        # in this case, use the pvalue threshold and set to NaN
        for method_name, method_nes in method_nes_dict.items():
            method_pvalues = method_pvalues_dict[method_name]
            for dataset in datasets:
                method_nes.loc[method_pvalues[dataset] > padj_threshold, dataset] = None

    if reference_method_name is not None:
        method_list = [reference_method_name]
        method_list.extend(
            [
                method_name
                for method_name in method_nes_dict.keys()
                if method_name != reference_method_name
            ]
        )

    else:
        method_list = list(method_nes_dict.keys())

    name_map = {
        "FedPyDESeq2 \n simulated": "FedPyDESeq2",
        "FedPyDESeq2 \n remote": "FedPyDESeq2",
        "PyDESeq2 \n pooled": "PyDESeq2",
    }

    ax_number = 0
    for method_name in method_list:
        method_nes = method_nes_dict[method_name]
        method_pvalues = method_pvalues_dict[method_name]
        make_single_plot(
            method_nes,
            axes[ax_number],
            name_map[method_name] if method_name in name_map else method_name,
            cbar_ax,
            "Normalized enrichment score (NES)",
            "seismic",
            logscale=False,
        )
        ax_number += 1
        if with_pvalues:
            make_single_plot(
                method_pvalues,
                axes[ax_number],
                name_map[method_name] if method_name in name_map else method_name,
                cbar_ax_pvalues,
                "p-value",
                cmap="RdYlGn",
                logscale=True,
                custom_cmap=True,
                min_value=padj_min_pvalues,
                max_value=padj_max_pvalues,
                white_value=padj_threshold_pvalues,
                reverse=True,
            )
            ax_number += 1
        if with_diff:
            assert reference_method_name is not None
            if method_name is not None and method_name == reference_method_name:
                continue
            diff_df = method_nes.subtract(
                method_nes_dict[reference_method_name], fill_value=0
            )
            make_single_plot(
                diff_df,
                axes[ax_number],
                (
                    f"{name_map[method_name]} \n - \n {name_map[reference_method_name]}"
                    if (method_name in name_map) and (reference_method_name in name_map)
                    else f"{method_name} - {reference_method_name}"
                ),
                cbar_ax,
                "Normalized enrichment score (NES)",
                "seismic",
                logscale=False,
            )
            ax_number += 1

    # Increase colorbar font size
    cbar_ax.yaxis.label.set_fontsize(18)
    cbar_ax.tick_params(labelsize=14)
    if with_pvalues:
        assert cbar_ax_pvalues is not None
        cbar_ax_pvalues.yaxis.label.set_fontsize(18)
        cbar_ax_pvalues.tick_params(labelsize=14)

    # Save figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close()


def get_fedpydeseq2_gsea_dict(
    fedpydeseq2_gsea_path: str | Path,
    experiment_id: str,
    method_type: str,
) -> dict[str, pd.DataFrame]:
    """Get the GSEA results for fedpydeseq2.

    Parameters
    ----------
    fedpydeseq2_gsea_path : Union[str, Path]
        Path to the GSEA results.

    experiment_id : str
        The experiment id.

    method_type : str
        The method type.
        Can be fedpydeseq2_simulated or fedpydeseq2_remote.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary containing the GSEA results for fedpydeseq2, where the key is
        "FedPyDESeq2" and the value is a pd.DataFrame containing the GSEA results for
        the specified experiment id (dataset)

    Raises
    ------
    ValueError
        If method_type is not recognized.
    """
    fedpydeseq2_gsea_path = Path(fedpydeseq2_gsea_path)
    gsea_result = pd.read_csv(
        fedpydeseq2_gsea_path / experiment_id / "gsea_results.csv",
        sep="\t",
        index_col=0,
    )
    if method_type == "fedpydeseq2_simulated":
        # TODO change key (at least plot for mapping)
        return {"FedPyDESeq2 \n simulated": gsea_result}
    elif method_type == "fedpydeseq2_remote":
        return {"FedPyDESeq2 \n remote": gsea_result}
    else:
        raise ValueError(f"Method {method_type} not recognized.")


def get_pooled_pydeseq2_gsea_dict(
    pooled_pydeseq2_gsea_path: str | Path,
    experiment_id: str,
    refit_cooks: bool = True,
    reference_dds_ref_level: tuple[str, str] = ("stage", "Advanced"),
) -> dict[str, pd.DataFrame]:
    """Get the GSEA results for pydeseq2 pooled.

    Parameters
    ----------
    pooled_pydeseq2_gsea_path : Union[str, Path]
        Path to the GSEA results.

    experiment_id : str
        The experiment id.

    refit_cooks : bool
        If True, refit the model after removing the Cook's distance outliers.

    reference_dds_ref_level : tuple[str, str]
        The reference level for the design factor.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary containing the GSEA results for pydeseq2 pooled, where the key is
        "PyDESeq2 pooled" and the value is a pd.DataFrame containing the GSEA
        results for the specified experiment id (dataset)
    """
    pooled_gsea_path = Path(pooled_pydeseq2_gsea_path)
    ground_truth_dds_name = get_ground_truth_dds_name(
        reference_dds_ref_level=reference_dds_ref_level,
        refit_cooks=refit_cooks,
        pooled=True,
    )
    return {
        # TODO change key
        "PyDESeq2 \n pooled": pd.read_csv(
            pooled_gsea_path
            / experiment_id
            / f"{ground_truth_dds_name}_gsea_results.csv",
            sep="\t",
            index_col=0,
        )
    }


def get_pydeseq2_largest_gsea_dict(
    pydeseq2_largest_gsea_path: str | Path,
    experiment_id: str,
    refit_cooks: bool = True,
    reference_dds_ref_level: tuple[str, str] = ("stage", "Advanced"),
) -> dict[str, pd.DataFrame]:
    """Get the GSEA results for pydeseq2 pooled.

    Parameters
    ----------
    pydeseq2_largest_gsea_path : str | Path
        Path to the GSEA results.

    experiment_id : str
        The experiment id.

    refit_cooks : bool
        If True, refit the model after removing the Cook's distance outliers.

    reference_dds_ref_level : tuple[str, str]
        The reference level for the design factor.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary containing the GSEA results for pydeseq2 pooled, where the key is
        "PyDESeq2 pooled" and the value is a pd.DataFrame containing the GSEA
        results for the specified experiment id (dataset)
    """
    pydeseq2_largest_gsea_path = Path(pydeseq2_largest_gsea_path)
    ground_truth_dds_name = get_ground_truth_dds_name(
        reference_dds_ref_level=reference_dds_ref_level,
        refit_cooks=refit_cooks,
        pooled=False,
    )
    return {
        "PyDESeq2 \n largest": pd.read_csv(
            pydeseq2_largest_gsea_path
            / experiment_id
            / f"{ground_truth_dds_name}_gsea_results.csv",
            sep="\t",
            index_col=0,
        )
    }


def get_per_center_pydeseq2_gsea_dict(
    per_center_pydeseq2_gsea_path: str | Path,
    experiment_id: str,
    refit_cooks: bool = True,
    reference_dds_ref_level: tuple[str, str] = ("stage", "Advanced"),
) -> dict[str, pd.DataFrame]:
    """Get the GSEA results for pydeseq2 per center.

    Parameters
    ----------
    per_center_pydeseq2_gsea_path : Union[str, Path]
        Path to the GSEA results.

    experiment_id : str
        The experiment id.

    refit_cooks : bool
        If True, refit the model after removing the Cook's distance outliers.

    reference_dds_ref_level : tuple[str, str]
        The reference level for the design factor.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary containing the GSEA results for pydeseq2 per center,
        where the key is
        "PyDESeq2 per center" and the value is a pd.DataFrame containing the GSEA
        results for the specified experiment id (dataset)
    """
    centers_gsea_path = Path(per_center_pydeseq2_gsea_path)
    ground_truth_dds_name = get_ground_truth_dds_name(
        reference_dds_ref_level=reference_dds_ref_level,
        refit_cooks=refit_cooks,
        pooled=False,
    )
    experiment_gsea_path = centers_gsea_path / experiment_id
    _, existing_centers = get_valid_centers_from_subfolders_file(
        experiment_gsea_path,
        f"{ground_truth_dds_name}_gsea_results.csv",
        pkl=False,
    )
    return {
        f"PyDESeq2 \n center {center_id}": pd.read_csv(
            experiment_gsea_path
            / f"center_{center_id}"
            / f"{ground_truth_dds_name}_gsea_results.csv",
            sep="\t",
            index_col=0,
        )
        for center_id in existing_centers
    }


def get_meta_analysis_gsea_dict(
    meta_analysis_gsea_path: str | Path,
    experiment_id: str,
    meta_analysis_parameters: list[MetaAnalysisParameter],
) -> dict[str, pd.DataFrame]:
    """Get the GSEA results for the meta analysis.

    Parameters
    ----------
    meta_analysis_gsea_path : Union[str, Path]
        Path to the GSEA results.

    experiment_id : str
        The experiment id.

    meta_analysis_parameters : list[MetaAnalysisParameter]
        The meta analysis parameters.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary containing the GSEA results for the meta analysis, where the key is
        the meta analysis parameters and the value is a pd.DataFrame containing the GSEA
        results for the specified experiment id (dataset)
    """
    meta_analysis_gsea_path = Path(meta_analysis_gsea_path)
    meta_analysis_ids = [
        get_meta_analysis_id(*meta_analysis_parameter)
        for meta_analysis_parameter in meta_analysis_parameters
    ]

    unordered_result = {
        " \n ".join(
            [
                "Meta analysis",
                *[str(subparameter) for subparameter in meta_analysis_parameter],
            ]
        ): pd.read_csv(
            meta_analysis_gsea_path
            / experiment_id
            / f"{meta_analysis_id}_gsea_results.csv",
            sep="\t",
            index_col=0,
        )
        for meta_analysis_id, meta_analysis_parameter in zip(
            meta_analysis_ids, meta_analysis_parameters, strict=False
        )
    }

    # Reorder the keys
    sorted_keys = sorted(unordered_result.keys())
    return {key: unordered_result[key] for key in sorted_keys}


def get_method_experiment_id_gsea_dict(
    method_type: str,
    method_gsea_path: str | Path,
    experiment_id: str,
    refit_cooks: bool = True,
    reference_dds_ref_level: tuple[str, str] | None = ("stage", "Advanced"),
    meta_analysis_parameters: list[MetaAnalysisParameter] | None = None,
) -> dict[str, pd.DataFrame]:
    """Get the GSEA results for a method.

    Parameters
    ----------
    method_type : str
        The method type.

    method_gsea_path : Union[str, Path]
        Path to the GSEA results.

    experiment_id : str
        The experiment id.

    refit_cooks : bool
        If True, refit the model after removing the Cook's distance outliers.

    reference_dds_ref_level : tuple[str, str] or None
        The reference level for the design factor.
        Must be specified if method_type is pydeseq2 or pydeseq2_per_center.

    meta_analysis_parameters : list[MetaAnalysisParameter] or None
        The meta analysis parameters.
        Must be specified if method_type is meta_analysis

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary containing the GSEA results for the method, where the key is
        the method name and the value is a pd.DataFrame containing the GSEA
        results for the specified experiment id (dataset)
    """
    assert method_type in DGE_MODES, f"Method {method_type} not recognized."
    if method_type.startswith("fedpydeseq2"):
        return get_fedpydeseq2_gsea_dict(method_gsea_path, experiment_id, method_type)
    elif method_type == "pydeseq2":
        assert reference_dds_ref_level is not None
        return get_pooled_pydeseq2_gsea_dict(
            method_gsea_path,
            experiment_id,
            refit_cooks=refit_cooks,
            reference_dds_ref_level=reference_dds_ref_level,
        )
    elif method_type == "pydeseq2_per_center":
        assert reference_dds_ref_level is not None
        return get_per_center_pydeseq2_gsea_dict(
            method_gsea_path,
            experiment_id,
            refit_cooks=refit_cooks,
            reference_dds_ref_level=reference_dds_ref_level,
        )
    elif method_type == "pydeseq2_largest":
        assert reference_dds_ref_level is not None
        return get_pydeseq2_largest_gsea_dict(
            method_gsea_path,
            experiment_id,
            refit_cooks=refit_cooks,
            reference_dds_ref_level=reference_dds_ref_level,
        )

    elif method_type == "meta_analysis":
        assert meta_analysis_parameters is not None
        return get_meta_analysis_gsea_dict(
            method_gsea_path, experiment_id, meta_analysis_parameters
        )
    else:
        raise ValueError(f"Method {method_type} not recognized.")


def make_gsea_plot_dge_pair(
    method_test_type: str,
    method_ref_type: str,
    method_test_gsea_path: str | Path,
    method_ref_gsea_path: str | Path,
    dataset_names: list[TCGADatasetNames],
    save_path: str | Path,
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    reference_dds_ref_level: tuple[str, str] | None = ("stage", "Advanced"),
    meta_analysis_parameters: list[MetaAnalysisParameter] | None = None,
    with_diff: bool = False,
    padj_threshold: float | None = 0.05,
    with_pvalues: bool = False,
    top_n: int = 10,
    separate_plots: bool = False,
    **pydeseq2_kwargs: Any,
):
    """Make a GSEA plot comparing the results of two DGE methods.

    Parameters
    ----------
    method_test_type : str
        The name of the test method.

    method_ref_type : str
        The name of the reference method.

    method_test_gsea_path : str or Path
        Path to the GSEA results for the test method.

    method_ref_gsea_path : str or Path
        Path to the GSEA results for the reference method.

    dataset_names : list[str]
        List of dataset names.

    save_path : str or Path
        Path where to save the figure.

    small_samples : bool
        If True, use the small samples dataset.

    small_genes : bool
        If True, use the small genes dataset.

    only_two_centers : bool
        If True, use only two centers.

    design_factors : str or list[str]
        The design factors to use.

    continuous_factors : list[str] or None
        The continuous factors to use.

    reference_dds_ref_level : tuple[str, str]
        The reference level for the design factor.

    meta_analysis_parameters : list[MetaAnalysisParameter] or None
        The meta analysis parameters.

    with_diff : bool
        If True, plot the difference between the NES of all methods and the reference

    padj_threshold : float or None
        If not None, The NES values with corresponding p-values above this threshold

    with_pvalues : bool
        If True, plot the p-values.

    top_n : int
        Number of top pathways to consider, for each dataset and method.

    separate_plots : bool
        If True, plot the GSEA results for each method separately.

    **pydeseq2_kwargs : Any
        Additional arguments for the pydeseq2 methods.
    """
    refit_cooks = pydeseq2_kwargs.get("refit_cooks", True)
    experiment_ids = [
        get_experiment_id(
            dataset_name=cast(TCGADatasetNames, dataset_name),
            small_samples=small_samples,
            small_genes=small_genes,
            only_two_centers=only_two_centers,
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            **pydeseq2_kwargs,
        )
        for dataset_name in dataset_names
    ]

    # Create the save folder
    full_experiment_id = get_experiment_id(
        dataset_name=dataset_names,
        small_samples=small_samples,
        small_genes=small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        **pydeseq2_kwargs,
    )

    save_folder = Path(save_path) / full_experiment_id
    save_folder.mkdir(parents=True, exist_ok=True)

    method_test_gsea_dicts = {
        dataset: get_method_experiment_id_gsea_dict(
            method_test_type,
            method_test_gsea_path,
            experiment_id,
            refit_cooks=refit_cooks,
            reference_dds_ref_level=reference_dds_ref_level,
            meta_analysis_parameters=meta_analysis_parameters,
        )
        for dataset, experiment_id in zip(dataset_names, experiment_ids, strict=False)
    }
    method_ref_gsea_dicts = {
        dataset: get_method_experiment_id_gsea_dict(
            method_ref_type,
            method_ref_gsea_path,
            experiment_id,
            refit_cooks=refit_cooks,
            reference_dds_ref_level=reference_dds_ref_level,
            meta_analysis_parameters=meta_analysis_parameters,
        )
        for dataset, experiment_id in zip(dataset_names, experiment_ids, strict=False)
    }
    # Invert both dictionaries from dataset -> method_name -> df to
    # method_name -> dataset -> df
    test_method_names = list(method_test_gsea_dicts[dataset_names[0]].keys())
    ref_method_names = list(method_ref_gsea_dicts[dataset_names[0]].keys())
    method_test_dicts = {
        method_name: {
            dataset: method_test_gsea_dicts[dataset][method_name]
            for dataset in dataset_names
        }
        for method_name in test_method_names
    }
    method_ref_dicts = {
        method_name: {
            dataset: method_ref_gsea_dicts[dataset][method_name]
            for dataset in dataset_names
        }
        for method_name in ref_method_names
    }

    if separate_plots and len(method_test_dicts) > 1:
        for method_test_name, method_test_dict in method_test_dicts.items():
            new_method_test_dicts = {method_test_name: method_test_dict}
            all_methods_dict = {**new_method_test_dicts, **method_ref_dicts}
            method_nes_dict, method_pvalues_dict = build_pathways_dfs(
                all_methods_dict, top_n=top_n
            )
            if with_diff:
                assert len(method_ref_dicts) == 1
                reference_method_name = list(method_ref_dicts.keys())[0]
            else:
                reference_method_name = None
            save_file_path = save_folder / get_gsea_plot_filename(
                method_test_type,
                method_ref_type,
                with_diff,
                padj_threshold,
                with_pvalues,
                method_name=method_test_name,
            )
            make_gsea_plot(
                method_nes_dict,
                method_pvalues_dict,
                save_file_path,
                reference_method_name=reference_method_name,
                with_diff=with_diff,
                padj_threshold=padj_threshold,
                with_pvalues=with_pvalues,
            )
    else:
        all_methods_dict = {**method_test_dicts, **method_ref_dicts}

        method_nes_dict, method_pvalues_dict = build_pathways_dfs(
            all_methods_dict, top_n=top_n
        )
        if with_diff:
            assert len(method_ref_dicts) == 1
            reference_method_name = list(method_ref_dicts.keys())[0]

        else:
            reference_method_name = None

        save_file_path = save_folder / get_gsea_plot_filename(
            method_test_type, method_ref_type, with_diff, padj_threshold, with_pvalues
        )

        make_gsea_plot(
            method_nes_dict,
            method_pvalues_dict,
            save_file_path,
            reference_method_name=reference_method_name,
            with_diff=with_diff,
            padj_threshold=padj_threshold,
            with_pvalues=with_pvalues,
        )


def make_gsea_plot_method_pairs(
    method_pairs: list[tuple[str, str]],
    method_gsea_paths: dict[str, Path],
    dataset_names: list[TCGADatasetNames],
    save_path: str | Path,
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    reference_dds_ref_level: tuple[str, str] | None = ("stage", "Advanced"),
    meta_analysis_parameters: list[MetaAnalysisParameter] | None = None,
    with_diff: bool = False,
    padj_threshold: float | None = 0.05,
    with_pvalues: bool = False,
    top_n: int = 10,
    separate_plots: list[bool] | None = None,
    **pydeseq2_kwargs: Any,
):
    """Make GSEA plots for pairs of methods.

    Parameters
    ----------
    method_pairs : list[tuple[str, str]]
        List of pairs of methods to compare.

    method_gsea_paths : dict[str, Path]
        Dictionary where keys are method types and values are paths to the GSEA results.

    dataset_names : list[TCGADatasetNames]
        List of dataset names.

    save_path : str or Path
        Path where to save the figure.

    small_samples : bool
        If True, use the small samples dataset.

    small_genes : bool
        If True, use the small genes dataset.

    only_two_centers : bool
        If True, use only two centers.

    design_factors : str or list[str]
        The design factors to use.

    continuous_factors : list[str] or None
        The continuous factors to use.

    reference_dds_ref_level : tuple[str, str]
        The reference level for the design factor.

    meta_analysis_parameters : list[MetaAnalysisParameter] or None
        The meta analysis parameters.

    with_diff : bool
        If True, plot the difference between the NES of all methods and the reference

    padj_threshold : float or None
        If not None, The NES values with corresponding p-values above this threshold

    with_pvalues : bool
        If True, plot the p-values.

    top_n : int
        Number of top pathways to consider, for each dataset and method.

    separate_plots : list[bool] or None
        If True, plot the GSEA results for each method separately.

    pydeseq2_kwargs : Any
        Additional arguments for the pydeseq2 methods.
    """
    if separate_plots is None:
        separate_plots = [False] * len(method_pairs)
    else:
        assert len(separate_plots) == len(method_pairs)

    for separate_plot_pair, (method_test_type, method_ref_type) in zip(
        separate_plots, method_pairs, strict=False
    ):
        method_test_gsea_path = method_gsea_paths[method_test_type]
        method_ref_gsea_path = method_gsea_paths[method_ref_type]
        make_gsea_plot_dge_pair(
            method_test_type,
            method_ref_type,
            method_test_gsea_path,
            method_ref_gsea_path,
            dataset_names,
            save_path,
            small_samples=small_samples,
            small_genes=small_genes,
            only_two_centers=only_two_centers,
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            reference_dds_ref_level=reference_dds_ref_level,
            meta_analysis_parameters=meta_analysis_parameters,
            with_diff=with_diff,
            padj_threshold=padj_threshold,
            with_pvalues=with_pvalues,
            top_n=top_n,
            separate_plots=separate_plot_pair,
            **pydeseq2_kwargs,
        )


def get_gsea_plot_filename(
    method_test_type: str,
    method_ref_type: str,
    with_diff: bool = False,
    padj_threshold: float | None = 0.05,
    with_pvalues: bool = False,
    method_name: str | None = None,
):
    """Get the filename for the GSEA plot.

    Parameters
    ----------
    method_test_type : str
        The test method type.

    method_ref_type : str
        The reference method type.

    with_diff : bool
        If True, plot the difference between the NES of all methods and the reference
        method.

    padj_threshold : float or None
        If not None, The NES values with corresponding p-values above this threshold
        are set to NaN.

    with_pvalues : bool
        If True, plot the p-values.

    method_name : str or None
        The name of the method. If None,
        the method name is the same as the method test type.

    Returns
    -------
    str
        The filename for the GSEA plot.
    """
    if method_name is not None:
        method_test_name = method_name
    else:
        method_test_name = method_test_type
    if with_diff:
        filename = f"{method_test_name}_vs_{method_ref_type}"
    else:
        filename = f"{method_test_name}_and_{method_ref_type}"

    if padj_threshold is not None:
        filename += f"_padj_{padj_threshold}"
    if with_pvalues:
        filename += "_with_pvalues"
    filename += "_gsea_plot.pdf"
    return filename


def create_cmap_from_diverging_cmap(
    white_value: float = 5e-2,
    min_value: float = 1e-3,
    max_value: float = 1e-1,
    cmap_name: str = "RdYlGn",
    reverse: bool = True,
    logspace: bool = True,
    n_points=1000,
) -> LinearSegmentedColormap:
    """Create a new colormap by interpolating the colors.

    The goal of this function is to create a colourmap relevant to visualize
    pvalues.

    Parameters
    ----------
    white_value : float
        The value that will be white in the colormap. In practice, this will be
        the pvalue threshold.

    min_value : float
        The minimum value in the colormap.
        In practice, this will be the pvalue under which we do not care to discriminate

    max_value : float
        The maximum value in the colormap.
        In practice, this will be the maximum pvalue we want to discriminate.

    cmap_name : str
        The name of the diverging colormap to use.

    reverse : bool
        If True, reverse the colormap.

    logspace : bool
        If True, use logspace interpolation

    n_points : int
        Number of points to use for the interpolation
        Must be particularly high for logspace interpolation

    Returns
    -------
    LinearSegmentedColormap
        The new colormap.
    """
    # Get the original colormap
    cmap = plt.get_cmap(cmap_name)
    if reverse:
        cmap = cmap.reversed()

    if logspace:
        mid_value = np.log10(white_value / min_value) / np.log10(max_value / min_value)
    else:
        mid_value = (white_value - min_value) / (max_value - min_value)

    original_interpolation = np.linspace(0.0, 1.0, 2 * n_points + 1, endpoint=True)

    before_white_new = np.linspace(0.0, mid_value, n_points, endpoint=False)
    after_white_new = np.linspace(mid_value, 1.0, n_points + 1, endpoint=True)
    # concatenate
    new_interpolation = np.concatenate((before_white_new, after_white_new))

    # Get cmap R values for original interpolation
    cmap_r = [cmap(i)[0] for i in original_interpolation]
    cmap_g = [cmap(i)[1] for i in original_interpolation]
    cmap_b = [cmap(i)[2] for i in original_interpolation]

    # Interpolate the RGB values between the min_value, white_value,
    # and max_value points
    r = np.interp(original_interpolation, new_interpolation, cmap_r)
    g = np.interp(original_interpolation, new_interpolation, cmap_g)
    b = np.interp(original_interpolation, new_interpolation, cmap_b)
    # Create a dictionary that defines the new colormap
    cdict = {
        "red": [
            (original_interpolation_value, r_value, r_value)
            for original_interpolation_value, r_value in zip(
                original_interpolation, r, strict=False
            )
        ],
        "green": [
            (original_interpolation_value, g_value, g_value)
            for original_interpolation_value, g_value in zip(
                original_interpolation, g, strict=False
            )
        ],
        "blue": [
            (original_interpolation_value, b_value, b_value)
            for original_interpolation_value, b_value in zip(
                original_interpolation, b, strict=False
            )
        ],
    }

    # Create the new colormap
    new_cmap = LinearSegmentedColormap(cmap_name, cdict)  # type: ignore

    return new_cmap
