from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fedpydeseq2_datasets.constants import TCGADatasetNames
from fedpydeseq2_datasets.utils import get_experiment_id
from loguru import logger
from matplotlib.scale import SymmetricalLogScale

from paper_experiments.figures.generate_cross_tables_utils import (
    get_padj_lfc_from_method,
)
from paper_experiments.utils.constants import MetaAnalysisParameter


def build_lfc_or_padj_rel_error_violin_plot(
    methods: list[str],
    method_results_paths: dict[str, Path],
    save_path: str | Path,
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
    **pydeseq2_kwargs: Any,
):
    """
    Build violin plots of the relative error of the log-fold change and padj.

    Parameters
    ----------
    methods : list[str]
        The list of methods to include in the violin plot.
        Must contain pydeseq2.

    method_results_paths : dict[str, Path]
        The dictionary of method results paths, mapping
        the method to the path where the results are stored.

    save_path : Union[str, Path]
        The path where to save the violin plots.

    dataset_name : TCGADatasetNames
        The name of the dataset to use.

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

    **pydeseq2_kwargs : Any
        Additional keyword arguments to pass to the PyDESeq2 and FedPyDESeq2.
        For example the contrast parameter.

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

    refit_cooks = pydeseq2_kwargs.get("refit_cooks", True)

    logger.info(
        f"Building LFC and padj relative error violin plots for {dataset_name}."
    )

    df = pd.DataFrame()

    all_methods = []

    for method in methods:
        method_padj, method_lfc = get_padj_lfc_from_method(
            method,
            method_results_paths[method],
            experiment_id,
            refit_cooks=refit_cooks,
            reference_dds_ref_level=reference_dds_ref_level,
            meta_analysis_parameters=meta_analysis_parameters,
        )
        if isinstance(method_padj, dict):
            assert isinstance(method_lfc, dict)
            assert set(method_padj.keys()) == set(method_lfc.keys())
            for method_id in method_padj.keys():
                # Extract the meta-analysis submethod name in a more readable format
                submethod_name = method_id.split(", ")[1:]
                submethod_name = "_".join(
                    [param for param in submethod_name if param != "None"]
                )
                all_methods.append(submethod_name)

                # Add the lfc and padj to the dataframe
                df[f"{submethod_name}_lfc"] = method_lfc[method_id]
                df[f"{submethod_name}_padj"] = method_padj[method_id]

        else:
            all_methods.append(method)
            # Add the lfc and padj to the dataframe
            df[f"{method}_lfc"] = method_lfc
            df[f"{method}_padj"] = method_padj

    rel_err_df = pd.DataFrame()
    for method in all_methods:
        if method != "pydeseq2":
            tmp_df = pd.DataFrame(
                {
                    "lfc_rel_error": (df[method + "_lfc"] - df["pydeseq2_lfc"]).abs()
                    / df["pydeseq2_lfc"].abs(),
                    "padj_rel_error": (df[method + "_padj"] - df["pydeseq2_padj"]).abs()
                    / df["pydeseq2_padj"].abs(),
                    "method": method,
                }
            )
            rel_err_df = pd.concat([rel_err_df, tmp_df])

    lfc_save_file_path = save_path / experiment_id / "lfc_plot.pdf"
    padj_save_file_path = save_path / experiment_id / "padj_plot.pdf"

    # Make the plot
    make_lfc_or_padj_violin_plot(
        rel_err_df,
        lfc_save_file_path,
        padj_or_lfc="lfc",
    )

    make_lfc_or_padj_violin_plot(
        rel_err_df,
        padj_save_file_path,
        padj_or_lfc="padj",
        linthresh=1e-6,
    )


def make_lfc_or_padj_violin_plot(
    df: pd.DataFrame,
    save_file_path: str | Path,
    linthresh: float = 1e-8,
    padj_or_lfc: str = "lfc",
):
    """
    Make a violin plot of the relative error of the log-fold change or padj.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the relative error of the log-fold change or padj.

    save_file_path : Union[str, Path]
        The path where to save the violin plot.

    linthresh : float
        The linthresh parameter of the symlog scale. Values in [-linthresh, linthresh]
        are linear, values outside are logarithmic.

    padj_or_lfc : str
        Whether to plot the padj or the log-fold change.
    """
    plt.clf()

    fig, ax = plt.subplots(figsize=(8, 8))

    assert padj_or_lfc in ["lfc", "padj"]

    symlogscale = SymmetricalLogScale(ax, linthresh=linthresh)

    df[
        f"symlog_{padj_or_lfc}_rel_error"
    ] = symlogscale.get_transform().transform_non_affine(df[f"{padj_or_lfc}_rel_error"])

    df.reset_index(drop=True, inplace=True)

    # Filter out infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    sns.violinplot(
        data=df[~df[f"{padj_or_lfc}_rel_error"].isna()],
        y=f"symlog_{padj_or_lfc}_rel_error",
        x="method",
        zorder=0,
        saturation=0.5,
        palette="bright",
        ax=ax,
    )

    # Set y ticks, knowing that the symlog scale is used
    ymax = df[f"{padj_or_lfc}_rel_error"].max()

    tick_range = np.concatenate(
        [
            [0],
            np.logspace(
                np.floor(np.log10(linthresh)),
                np.ceil(np.log10(ymax)),
                num=int(np.ceil(np.log10(ymax)))
                - int(np.floor(np.log10(linthresh)))
                + 1,
                endpoint=True,
            ),
        ]
    )
    ax.set_yticks(symlogscale.get_transform().transform_non_affine(tick_range))
    ax.set_yticklabels([f"{tick:.0e}" if tick != 0 else "0" for tick in tick_range])

    (
        ax.set_ylim(
            0,
            min(
                symlogscale.get_transform()
                .transform_non_affine(np.array([ymax]))
                .item(),
                symlogscale.get_transform()
                .transform_non_affine(np.array([1e6]))
                .item(),
            ),
        ),
    )

    for axis in ["bottom", "left"]:
        ax.spines[axis].set_linewidth(2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(width=2)

    plt.xticks(size=12, weight="bold")
    plt.yticks(size=12, weight="bold")

    # Turn off x label
    plt.xlabel("")

    # Set x tick labels to vertical
    plt.xticks(rotation=90)
    if padj_or_lfc == "padj":
        plt.ylabel("FDR relative error", size=15)
    else:
        plt.ylabel("$log_{2}$ fold change relative error", size=15)

    plt.tight_layout()
    save_file_path = Path(save_file_path)
    save_file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file_path, transparent=True)
    plt.close()
