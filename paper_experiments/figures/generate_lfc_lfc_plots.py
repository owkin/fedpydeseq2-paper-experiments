from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from fedpydeseq2_datasets.constants import TCGADatasetNames
from fedpydeseq2_datasets.utils import get_experiment_id
from loguru import logger
from matplotlib.lines import Line2D

from paper_experiments.figures.generate_cross_tables_utils import NAME_MAPPING
from paper_experiments.figures.generate_cross_tables_utils import (
    get_padj_lfc_from_method,
)
from paper_experiments.utils.constants import MetaAnalysisParameter


def build_lfc_lfc_and_padj_padj_plot(
    method_name: str,
    method_results_path: str | Path,
    pydeseq2_results_path: str | Path,
    save_path: str | Path,
    dataset_name: TCGADatasetNames,
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    reference_dds_ref_level: tuple[str, str] | None = ("stage", "Advanced"),
    meta_analysis_parameters: list[MetaAnalysisParameter] | None = None,
    log2fc_threshold: float = 2.0,
    padj_threshold: float = 0.05,
    nlog10pval_clip: float | None = 10,
    log2fold_clip: float | None = 10,
    **pydeseq2_kwargs: Any,
):
    """Build a lfc / lfc and a pad / padj plot for a given method.

    Compares the log-fold changes and adjusted p-values of a method with PyDESeq2.

    Parameters
    ----------
    method_name : str
        The name of the method to build the plots for.

    method_results_path : str or Path
        The path to the method results.

    pydeseq2_results_path : str or Path
        The path to the PyDESeq2 results.

    save_path : str or Path
        The path where to save the lfc / lfc and padj / padj plots.

    dataset_name : TCGADatasetNames
        The name of the dataset to build the lfc / lfc and padj / padj plots for.

    small_samples : bool
        Whether to use the small samples dataset.

    small_genes : bool
        Whether to use the small genes dataset.

    only_two_centers : bool
        Whether to use only two centers.

    design_factors : str or list of str
        The design factors used in the analysis.

    continuous_factors : list of str or None
        The continuous factors used in the analysis.

    reference_dds_ref_level : tuple[str, str] or None
        The reference dds ref level to use.
        Necessary if the test method is PyDESeq2 per center or pooled.

    meta_analysis_parameters : list[MetaAnalysisParameter] or None
        The meta-analysis parameters to use.
        Necessary if the test method is Meta-analysis.

    log2fc_threshold : float
        The log-fold change threshold to consider a gene as differentially expressed.

    padj_threshold : float
        The adjusted p-value threshold to consider a gene as differentially expressed.

    nlog10pval_clip : float or None
        The maximum value to clip the -log10(p-value) values to.

    log2fold_clip : float or None
        The maximum value to clip the log2 fold change values to.

    pydeseq2_kwargs : Any
        Additional keyword arguments to pass to the PyDESeq2 method.
    """
    experiment_id = get_experiment_id(
        dataset_name=dataset_name,
        small_samples=small_samples,
        small_genes=small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        heterogeneity_method=None,
        heterogeneity_method_param=None,
        **pydeseq2_kwargs,
    )

    refit_cooks = pydeseq2_kwargs.get("refit_cooks", True)
    method_padj, method_lfc = get_padj_lfc_from_method(
        method_name,
        method_results_path,
        experiment_id,
        refit_cooks=refit_cooks,
        reference_dds_ref_level=reference_dds_ref_level,
        meta_analysis_parameters=meta_analysis_parameters,
    )

    # Also get the pydeseq2 results to compare
    pydeseq2_experiment_id = get_experiment_id(
        dataset_name=dataset_name,
        small_samples=small_samples,
        small_genes=small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        **pydeseq2_kwargs,
    )

    pydeseq2_padj, pydeseq2_lfc = get_padj_lfc_from_method(
        "pydeseq2",
        pydeseq2_results_path,
        pydeseq2_experiment_id,
        refit_cooks=refit_cooks,
        reference_dds_ref_level=reference_dds_ref_level,
        meta_analysis_parameters=meta_analysis_parameters,
    )

    assert isinstance(pydeseq2_padj, pd.Series)
    assert isinstance(pydeseq2_lfc, pd.Series)

    logger.info(
        f"Building lfc-lfc and padj-padj plots for {dataset_name} ({method_name})."
    )

    if isinstance(method_padj, dict):
        assert isinstance(method_lfc, dict)
        assert set(method_padj.keys()) == set(method_lfc.keys())
        for method_id in method_padj.keys():
            # Extract the meta-analysis submethod name in a more readable format
            submethod_name_list = method_id.split(", ")[1:]
            submethod_name = "_".join(
                [param for param in submethod_name_list if param != "None"]
            )

            lfc_save_file_path = (
                save_path
                / experiment_id
                / f"lfc_lfc_plot_{method_name}_{submethod_name}.pdf"
            )

            padj_save_file_path = (
                save_path
                / experiment_id
                / f"padj_padj_plot_{method_name}_{submethod_name}.pdf"
            )

            make_lfc_lfc_plot(
                method_padj[method_id],
                method_lfc[method_id],
                pydeseq2_lfc,
                padj_threshold,
                log2fc_threshold,
                save_file_path=lfc_save_file_path,
                plot_title=f"{NAME_MAPPING[method_name]} ({submethod_name})",
                annotate_genes=False,
                write_legend=True,
                pydeseq2_padj=pydeseq2_padj,
                log2fold_clip=log2fold_clip,
            )

            make_padj_padj_plot(
                method_padj[method_id],
                method_lfc[method_id],
                pydeseq2_padj,
                padj_threshold,
                log2fc_threshold,
                save_file_path=padj_save_file_path,
                plot_title=f"{NAME_MAPPING[method_name]} ({submethod_name})",
                annotate_genes=False,
                write_legend=True,
                pydeseq2_lfc=pydeseq2_lfc,
                nlog10pval_clip=nlog10pval_clip,
            )

    else:
        lfc_save_file_path = (
            save_path / experiment_id / f"lfc_lfc_plot_{method_name}.pdf"
        )

        padj_save_file_path = (
            save_path / experiment_id / f"padj_padj_plot_{method_name}.pdf"
        )
        assert isinstance(method_padj, pd.Series)
        assert isinstance(method_lfc, pd.Series)

        make_lfc_lfc_plot(
            method_padj,
            method_lfc,
            pydeseq2_lfc,
            padj_threshold,
            log2fc_threshold,
            save_file_path=lfc_save_file_path,
            plot_title=f"{NAME_MAPPING[method_name]}",
            annotate_genes=False,
            write_legend=True,
            pydeseq2_padj=pydeseq2_padj,
            log2fold_clip=log2fold_clip,
        )

        make_padj_padj_plot(
            method_padj,
            method_lfc,
            pydeseq2_padj,
            padj_threshold,
            log2fc_threshold,
            save_file_path=padj_save_file_path,
            plot_title=f"{NAME_MAPPING[method_name]}",
            annotate_genes=False,
            write_legend=True,
            pydeseq2_lfc=pydeseq2_lfc,
            nlog10pval_clip=nlog10pval_clip,
        )


def make_lfc_lfc_plot(
    method_padj: pd.Series,
    method_lfc: pd.Series,
    pydeseq2_lfc: pd.Series,
    padj_threshold: float,
    log2fc_threshold: float,
    save_file_path: str | Path,
    plot_title: str,
    annotate_genes: bool = False,
    write_legend: bool = True,
    pydeseq2_padj: pd.Series | None = None,
    log2fold_clip: float | None = 10,
):
    """Create a plot comparing log2-fold changes of a method with PyDESeq2.

    Summarizes the results of a differential expression analysis by plotting
    the log2 fold changes of a tested method against the log2 fold change of pydeseq2,
    showing true/false positives/negatives for both up and down-regulated genes.
    If pydeseq2 adjusted p-values are also provided, points will be colored according
    to whether they are true/false positives/negatives up/down-regulated genes.

    Parameters
    ----------
    method_padj : pd.Series
        The adjusted p-values of the method.

    method_lfc : pd.Series
        The log-fold changes of the method.

    pydeseq2_lfc : pd.Series
        The log-fold changes of PyDESeq2.

    padj_threshold : float
        The adjusted p-value threshold to consider a gene as differentially expressed.

    log2fc_threshold : float
        The log-fold change threshold to consider a gene as differentially expressed.

    save_file_path :str or Path
        The path where to save the plot.

    plot_title : str
        The title to display on the plot.

    annotate_genes : bool
        Whether or not to annotate genes that pass the LFC and p-value thresholds.
        (default: ``True``).

    write_legend : bool
        Whether or not to write the legend on the plot. (default: ``True``).

    pydeseq2_padj : pd.Series or None
        The adjusted p-values of PyDESeq2. If not None, will be used to color the points
        in the plot according to whether they are true/false positives/negatives
        up/down-regulated genes.

    log2fold_clip : float or None
        The maximum value to clip the log2 fold change values to.
    """
    plt.clf()

    fig, ax = plt.subplots(figsize=(8, 8))

    nlog10_padj_threshold = -np.log10(padj_threshold)

    def map_DE(row):
        log2FoldChange = row.log2FoldChange
        nlog10 = row.nlog10

        if ("pydeseq2_padj" not in row.index) or ("pydeseq2_nlog10" not in row.index):
            # As there is no pydeseq2 reference, genes will be considered as a true
            # positive if they meet the p-value and log2 fold change thresholds.
            if nlog10 > nlog10_padj_threshold:
                if log2FoldChange > log2fc_threshold:
                    return "true_pos_upreg"
                elif log2FoldChange < -log2fc_threshold:
                    return "true_pos_downreg"
            return "none"

        pydeseq2_nlog10 = row.pydeseq2_nlog10
        pydeseq2_log2FoldChange = row.pydeseq2_log2FoldChange

        if nlog10 > nlog10_padj_threshold:
            if pydeseq2_nlog10 > nlog10_padj_threshold:
                # Both methods meet the p-value threshold
                if log2FoldChange > log2fc_threshold:
                    # The tested method also meets one LFC threshold.
                    return (
                        "true_pos_upreg"
                        if pydeseq2_log2FoldChange > log2fc_threshold
                        else "false_pos_upreg"
                    )
                elif log2FoldChange < -log2fc_threshold:
                    # The tested method also meets one LFC threshold.
                    return (
                        "true_pos_downreg"
                        if pydeseq2_log2FoldChange < -log2fc_threshold
                        else "false_pos_downreg"
                    )
                elif pydeseq2_log2FoldChange > log2fc_threshold:
                    return "false_neg_upreg"
                elif pydeseq2_log2FoldChange < -log2fc_threshold:
                    return "false_neg_downreg"
            else:
                # Only the tested method meets the p-value threshold
                if log2FoldChange > log2fc_threshold:
                    return "false_pos_upreg"
                elif log2FoldChange < -log2fc_threshold:
                    return "false_pos_downreg"
        elif pydeseq2_nlog10 > nlog10_padj_threshold:
            # Only PyDESeq2 meets the p-value threshold
            if pydeseq2_log2FoldChange > log2fc_threshold:
                return "false_neg_upreg"
            elif pydeseq2_log2FoldChange < -log2fc_threshold:
                return "false_neg_downreg"

        return "none"

    df = pd.DataFrame(
        {
            "log2FoldChange": method_lfc / np.log(2),
            "pydeseq2_log2FoldChange": pydeseq2_lfc / np.log(2),
            "padj": method_padj,
        }
    )

    df["nlog10"] = -df.apply(lambda x: np.log10(x["padj"]), axis=1)

    if pydeseq2_padj is not None:
        df["pydeseq2_padj"] = pydeseq2_padj
        df["pydeseq2_nlog10"] = -df.apply(
            lambda x: np.log10(x["pydeseq2_padj"]), axis=1
        )

    df["DE"] = df.apply(map_DE, axis=1)

    df["clipped"] = False

    if log2fold_clip is not None:
        df["log2FoldChange"] = df["log2FoldChange"].clip(
            upper=log2fold_clip, lower=-log2fold_clip
        )
        df["clipped"] = df["clipped"] | (df["log2FoldChange"].abs() == log2fold_clip)

        df["pydeseq2_log2FoldChange"] = df["pydeseq2_log2FoldChange"].clip(
            upper=log2fold_clip, lower=-log2fold_clip
        )

        df["clipped"] = df["clipped"] | (
            df["pydeseq2_log2FoldChange"].abs() == log2fold_clip
        )

    sns.scatterplot(
        data=df[(~df["clipped"]) & (df["DE"] == "none")],
        x="pydeseq2_log2FoldChange",
        y="log2FoldChange",
        color="lightgrey",
        s=40,
        ax=ax,
        zorder=10,
    )

    sns.scatterplot(
        data=df[(~df["clipped"]) & (df["DE"] != "none")],
        x="pydeseq2_log2FoldChange",
        y="log2FoldChange",
        hue="DE",
        hue_order=[
            "true_pos_upreg",
            "true_pos_downreg",
            "false_pos_upreg",
            "false_pos_downreg",
            "false_neg_upreg",
            "false_neg_downreg",
            "none",
        ],
        palette=[
            "orangered",
            "blue",
            "lawngreen",
            "aqua",
            "orange",
            "darkorchid",
            "lightgrey",
        ],
        s=40,
        ax=ax,
        zorder=100,
    )

    if log2fold_clip is not None:
        # Plot points with log2FoldChange == log2fold_clip with '<', '>', '^' or 'V'
        # marker depending on the sign of log2FoldChange and pydeseq2_log2FoldChange
        ax.set_xlim(
            max(-log2fold_clip - 0.1, df["pydeseq2_log2FoldChange"].min() - 0.5),
            min(log2fold_clip + 0.1, df["pydeseq2_log2FoldChange"].max() + 0.5),
        )
        ax.set_ylim(
            max(-log2fold_clip - 0.1, df["log2FoldChange"].min() - 0.5),
            min(log2fold_clip + 0.1, df["log2FoldChange"].max() + 0.5),
        )

        def determine_marker(row):
            if row["log2FoldChange"] == log2fold_clip:
                return "up"
            elif row["log2FoldChange"] == -log2fold_clip:
                return "down"
            elif row["pydeseq2_log2FoldChange"] == log2fold_clip:
                return "right"
            else:
                return "left"

        df["marker"] = df.apply(determine_marker, axis=1)

        sns.scatterplot(
            data=df[df["clipped"]],
            x="pydeseq2_log2FoldChange",
            y="log2FoldChange",
            hue="DE",
            hue_order=[
                "true_pos_upreg",
                "true_pos_downreg",
                "false_pos_upreg",
                "false_pos_downreg",
                "false_neg_upreg",
                "false_neg_downreg",
                "none",
            ],
            palette=[
                "orangered",
                "blue",
                "lawngreen",
                "aqua",
                "orange",
                "darkorchid",
                "lightgrey",
            ],
            style=df[df["clipped"]]["marker"],
            markers={"left": "<", "right": ">", "up": "^", "down": "v"},
            s=40,
            zorder=2,
            ax=ax,
        )

    # add a x=y line
    ax.plot(
        [df["pydeseq2_log2FoldChange"].min(), df["pydeseq2_log2FoldChange"].max()],
        [df["pydeseq2_log2FoldChange"].min(), df["pydeseq2_log2FoldChange"].max()],
        color="black",
        linestyle="--",
        linewidth=2,
    )

    if annotate_genes:
        texts = []
        for i in range(len(df)):
            if (
                df.iloc[i].nlog10 > nlog10_padj_threshold
                and abs(df.iloc[i].log2FoldChange) > log2fc_threshold
            ):
                texts.append(
                    plt.text(
                        x=df.iloc[i].pydeseq2_log2FoldChange,
                        y=df.iloc[i].log2FoldChange,
                        s=df.index[i],
                        fontsize=12,
                        weight="bold",
                    )
                )

        adjust_text(texts, arrowprops={"arrowstyle": "-", "color": "k"})

    ax.get_legend().remove()
    if write_legend:
        if pydeseq2_padj is not None:
            # create manual symbols for legend
            up_TP = Line2D(
                [0],
                [0],
                label="Up (TP)",
                marker="o",
                color="orangered",
                linestyle="",
            )
            up_FP = Line2D(
                [0],
                [0],
                label="Up (FP)",
                marker="o",
                color="lawngreen",
                linestyle="",
            )
            up_FN = Line2D(
                [0],
                [0],
                label="Up (FN)",
                marker="o",
                color="orange",
                linestyle="",
            )
            down_TP = Line2D(
                [0],
                [0],
                label="Down (TP)",
                marker="o",
                color="blue",
                linestyle="",
            )
            down_FP = Line2D(
                [0],
                [0],
                label="Down (FP)",
                marker="o",
                color="aqua",
                linestyle="",
            )
            down_FN = Line2D(
                [0],
                [0],
                label="Down (FN)",
                marker="o",
                color="darkorchid",
                linestyle="",
            )
            none = Line2D(
                [0],
                [0],
                label="None",
                marker="o",
                color="lightgrey",
                linestyle="",
            )

            # add manual symbols to auto legend
            handles = [up_TP, up_FP, up_FN, down_TP, down_FP, down_FN, none]

        else:
            # There are no PyDESeq2 results to compare with
            up_TP = Line2D(
                [0],
                [0],
                label="Up",
                marker="o",
                color="orangered",
                linestyle="",
            )

            down_TP = Line2D(
                [0],
                [0],
                label="Down",
                marker="o",
                color="blue",
                linestyle="",
            )

            none = Line2D(
                [0],
                [0],
                label="None",
                marker="o",
                color="lightgrey",
                linestyle="",
            )

            # add manual symbols to auto legend
            handles = [up_TP, down_TP, none]

        plt.legend(
            handles=handles,
            loc=1,
            bbox_to_anchor=(1.4, 1),
            frameon=False,
            prop={"weight": "bold"},
        )

    for axis in ["bottom", "left"]:
        ax.spines[axis].set_linewidth(2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(width=2)

    plt.xticks(size=12, weight="bold")
    plt.yticks(size=12, weight="bold")

    plt.xlabel("$log_{2}$ fold change (pydeseq2)", size=15)
    plt.ylabel(f"$log_{2}$ fold change ({plot_title})", size=15)

    plt.title(plot_title, size=16, weight="bold")

    plt.tight_layout()
    save_file_path = Path(save_file_path)
    save_file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file_path, transparent=True)
    plt.close()


def make_padj_padj_plot(
    method_padj: pd.Series,
    method_lfc: pd.Series,
    pydeseq2_padj: pd.Series,
    padj_threshold: float,
    log2fc_threshold: float,
    save_file_path: str | Path,
    plot_title: str,
    annotate_genes: bool = False,
    write_legend: bool = True,
    pydeseq2_lfc: pd.Series | None = None,
    nlog10pval_clip: float | None = 10,
):
    """Create a plot comparing adjusted p-values of a method with PyDESeq2.

    This plot is done in log scale.

    Summarizes the results of a differential expression analysis by plotting
    the adjusted p-values of a tested method against the adjusted p-values of pydeseq2,
    showing true/false positives/negatives for both up and down-regulated genes.
    If pydeseq2 LFC are also provided, points will be colored according
    to whether they are true/false positives/negatives up/down-regulated genes.

    Parameters
    ----------
    method_padj : pd.Series
        The adjusted p-values of the method.

    method_lfc : pd.Series
        The log-fold changes of the method.

    pydeseq2_padj : pd.Series
        The adjusted p-values of PyDESeq2.

    padj_threshold : float
        The adjusted p-value threshold to consider a gene as differentially expressed.

    log2fc_threshold : float
        The log-fold change threshold to consider a gene as differentially expressed.

    save_file_path :str or Path
        The path where to save the plot.

    plot_title : str
        The title to display on the plot.

    annotate_genes : bool
        Whether or not to annotate genes that pass the LFC and p-value thresholds.
        (default: ``True``).

    write_legend : bool
        Whether or not to write the legend on the plot. (default: ``True``).

    pydeseq2_lfc : pd.Series
        The log-fold changes of PyDESeq2.
        If not None, will be used to color the points in the plot according to whether
        they are true/false positives/negatives up/down-regulated genes.

    nlog10pval_clip : float or None
        The maximum value to clip the -log10(p-value) values to.
    """
    plt.clf()

    fig, ax = plt.subplots(figsize=(8, 8))

    nlog10_padj_threshold = -np.log10(padj_threshold)

    def map_DE(row):
        log2FoldChange = row.log2FoldChange
        nlog10 = row.nlog10

        if ("pydeseq2_padj" not in row.index) or ("pydeseq2_nlog10" not in row.index):
            # As there is no pydeseq2 reference, genes will be considered as a true
            # positive if they meet the p-value and log2 fold change thresholds.
            if nlog10 > nlog10_padj_threshold:
                if log2FoldChange > log2fc_threshold:
                    return "true_pos_upreg"
                elif log2FoldChange < -log2fc_threshold:
                    return "true_pos_downreg"
            return "none"

        pydeseq2_nlog10 = row.pydeseq2_nlog10
        pydeseq2_log2FoldChange = row.pydeseq2_log2FoldChange

        if nlog10 > nlog10_padj_threshold:
            if pydeseq2_nlog10 > nlog10_padj_threshold:
                # Both methods meet the p-value threshold
                if log2FoldChange > log2fc_threshold:
                    # The tested method also meets one LFC threshold.
                    return (
                        "true_pos_upreg"
                        if pydeseq2_log2FoldChange > log2fc_threshold
                        else "false_pos_upreg"
                    )
                elif log2FoldChange < -log2fc_threshold:
                    # The tested method also meets one LFC threshold.
                    return (
                        "true_pos_downreg"
                        if pydeseq2_log2FoldChange < -log2fc_threshold
                        else "false_pos_downreg"
                    )
                elif pydeseq2_log2FoldChange > log2fc_threshold:
                    return "false_neg_upreg"
                elif pydeseq2_log2FoldChange < -log2fc_threshold:
                    return "false_neg_downreg"
            else:
                # Only the tested method meets the p-value threshold
                if log2FoldChange > log2fc_threshold:
                    return "false_pos_upreg"
                elif log2FoldChange < -log2fc_threshold:
                    return "false_pos_downreg"
        elif pydeseq2_nlog10 > nlog10_padj_threshold:
            # Only PyDESeq2 meets the p-value threshold
            if pydeseq2_log2FoldChange > log2fc_threshold:
                return "false_neg_upreg"
            elif pydeseq2_log2FoldChange < -log2fc_threshold:
                return "false_neg_downreg"

        return "none"

    df = pd.DataFrame(
        {
            "log2FoldChange": method_lfc / np.log(2),
            "pydeseq2_padj": pydeseq2_padj,
            "padj": method_padj,
        }
    )

    df["nlog10"] = -df.apply(lambda x: np.log10(x["padj"]), axis=1)
    df["pydeseq2_nlog10"] = -df.apply(lambda x: np.log10(x["pydeseq2_padj"]), axis=1)

    if pydeseq2_lfc is not None:
        df["pydeseq2_log2FoldChange"] = pydeseq2_lfc / np.log(2)

    df["DE"] = df.apply(map_DE, axis=1)

    df["clipped"] = False

    if nlog10pval_clip is not None:
        df["nlog10"] = df["nlog10"].clip(upper=nlog10pval_clip, lower=-nlog10pval_clip)
        df["clipped"] = df["clipped"] | (df["nlog10"].abs() == nlog10pval_clip)

        df["pydeseq2_nlog10"] = df["pydeseq2_nlog10"].clip(
            upper=nlog10pval_clip, lower=-nlog10pval_clip
        )

        df["clipped"] = df["clipped"] | (df["pydeseq2_nlog10"].abs() == nlog10pval_clip)

    sns.scatterplot(
        data=df[(~df["clipped"]) & (df["DE"] == "none")],
        x="pydeseq2_nlog10",
        y="nlog10",
        color="lightgrey",
        s=40,
        ax=ax,
        zorder=10,
    )

    sns.scatterplot(
        data=df[(~df["clipped"]) & (df["DE"] != "none")],
        x="pydeseq2_nlog10",
        y="nlog10",
        hue="DE",
        hue_order=[
            "true_pos_upreg",
            "true_pos_downreg",
            "false_pos_upreg",
            "false_pos_downreg",
            "false_neg_upreg",
            "false_neg_downreg",
            "none",
        ],
        palette=[
            "orangered",
            "blue",
            "lawngreen",
            "aqua",
            "orange",
            "darkorchid",
            "lightgrey",
        ],
        s=40,
        ax=ax,
        zorder=100,
    )

    if nlog10pval_clip is not None:
        # Plot points with nlog10 == nlog10pval_clip with '<', '>', '^' or 'V'
        # marker depending on the sign of nlog10 and pydeseq2_nlog10
        ax.set_xlim(
            max(-nlog10pval_clip - 0.1, df["pydeseq2_nlog10"].min() - 0.5),
            min(nlog10pval_clip + 0.1, df["pydeseq2_nlog10"].max() + 0.5),
        )
        ax.set_ylim(
            max(-nlog10pval_clip - 0.1, df["nlog10"].min() - 0.5),
            min(nlog10pval_clip + 0.1, df["nlog10"].max() + 0.5),
        )

        def determine_marker(row):
            if row["nlog10"] == nlog10pval_clip:
                return "up"
            elif row["nlog10"] == -nlog10pval_clip:
                return "down"
            elif row["pydeseq2_nlog10"] == nlog10pval_clip:
                return "right"
            else:
                return "left"

        df["marker"] = df.apply(determine_marker, axis=1)

        sns.scatterplot(
            data=df[df["clipped"]],
            x="pydeseq2_nlog10",
            y="nlog10",
            hue="DE",
            hue_order=[
                "true_pos_upreg",
                "true_pos_downreg",
                "false_pos_upreg",
                "false_pos_downreg",
                "false_neg_upreg",
                "false_neg_downreg",
                "none",
            ],
            palette=[
                "orangered",
                "blue",
                "lawngreen",
                "aqua",
                "orange",
                "darkorchid",
                "lightgrey",
            ],
            style=df[df["clipped"]]["marker"],
            markers={"left": "<", "right": ">", "up": "^", "down": "v"},
            s=40,
            zorder=2,
            ax=ax,
        )

    # add a x=y line
    ax.plot(
        [df["pydeseq2_nlog10"].min(), df["pydeseq2_nlog10"].max()],
        [df["pydeseq2_nlog10"].min(), df["pydeseq2_nlog10"].max()],
        color="black",
        linestyle="--",
        linewidth=2,
    )

    if annotate_genes:
        texts = []
        for i in range(len(df)):
            if (
                df.iloc[i].nlog10 > nlog10_padj_threshold
                and abs(df.iloc[i].log2FoldChange) > log2fc_threshold
            ):
                texts.append(
                    plt.text(
                        x=df.iloc[i].pydeseq2_nlog10,
                        y=df.iloc[i].log2FoldChange,
                        s=df.index[i],
                        fontsize=12,
                        weight="bold",
                    )
                )

        adjust_text(texts, arrowprops={"arrowstyle": "-", "color": "k"})

    ax.get_legend().remove()
    if write_legend:
        if pydeseq2_lfc is not None:
            # create manual symbols for legend
            up_TP = Line2D(
                [0],
                [0],
                label="Up (TP)",
                marker="o",
                color="orangered",
                linestyle="",
            )
            up_FP = Line2D(
                [0],
                [0],
                label="Up (FP)",
                marker="o",
                color="lawngreen",
                linestyle="",
            )
            up_FN = Line2D(
                [0],
                [0],
                label="Up (FN)",
                marker="o",
                color="orange",
                linestyle="",
            )
            down_TP = Line2D(
                [0],
                [0],
                label="Down (TP)",
                marker="o",
                color="blue",
                linestyle="",
            )
            down_FP = Line2D(
                [0],
                [0],
                label="Down (FP)",
                marker="o",
                color="aqua",
                linestyle="",
            )
            down_FN = Line2D(
                [0],
                [0],
                label="Down (FN)",
                marker="o",
                color="darkorchid",
                linestyle="",
            )
            none = Line2D(
                [0],
                [0],
                label="None",
                marker="o",
                color="lightgrey",
                linestyle="",
            )

            # add manual symbols to auto legend
            handles = [up_TP, up_FP, up_FN, down_TP, down_FP, down_FN, none]

        else:
            # There are no PyDESeq2 results to compare with
            up_TP = Line2D(
                [0],
                [0],
                label="Up",
                marker="o",
                color="orangered",
                linestyle="",
            )

            down_TP = Line2D(
                [0],
                [0],
                label="Down",
                marker="o",
                color="blue",
                linestyle="",
            )

            none = Line2D(
                [0],
                [0],
                label="None",
                marker="o",
                color="lightgrey",
                linestyle="",
            )

            # add manual symbols to auto legend
            handles = [up_TP, down_TP, none]

        plt.legend(
            handles=handles,
            loc=1,
            bbox_to_anchor=(1.4, 1),
            frameon=False,
            prop={"weight": "bold"},
        )

    for axis in ["bottom", "left"]:
        ax.spines[axis].set_linewidth(2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(width=2)

    plt.xticks(size=12, weight="bold")
    plt.yticks(size=12, weight="bold")

    plt.xlabel("-$log_{10}$ FDR (pydeseq2)", size=15)
    plt.ylabel(f"-$log_{{10}}$ FDR ({plot_title})", size=15)

    plt.title(plot_title, size=16, weight="bold")

    plt.tight_layout()
    save_file_path = Path(save_file_path)
    save_file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file_path)
    plt.close()
