import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fedpydeseq2_datasets.constants import TCGADatasetNames
from fedpydeseq2_datasets.utils import get_experiment_id
from fedpydeseq2_datasets.utils import get_ground_truth_dds_name
from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend
from loguru import logger
from scipy.stats import combine_pvalues
from scipy.stats import false_discovery_control
from scipy.stats import norm
from statsmodels.stats.meta_analysis import combine_effects


def run_tcga_meta_analysis_experiments(
    dataset_name: TCGADatasetNames,
    stats_per_center_path: str | Path,
    stats_meta_path: str | Path,
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    reference_dds_ref_level: tuple[str, str] = ("stage", "Advanced"),
    meta_analysis_type: str = "random_effect",
    method_random_effects: str | None = "dl",
    method_combination: str | None = "stouffer",
    ignore_nan_centers: bool = True,
    stats_clip_value: float | None = None,
    heterogeneity_method: str | None = None,
    heterogeneity_method_param: float | None = None,
    **pydeseq2_kwargs: Any,
):
    """Run TCGA meta-analysis experiments.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to use.

    stats_per_center_path : Union[str, Path]
        The path to the results of the DS.

    stats_meta_path : Union[str, Path]
        The path where the output will be saved.

    small_samples : bool, optional
        If True, use small samples, by default False.

    small_genes : bool, optional
        If True, use small genes, by default False.

    only_two_centers : bool, optional
        If True, use only two centers, by default False.

    design_factors : Union[str, list[str]], optional
        The design factors to use, by default "stage".

    continuous_factors : Optional[list[str]], optional
        The continuous factors to use, by default None.

    reference_dds_ref_level : tuple[str, str], optional
        The reference level for the DDS, by default ("stage", "Advanced").

    meta_analysis_type : str, optional
        The type of meta-analysis to use, by default "random_effect".
        Can be in ["pvalue_combination", "fixed_effect", "random_effect"].

    method_random_effects : Optional[str], optional
        The method for random effects, by default "dl".
        Can be in ["dl", "iterated", "chi2"].

    method_combination : Optional[str], optional
        The method for combination, by default "stouffer".
        Can be in ["stouffer", "fisher"].

    ignore_nan_centers : bool, optional
        If True, ignore NaN centers in meta-analysis combination. This helps reducing
        the number of genes with a NaN output. (default: True).

    stats_clip_value : Optional[float], optional
        The value to clip the statistics to, by default None.
        If None, statistics are clipped in the interval
        [-1e6, 1e6].

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
        Additional arguments to pass to the pydeseq2 function.
    """
    meta_analysis_id = get_meta_analysis_id(
        meta_analysis_type, method_random_effects, method_combination, stats_clip_value
    )
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
    ground_truth_dds_name = get_ground_truth_dds_name(
        reference_dds_ref_level=reference_dds_ref_level,
        refit_cooks=refit_cooks,
        pooled=False,
    )

    # The number of centers is saved in a n_centers.txt at the root of the
    # stats_per_center_path
    with open(Path(stats_per_center_path) / experiment_id / "n_centers.txt") as f:
        n_centers = int(f.read())

    # Load all local stats results
    local_stats_results = {}
    local_sizes = {}
    for center_id in range(n_centers):
        # Load the stats results

        stats_path = (
            stats_per_center_path
            / experiment_id
            / f"center_{center_id}"
            / f"{ground_truth_dds_name}_stats_res.pkl"
        )
        with open(stats_path, "rb") as f:
            stats_res = pickle.load(f)
        if stats_res is not None:
            local_stats_results[center_id] = stats_res["results_df"]
            # Load the local size from the dds
            local_sizes[center_id] = stats_res["n_obs"]
        else:
            logger.warning(
                f"Center {center_id} has no results, because did not have a full design"
            )

    center_ids = list(local_stats_results.keys())
    sizes = np.array([local_sizes[pid] for pid in center_ids])
    gene_names = get_common_genes(local_stats_results)
    local_lfcs = np.stack(
        [
            local_stats_results[pid]["lfc"].loc[gene_names].to_numpy()
            for pid in center_ids
        ]
    )
    local_lfcVars = np.stack(
        [
            ((local_stats_results[pid]["lfcSE"].loc[gene_names]) ** 2).to_numpy()
            for pid in center_ids
        ]
    )

    local_p_values = np.stack(
        [
            local_stats_results[pid]["pvalue"].loc[gene_names].to_numpy()
            for pid in center_ids
        ]
    )

    # Here, add the option to ignore NaN centers

    if meta_analysis_type in {"fixed_effect", "random_effect"}:
        with parallel_backend("loky"):
            res = Parallel(n_jobs=-1, verbose=0)(
                delayed(run_statsmodels_meta_analysis_single_gene)(
                    local_lfcs[:, i],
                    local_lfcVars[:, i],
                    meta_analysis_type,
                    method_random_effects,
                    ignore_nan_centers,
                )
                for i in range(len(gene_names))
            )

    elif meta_analysis_type == "pvalue_combination":
        with parallel_backend("loky"):
            res = Parallel(n_jobs=-1, verbose=0)(
                delayed(run_pvalue_combination_single_gene)(
                    local_lfcs[:, i], local_p_values[:, i], sizes, method_combination
                )
                for i in range(len(gene_names))
            )

    meta_analysis_results = pd.DataFrame(res, index=gene_names)
    # Create false discovery control on p-values
    meta_analysis_results["padj"] = pd.NA
    meta_analysis_results["padj"] = pd.NA

    meta_analysis_results.loc[~meta_analysis_results["pvalue"].isna(), "padj"] = (
        false_discovery_control(
            meta_analysis_results.loc[~meta_analysis_results["pvalue"].isna(), "pvalue"]
        )
    )

    # Save the results
    output_file = Path(stats_meta_path) / experiment_id / f"{meta_analysis_id}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    meta_analysis_results.index.name = "gene"
    # Drop rows with NA values
    meta_analysis_results.dropna(subset=["padj", "stat"], inplace=True)
    # clamp statistics to avoid infinity values
    if stats_clip_value is None:
        logger.info("Parameter stats_clip_value is None, clamping statistics to 1e6")
        stats_clip_value = 1e6

    large_genes = meta_analysis_results[
        (meta_analysis_results["stat"] > stats_clip_value)
        | (meta_analysis_results["stat"] < -stats_clip_value)
    ]
    if not large_genes.empty:
        logger.warning(
            "Found large values in the statistics "
            f"for {len(large_genes)} genes. Clipping to {stats_clip_value}."
        )
        meta_analysis_results["stat"] = np.clip(
            meta_analysis_results["stat"], -stats_clip_value, stats_clip_value
        )

    meta_analysis_results.to_csv(output_file)
    logger.success(f"Saved the meta analysis results at {output_file}")


def get_meta_analysis_id(
    meta_analysis_type: str,
    method_random_effects: str | None = None,
    method_combination: str | None = None,
    stats_clip_value: float | None = None,
) -> str:
    """Get the meta-analysis id.

    Parameters
    ----------
    meta_analysis_type : str
        The type of meta-analysis to use.
        Can be in ["pvalue_combination", "fixed_effect", "random_effect"].

    method_random_effects : str or None
        The method for random effects, by default None.
        Can be in ["dl", "iterated", "chi2"].
        Must be provided when meta_analysis_type is 'random_effect'.

    method_combination : str or None
        The method for combination, by default None.
        Can be in ["stouffer", "fisher"].
        Must be provided when meta_analysis_type is 'pvalue_combination'.

    stats_clip_value : float or None
        The value to clip the statistics to, by default None.
        If None, statistics are clipped in the interval
        [-1e6, 1e6], but this value is not specified in the id.

    Returns
    -------
    str
        The meta-analysis id.
    """
    assert meta_analysis_type in {
        "pvalue_combination",
        "fixed_effect",
        "random_effect",
    }, (
        "meta_analysis_type should be in ['pvalue_combination', "
        "'fixed_effect','random_effect']"
        f"Got {meta_analysis_type}"
    )
    if meta_analysis_type == "random_effect":
        assert method_random_effects is not None, (
            "method_random_effects should be provided "
            "when meta_analysis_type is 'random_effect'"
        )
        assert method_random_effects in {"dl", "iterated", "chi2"}, (
            "method_random_effects should be in ['dl', 'iterated', 'chi2']"
            f"Got {method_random_effects}"
        )
        main_id = f"meta_analysis--{meta_analysis_type}--{method_random_effects}"
    elif meta_analysis_type == "pvalue_combination":
        assert method_combination is not None, (
            "method_combination should be provided when meta_analysis_type "
            "is 'pvalue_combination'"
        )
        assert method_combination in {"stouffer", "fisher"}, (
            "method_combination should be in ['stouffer', 'fisher']"
            f"Got {method_combination}"
        )
        main_id = f"meta_analysis--{meta_analysis_type}--{method_combination}"
    else:
        main_id = f"meta_analysis--{meta_analysis_type}"
    if stats_clip_value is not None:
        return f"{main_id}--{stats_clip_value}"
    return main_id


def get_common_genes(local_stats_results: dict, dropna: bool = False) -> pd.Index:
    """Get the common genes between all the local stats results.

    Parameters
    ----------
    local_stats_results : dict
        The mapping between center id and the stats results, as a dataframe.

    dropna : bool
        If True, drop the na values, by default False.


    Returns
    -------
    pd.Index
        The index of the common genes.
    """
    cols = ["lfc", "lfcSE"]
    # get first key
    keys = list(local_stats_results.keys())
    first_key = keys[0]
    # get first df
    if dropna:
        common_genes = local_stats_results[first_key][cols].dropna().index
    else:
        common_genes = local_stats_results[first_key].index

    for key in keys[1:]:
        if dropna:
            common_genes = common_genes.intersection(
                local_stats_results[key][cols].dropna().index
            )
        else:
            common_genes = common_genes.intersection(local_stats_results[key].index)
    return common_genes


def run_statsmodels_meta_analysis_single_gene(
    lfcs: np.ndarray,
    lfcVars: np.ndarray,
    meta_analysis_type: str = "random_effect",
    method_random_effects: str | None = "dl",
    ignore_nan_centers: bool = True,
) -> pd.Series:
    """Run the statsmodels meta-analysis for a single gene.

    Parameters
    ----------
    lfcs : np.ndarray
        The log fold changes for the gene, per center (of size (n_centers,)).

    lfcVars : np.ndarray
        The log fold change standard errors for the gene,
        per center (of size (n_centers,)).

    meta_analysis_type : str
        The type of meta-analysis to use, by default "random_effect".
        Can be in ["fixed_effect", "random_effect"].

    method_random_effects : str or None
        The method for random effects, by default "dl".
        Can be in ["dl", "iterated", "chi2"].
        Must be provided when meta_analysis_type is 'random_effect'.

    ignore_nan_centers : bool, optional
        If True, ignore NaN centers in meta-analysis combination. This helps reducing
        the number of genes with a NaN output. (default: True).

    Returns
    -------
    pd.Series
        The meta-analysis results for the gene
    """
    if meta_analysis_type == "fixed_effect":
        # Set the method_re to "dl" as it does not matter
        # for fixed effect meta-analysis
        current_method_random_effects = "dl"
    else:
        assert method_random_effects is not None, (
            "method_random_effects should be provided "
            "when meta_analysis_type is 'random_effect'"
        )
        current_method_random_effects = method_random_effects

    result_dict = {}

    if ignore_nan_centers:
        if np.isnan(lfcs).all():
            # No analysis can be performed
            result_dict["lfc"] = np.nan
            result_dict["lfcSE"] = np.nan
            result_dict["stat"] = np.nan
            result_dict["pvalue"] = np.nan
            return pd.Series(result_dict)
        else:
            mask = ~np.isnan(lfcs)
            lfcs = lfcs[mask]
            lfcVars = lfcVars[mask]

    meta_analysis_df = combine_effects(
        lfcs,
        lfcVars,
        method_re=current_method_random_effects,
    ).summary_frame()

    meta_analysis_type_space = meta_analysis_type.replace("_", " ")
    result_dict["lfc"] = meta_analysis_df.loc[meta_analysis_type_space, "eff"]
    result_dict["lfcSE"] = meta_analysis_df.loc[meta_analysis_type_space, "sd_eff"]
    result_dict["stat"] = result_dict["lfc"] / result_dict["lfcSE"]
    result_dict["pvalue"] = 2 * norm.sf(np.abs(result_dict["stat"]))

    return pd.Series(result_dict)


def run_pvalue_combination_single_gene(
    lfcs: np.ndarray,
    p_values: np.ndarray,
    sizes: np.ndarray,
    method_combination: str,
    ignore_nan_centers: bool = True,
) -> pd.Series:
    """Run the p-value combination for a single gene.

    Ignores local NaN pvalues and lfc.

    Parameters
    ----------
    lfcs : np.ndarray
        The log fold changes for the gene, per center (of size (n_centers,)).

    p_values : np.ndarray
        The p-values for the gene, per center (of size (n_centers,)).

    sizes : np.ndarray
        The sizes of the centers (of size (n_centers,)).

    method_combination : str
        The method for combination, by default "stouffer".
        Can be in ["stouffer", "fisher"].

    ignore_nan_centers : bool
        If True, ignore NaN centers in meta-analysis combination. This helps reducing
        the number of genes with a NaN output. (default: True).

    Returns
    -------
    pd.Series
        The meta-analysis results for the gene
    """
    num_samples = np.sum(sizes)

    coefs = sizes / num_samples
    result_dict = {}
    result_combination = combine_pvalues(
        p_values,
        method=method_combination,
        weights=coefs,
        nan_policy="omit" if ignore_nan_centers else "propagate",
    )
    result_dict["pvalue"] = result_combination.pvalue
    result_dict["stat"] = result_combination.statistic
    if ignore_nan_centers:
        result_dict["lfc"] = (
            np.NaN
            if np.isnan(lfcs).all()
            else np.average(lfcs[~np.isnan(lfcs)], weights=coefs[~np.isnan(lfcs)])
        )
    else:
        result_dict["lfc"] = np.average(lfcs, weights=coefs)
    result_dict["lfcSE"] = pd.NA
    return pd.Series(result_dict)
