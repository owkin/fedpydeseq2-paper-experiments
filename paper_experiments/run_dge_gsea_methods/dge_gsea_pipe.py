import argparse
from itertools import product
from pathlib import Path
from typing import Any
from typing import cast

from fedpydeseq2_datasets.constants import TCGADatasetNames
from loguru import logger

from paper_experiments.run_dge_gsea_methods.fedpydeseq2_tcga_pipe import (
    run_fedpydeseq2_tcga_pipe,
)
from paper_experiments.run_dge_gsea_methods.gsea_utils.gsea_utils import run_gsea_method
from paper_experiments.run_dge_gsea_methods.meta_analysis_tcga_pipe import (
    run_tcga_meta_analysis_experiments,
)
from paper_experiments.run_dge_gsea_methods.pydeseq2_per_center_tcga_pipe import (
    run_tcga_pydeseq2_per_center_experiments,
)
from paper_experiments.run_dge_gsea_methods.pydeseq2_pooled_tcga_pipe import (
    run_tcga_pooled_experiments,
)
from paper_experiments.utils.config_utils import load_config
from paper_experiments.utils.constants import EXPERIMENT_PATHS_FILE
from paper_experiments.utils.constants import SPECS_DIR


def run_dge_gsea_methods(
    config: dict[str, Any],
    paths: dict[str, Any],
    raw_data_path: str | Path,
    conda_activate_path: str | Path | None,
):
    """Run the DGE and GSEA methods.

    Parameters
    ----------
    config : dict[str, Any]
        The configuration dictionary.
        It contains the following fields:
        - pydeseq2_parameters: dict[str, Any]
            The parameters to pass to the pydeseq2 method.
            It must at least contain the fields:
            - design_factors: list[str]
                The design factors to use in the pydeseq2 method.
            - continuous_factors: list[str]
                The continuous factors to use in the pydeseq2 method.
        - datasets: list[str]
            The datasets to use.
        - keep_original_centers: bool
            Whether to keep the original centers or not.
        - heterogeneity: Optional[dict[str, Any]]
            The heterogeneity parameters.
            It must contain the fields:
            - heterogeneity_method: str
                The heterogeneity method to use.
            - heterogeneity_method_params: float
                The heterogeneity method parameters.
        - meta_analysis: list[Union[tuple[str, str, str], tu
        ple[str, str, str, float]]]
            The meta-analysis parameters.
            It must contain the fields:
            - meta_analysis_type: str
                The meta-analysis type to use.
            - method_random_effects: str
                The method to use for random effects.
            - method_combination: str
                The method to use for combination.
            - stats_clip_value: Optional[float]
                The stats clip value to use.
        - run_dge: Optional[list[str]]
            The DGE methods to run.
            If None, no DGE method will be run.
        - run_gsea: Optional[list[str]]
            The DGE methods on which to run GSEA.
            If None, no GSEA method will be run.

    paths : dict[str, Any]
        The paths dictionary. It contains the following fields:
        - results: str
            The path to the results directory.
        - remote_results: str
            The path to the remote results directory.

    raw_data_path : str or Path
        The path to the raw data.

    conda_activate_path : str or Path or None
        The path to the conda activate script.
    """
    raw_data_path = Path(raw_data_path)
    # Load parameters from config
    pydeseq2_parameters = config["pydeseq2_parameters"]
    # pop the continuous factors
    design_factors = pydeseq2_parameters.pop("design_factors")
    continuous_factors = pydeseq2_parameters.pop("continuous_factors")
    logger.info(f"Design factors: {design_factors}")
    logger.info(f"Continuous factors: {continuous_factors}")
    pydeseq2_kwargs = {
        key: value for key, value in pydeseq2_parameters.items() if value is not None
    }

    only_two_centers = not (config["keep_original_centers"])
    datasets = config["datasets"]
    heterogeneity = config.get("heterogeneity", None)
    if heterogeneity is not None:
        heterogeneity_method = heterogeneity.get("heterogeneity_method", "binomial")
        heterogeneity_method_params = (
            heterogeneity["heterogeneity_method_params"]
            if heterogeneity["heterogeneity_method_params"] is not None
            else [None]
        )

    else:
        heterogeneity_method = None
        heterogeneity_method_params = [None]

    meta_analysis_parameters = config["meta_analysis"]

    dge_methods_to_run = [] if config["run_dge"] is None else config["run_dge"]
    dge_methods_for_gsea = config["run_gsea"]

    logger.info(f"Running DGE for {dge_methods_to_run} methods...")

    assert all(
        dge_method_to_run
        in [
            "pydeseq2",
            "fedpydeseq2_simulated",
            "meta_analysis",
            "pydeseq2_largest",
        ]
        for dge_method_to_run in dge_methods_to_run
    ), (
        "Local DGE methods must be either 'pydeseq2'"
        ", 'fedpydeseq2_simulated', 'pydeseq2_largest'"
        " or 'meta_analysis'."
    )
    experiment_results_path = Path(paths["results"])

    dge_results_path = experiment_results_path / "dge_results"

    gsea_results_path = experiment_results_path / "gsea_results"

    dataset_names = [cast(TCGADatasetNames, dataset) for dataset in datasets]
    if "pydeseq2" in dge_methods_to_run:
        logger.info("Running pydeseq2 experiments...")
        for dataset in dataset_names:
            run_tcga_pooled_experiments(
                dataset_name=dataset,
                raw_data_path=raw_data_path,
                results_path=dge_results_path / "pydeseq2",
                small_samples=False,
                small_genes=False,
                only_two_centers=only_two_centers,
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                heterogeneity_method=None,
                heterogeneity_method_param=None,
                **pydeseq2_kwargs,
            )
        logger.success("Finished running pydeseq2 experiments !")

    if ("meta_analysis" in dge_methods_to_run) or (
        "pydeseq2_largest" in dge_methods_to_run
    ):
        logger.info("Running pydeseq2 per center experiments...")
        for dataset, heterogeneity_method_param in product(
            dataset_names, heterogeneity_method_params
        ):
            run_tcga_pydeseq2_per_center_experiments(
                dataset_name=dataset,
                raw_data_path=raw_data_path,
                results_path=dge_results_path / "pydeseq2_per_center",
                small_samples=False,
                small_genes=False,
                only_two_centers=only_two_centers,
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                heterogeneity_method=heterogeneity_method,
                heterogeneity_method_param=heterogeneity_method_param,
                **pydeseq2_kwargs,
            )
        logger.success("Finished running pydeseq2 per center experiments !")

    if "fedpydeseq2_simulated" in dge_methods_to_run:
        logger.info("Running fedpydeseq2 simulate experiments...")
        for dataset, heterogeneity_method_param in product(
            dataset_names, heterogeneity_method_params
        ):
            run_fedpydeseq2_tcga_pipe(
                raw_data_path=raw_data_path,
                dataset_name=dataset,
                backend="subprocess",
                simulate=True,
                register_data=True,
                force=True,
                save_filepath=dge_results_path / "fedpydeseq2_simulated",
                keep_original_centers=not (only_two_centers),
                heterogeneity_method=heterogeneity_method,
                heterogeneity_method_param=heterogeneity_method_param,
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                **pydeseq2_kwargs,
            )
        logger.success("Finished running fedpydeseq2 simulate experiments !")

    if "meta_analysis" in dge_methods_to_run:
        logger.info("Running meta-analysis experiments...")
        for dataset, meta_analysis_parameter, heterogeneity_method_param in list(
            product(datasets, meta_analysis_parameters, heterogeneity_method_params)
        ):
            if len(meta_analysis_parameter) == 3:
                (
                    meta_analysis_type,
                    method_random_effects,
                    method_combination,
                ) = meta_analysis_parameter
                stats_clip_value = None
            else:
                (
                    meta_analysis_type,
                    method_random_effects,
                    method_combination,
                    stats_clip_value,
                ) = meta_analysis_parameter
            logger.info(
                f"Running meta-analysis experiments for dataset {dataset} "
                f"with meta_analysis_type {meta_analysis_type}, "
                f"method_random_effects {method_random_effects}, "
                f"method_combination {method_combination}"
                f"stats_clip_value {stats_clip_value}"
            )
            run_tcga_meta_analysis_experiments(
                dataset_name=dataset,
                stats_per_center_path=dge_results_path / "pydeseq2_per_center",
                stats_meta_path=dge_results_path / "meta_analysis",
                meta_analysis_type=meta_analysis_type,
                method_random_effects=method_random_effects,
                method_combination=method_combination,
                stats_clip_value=stats_clip_value,
                only_two_centers=only_two_centers,
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                heterogeneity_method=heterogeneity_method,
                heterogeneity_method_param=heterogeneity_method_param,
                **pydeseq2_kwargs,
            )
            logger.success(
                f"Finished running meta-analysis experiments for dataset {dataset} "
                f"with meta_analysis_type {meta_analysis_type}, "
                f"method_random_effects {method_random_effects}, "
                f"method_combination {method_combination}"
                f"stats_clip_value {stats_clip_value}"
            )
        logger.success("Finished running meta-analysis experiments !")

    if dge_methods_for_gsea is not None:
        logger.info("Running GSEA experiments...")
        dge_results_paths = {}
        if "fedpydeseq2_simulated" in dge_methods_for_gsea:
            dge_results_paths["fedpydeseq2_simulated"] = (
                dge_results_path / "fedpydeseq2_simulated"
            )
        if "pydeseq2" in dge_methods_for_gsea:
            dge_results_paths["pydeseq2"] = dge_results_path / "pydeseq2"
        if "meta_analysis" in dge_methods_for_gsea:
            dge_results_paths["meta_analysis"] = dge_results_path / "meta_analysis"
        if "pydeseq2_largest" in dge_methods_for_gsea:
            dge_results_paths["pydeseq2_largest"] = (
                dge_results_path / "pydeseq2_per_center"
            )
        if "fedpydeseq2_remote" in dge_methods_for_gsea:
            assert "remote_results" in paths, (
                "The remote results path must be provided "
                "when running the fedpydeseq2_remote method."
            )
            dge_results_paths["fedpydeseq2_remote"] = paths["remote_results"]

        gsea_results_paths = {
            dge_method: gsea_results_path / dge_method
            for dge_method in dge_methods_for_gsea
        }

        run_gsea_method(
            dataset_names=dataset_names,
            dge_results_paths=dge_results_paths,
            gsea_results_paths=gsea_results_paths,
            small_samples=False,
            small_genes=False,
            only_two_centers=only_two_centers,
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            heterogeneity_method=heterogeneity_method,
            heterogeneity_method_params=heterogeneity_method_params,
            meta_analysis_parameters=meta_analysis_parameters,
            conda_activate_path=conda_activate_path,
            **pydeseq2_kwargs,
        )
        logger.success("Finished running GSEA experiments !")


def main():
    """Run the main function."""
    parser = argparse.ArgumentParser(
        """Run an inference on a dataset and store results in an experiment path."""
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=False,
        help="Path to the inference configuration file",
    )
    parser.add_argument(
        "--paths_file", type=str, required=False, help="Path to the paths file"
    )
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Name of the experiment"
    )
    parser.add_argument(
        "--conda_activate_path",
        type=str,
        required=False,
        help="Path to the conda activate script",
    )
    args = parser.parse_args()
    if args.config_file is None:
        config_file = SPECS_DIR / f"{args.experiment_name}_specs.yaml"
    else:
        config_file = Path(args.config_file)

    if args.paths_file is None:
        paths_file = EXPERIMENT_PATHS_FILE

    logger.info("Loading the configuration file...")
    config = load_config(config_file)
    logger.success("Config successfully loaded !")

    logger.info("Loading the paths file...")

    paths = load_config(paths_file)["experiments"][args.experiment_name]
    raw_data_path = load_config(paths_file)["raw_data_path"]
    logger.success("Paths successfully loaded !")

    logger.info("Running the experiment...")
    run_dge_gsea_methods(
        config=config,
        paths=paths,
        raw_data_path=raw_data_path,
        conda_activate_path=args.conda_activate_path,
    )
    logger.info("END ...")


if __name__ == "__main__":
    main()
