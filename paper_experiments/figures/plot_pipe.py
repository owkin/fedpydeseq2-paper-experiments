import argparse
from itertools import product
from pathlib import Path
from typing import cast

from fedpydeseq2_datasets.constants import TCGADatasetNames
from loguru import logger

from paper_experiments.figures.generate_cross_tables_utils import (
    build_dataset_comparison_cross_table,
)
from paper_experiments.figures.generate_cross_tables_utils import (
    build_pan_cancer_confusion_matrix,
)
from paper_experiments.figures.generate_cross_tables_utils import (
    build_test_vs_ref_cross_tables,
)
from paper_experiments.figures.generate_gsea_plots_utils import (
    make_gsea_plot_method_pairs,
)
from paper_experiments.figures.generate_heterogeneity_plots_utils import (
    build_heterogeneity_grid_plot,
)
from paper_experiments.figures.generate_heterogeneity_plots_utils import (
    build_test_vs_ref_heterogeneity_plot,
)
from paper_experiments.figures.generate_lfc_lfc_plots import (
    build_lfc_lfc_and_padj_padj_plot,
)
from paper_experiments.figures.generate_lfc_padj_violin_plots import (
    build_lfc_or_padj_rel_error_violin_plot,
)
from paper_experiments.figures.generate_volcano_plots import build_volcano_plot
from paper_experiments.figures.utils import SCORING_FUNCTIONS_YLABELS
from paper_experiments.utils.config_utils import load_config
from paper_experiments.utils.constants import EXPERIMENT_PATHS_FILE
from paper_experiments.utils.constants import SPECS_DIR


def run_plot_pipe(
    config: dict,
    paths: dict,
    raw_data_path: str | Path,
) -> None:
    """Run the plot pipe.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
        It must contain the following keys:
        - pydeseq2_parameters: The parameters for the pydeseq2 method.
        - datasets: The datasets to run the inference on.
        - heterogeneity: The heterogeneity parameters.
        - meta_analysis: The meta analysis parameters.
        - plots: The plots to generate.
        - run_dge: The differential gene expression methods to run.
        - run_gsea: The differential gene expression methods to run for gsea.

    paths : dict
        The paths dictionary.
        It should contain the following keys:
        - results: The path to the results directory.
        - remote_results: The path to the remote results directory.
        Necessary for the fedpydeseq2_remote method.


    raw_data_path : str | Path
        The path to the raw data.
    """
    raw_data_path = Path(raw_data_path)
    # Load parameters from config
    pydeseq2_parameters = config["pydeseq2_parameters"]
    # pop the continuous factors
    design_factors = pydeseq2_parameters.pop("design_factors")
    continuous_factors = pydeseq2_parameters.pop("continuous_factors")
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

    experiment_results_path = Path(paths["results"])
    dge_results_path = experiment_results_path / "dge_results"

    gsea_results_path = experiment_results_path / "gsea_results"

    plot_results_path = experiment_results_path / "figures"

    dataset_names = [cast(TCGADatasetNames, dataset) for dataset in datasets]

    plots_config = config["plots"]

    if "datasets_cross_table_config" in plots_config:
        datasets_cross_table_config = plots_config["datasets_cross_table_config"]
        datasets_cross_table_path = plot_results_path / "datasets_cross_tables"
        datasets_cross_table_path.mkdir(parents=True, exist_ok=True)
        log2fc_threshold = datasets_cross_table_config.get("log2fc_threshold", 2.0)
        padj_threshold = datasets_cross_table_config.get("padj_threshold", 0.05)
        dataset_pairs_in_config = datasets_cross_table_config["dataset_pairs"]
        ref_with_heterogeneity = datasets_cross_table_config.get(
            "ref_with_heterogeneity", False
        )
        # Check that the methods are in the list of methods to run
        method_results_paths = get_dge_methods_paths(
            all_methods=["pydeseq2"],
            dge_results_path=dge_results_path,
            paths=paths,
            dge_methods_to_run=dge_methods_to_run,
        )
        for dataset1_name, dataset2_name in dataset_pairs_in_config:
            build_dataset_comparison_cross_table(
                method="pydeseq2",
                method_results_path=method_results_paths["pydeseq2"],
                dataset1_name=dataset1_name,
                dataset2_name=dataset2_name,
                save_file_path=datasets_cross_table_path
                / f"cross_table_{dataset1_name}_vs_{dataset2_name}.pdf",
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                reference_dds_ref_level=("stage", "Advanced"),
                log2fc_threshold=log2fc_threshold,
                padj_threshold=padj_threshold,
                **pydeseq2_kwargs,
            )

    if "pancancer_cross_table" in plots_config:
        pancancer_table_config = plots_config["pancancer_cross_table"]
        pancancer_table_path = plot_results_path / "pancancer_tables"
        pancancer_table_path.mkdir(parents=True, exist_ok=True)
        log2fc_threshold = pancancer_table_config.get("log2fc_threshold", 2.0)
        padj_threshold = pancancer_table_config.get("padj_threshold", 0.05)
        method_pairs_in_config = pancancer_table_config["method_pairs"]
        ref_with_heterogeneity = pancancer_table_config.get(
            "ref_with_heterogeneity", False
        )
        # Method pairs is a list of list of string, convert to list of tuples
        method_pairs = [
            (method_pair[0], method_pair[1]) for method_pair in method_pairs_in_config
        ]
        all_methods = list(
            {method for method_pair in method_pairs for method in method_pair}
        )
        # Check that the methods are in the list of methods to run
        method_results_paths = get_dge_methods_paths(
            all_methods=all_methods,
            dge_results_path=dge_results_path,
            paths=paths,
            dge_methods_to_run=dge_methods_to_run,
        )

        for method_test, method_ref in method_pairs:
            build_pan_cancer_confusion_matrix(
                method_test=method_test,
                method_ref=method_ref,
                method_test_results_path=method_results_paths[method_test],
                method_ref_results_path=method_results_paths[method_ref],
                save_file_path=pancancer_table_path,
                dataset_names=dataset_names,
                small_samples=False,
                small_genes=False,
                only_two_centers=only_two_centers,
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                reference_dds_ref_level=("stage", "Advanced"),
                meta_analysis_parameters=meta_analysis_parameters,
                ref_with_heterogeneity=ref_with_heterogeneity,
                log2fc_threshold=log2fc_threshold,
                padj_threshold=padj_threshold,
                **pydeseq2_kwargs,
            )

    if "heterogeneity_grid" in plots_config:
        heterogeneity_plots_config = plots_config["heterogeneity_grid"]
        heterogeneity_plots_path = plot_results_path / "heterogeneity_grid"
        heterogeneity_plots_path.mkdir(parents=True, exist_ok=True)

        methods_test = heterogeneity_plots_config["methods_test"]
        method_ref = heterogeneity_plots_config["method_ref"]
        methods_test_results_paths = get_dge_methods_paths(
            all_methods=methods_test,
            dge_results_path=dge_results_path,
            paths=paths,
            dge_methods_to_run=dge_methods_to_run,
        )

        method_ref_results_paths = get_dge_methods_paths(
            all_methods=[method_ref],
            dge_results_path=dge_results_path,
            paths=paths,
            dge_methods_to_run=dge_methods_to_run,
        )[method_ref]

        scoring_function_names = heterogeneity_plots_config["scoring_function_names"]

        build_heterogeneity_grid_plot(
            methods_test=methods_test,
            method_ref=method_ref,
            methods_test_results_paths=methods_test_results_paths,
            method_ref_results_path=method_ref_results_paths,
            save_file_path=heterogeneity_plots_path,
            dataset_names=dataset_names,
            heterogeneity_method_params=heterogeneity_method_params,
            scoring_function_names=scoring_function_names,
            design_factors=design_factors,
            only_two_centers=only_two_centers,
            continuous_factors=continuous_factors,
            meta_analysis_parameters=meta_analysis_parameters,
            **pydeseq2_kwargs,
        )

    if "heterogeneity" in plots_config:
        heterogeneity_plots_config = plots_config["heterogeneity"]
        heterogeneity_plots_path = plot_results_path / "heterogeneity"
        heterogeneity_plots_path.mkdir(parents=True, exist_ok=True)

        method_test = heterogeneity_plots_config["method_test"]
        method_ref = heterogeneity_plots_config["method_ref"]
        all_methods = list(
            set(method_test + [method_ref])
        )  # Check that the methods are in the list of methods to run
        method_results_paths = get_dge_methods_paths(
            all_methods=all_methods,
            dge_results_path=dge_results_path,
            paths=paths,
            dge_methods_to_run=dge_methods_to_run,
        )
        scoring_function_names = heterogeneity_plots_config["scoring_function_names"]

        for dataset, scoring_function_name in product(
            dataset_names, scoring_function_names
        ):
            build_test_vs_ref_heterogeneity_plot(
                methods_test=method_test,
                method_ref=method_ref,
                methods_results_path=method_results_paths,
                heterogeneity_plot_save_path=heterogeneity_plots_path,
                plot_title=(
                    f"{dataset} - {SCORING_FUNCTIONS_YLABELS[scoring_function_name]}"
                ),
                dataset_name=dataset,
                heterogeneity_method=heterogeneity_method,
                heterogeneity_method_params=heterogeneity_method_params,
                scoring_function_name=scoring_function_name,
                design_factors=design_factors,
                only_two_centers=only_two_centers,
                continuous_factors=continuous_factors,
                meta_analysis_parameters=meta_analysis_parameters,
                **pydeseq2_kwargs,
            )

    if "gsea_plots" in plots_config:
        gsea_plots_config = plots_config["gsea_plots"]
        gsea_plots_path = plot_results_path / "gsea_plots"
        gsea_plots_path.mkdir(parents=True, exist_ok=True)
        method_pairs_in_config = gsea_plots_config["method_pairs"]
        # Method pairs is a list of list of string, convert to list of tuples
        method_pairs = [
            (method_pair[0], method_pair[1]) for method_pair in method_pairs_in_config
        ]
        all_methods = list(
            {method for method_pair in method_pairs for method in method_pair}
        )
        # Check that the methods are in the list of methods to run
        for method in all_methods:
            if method not in dge_methods_for_gsea:
                logger.warning(
                    f"Method {method} is not in the list of methods to run for gsea"
                )

        gsea_results_paths = {
            method: gsea_results_path / method for method in all_methods
        }

        plot_parameters = gsea_plots_config["plot_parameters"]
        for plot_kwargs, heterogeneity_method_param in product(
            plot_parameters, heterogeneity_method_params
        ):
            make_gsea_plot_method_pairs(
                method_pairs=method_pairs,
                method_gsea_paths=gsea_results_paths,
                dataset_names=dataset_names,
                save_path=gsea_plots_path,
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                meta_analysis_parameters=meta_analysis_parameters,
                heterogeneity_method=heterogeneity_method,
                heterogeneity_method_param=heterogeneity_method_param,
                **plot_kwargs,
                **pydeseq2_kwargs,
            )

    if "violin_plots" in plots_config:
        violin_plots_config = plots_config["violin_plots"]
        violin_plots_path = plot_results_path / "violin_plots"
        violin_plots_path.mkdir(parents=True, exist_ok=True)
        methods_in_config = violin_plots_config["methods"]

        # Check that the methods are in the list of methods to run
        method_results_paths = get_dge_methods_paths(
            all_methods=methods_in_config,
            dge_results_path=dge_results_path,
            paths=paths,
            dge_methods_to_run=dge_methods_to_run,
        )

        for dataset in dataset_names:
            build_lfc_or_padj_rel_error_violin_plot(
                methods=methods_in_config,
                method_results_paths=method_results_paths,
                save_path=violin_plots_path,
                dataset_name=dataset,
                small_samples=False,
                small_genes=False,
                only_two_centers=only_two_centers,
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                meta_analysis_parameters=meta_analysis_parameters,
                heterogeneity_method=heterogeneity_method,
                heterogeneity_method_params=heterogeneity_method_params,
                ref_with_heterogeneity=ref_with_heterogeneity,
                log2fc_threshold=log2fc_threshold,
                padj_threshold=padj_threshold,
                **pydeseq2_kwargs,
            )

    if "lfc_lfc_plots" in plots_config:
        lfc_lfc_plots_config = plots_config["lfc_lfc_plots"]
        lfc_lfc_plots_path = plot_results_path / "lfc_lfc_plots"
        lfc_lfc_plots_path.mkdir(parents=True, exist_ok=True)
        log2fc_threshold = lfc_lfc_plots_config.get("log2fc_threshold", 2.0)
        padj_threshold = lfc_lfc_plots_config.get("padj_threshold", 0.05)
        methods_in_config = lfc_lfc_plots_config["methods"]

        # Check that the methods are in the list of methods to run
        method_results_paths = get_dge_methods_paths(
            all_methods=methods_in_config,
            dge_results_path=dge_results_path,
            paths=paths,
            dge_methods_to_run=dge_methods_to_run,
        )
        for dataset, method in product(dataset_names, methods_in_config):
            if method != "pydeseq2":  # cant compare pydeseq2 with itself
                build_lfc_lfc_and_padj_padj_plot(
                    method_name=method,
                    method_results_path=method_results_paths[method],
                    pydeseq2_results_path=method_results_paths["pydeseq2"],
                    save_path=lfc_lfc_plots_path,
                    dataset_name=dataset,
                    small_samples=False,
                    small_genes=False,
                    only_two_centers=only_two_centers,
                    design_factors=design_factors,
                    continuous_factors=continuous_factors,
                    reference_dds_ref_level=("stage", "Advanced"),
                    meta_analysis_parameters=meta_analysis_parameters,
                    log2fc_threshold=log2fc_threshold,
                    padj_threshold=padj_threshold,
                    **pydeseq2_kwargs,
                )

    if "volcano_plots" in plots_config:
        volcano_plots_config = plots_config["volcano_plots"]
        volcano_plots_path = plot_results_path / "volcano_plots"
        volcano_plots_path.mkdir(parents=True, exist_ok=True)
        log2fc_threshold = volcano_plots_config.get("log2fc_threshold", 2.0)
        padj_threshold = volcano_plots_config.get("padj_threshold", 0.05)
        methods_in_config = volcano_plots_config["methods"]

        # Check that the methods are in the list of methods to run
        method_results_paths = get_dge_methods_paths(
            all_methods=methods_in_config,
            dge_results_path=dge_results_path,
            paths=paths,
            dge_methods_to_run=dge_methods_to_run,
        )
        for dataset, method in product(dataset_names, methods_in_config):
            build_volcano_plot(
                method_name=method,
                method_results_path=method_results_paths[method],
                pydeseq2_results_path=method_results_paths["pydeseq2"],
                volcano_plot_save_path=volcano_plots_path,
                dataset_name=dataset,
                small_samples=False,
                small_genes=False,
                only_two_centers=only_two_centers,
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                reference_dds_ref_level=("stage", "Advanced"),
                meta_analysis_parameters=meta_analysis_parameters,
                log2fc_threshold=log2fc_threshold,
                padj_threshold=padj_threshold,
                **pydeseq2_kwargs,
            )

    if "cross_table" in plots_config:
        cross_table_config = plots_config["cross_table"]
        cross_table_path = plot_results_path / "cross_tables"
        cross_table_path.mkdir(parents=True, exist_ok=True)
        log2fc_threshold = cross_table_config.get("log2fc_threshold", 2.0)
        padj_threshold = cross_table_config.get("padj_threshold", 0.05)
        method_pairs_in_config = cross_table_config["method_pairs"]
        ref_with_heterogeneity = cross_table_config.get("ref_with_heterogeneity", False)
        # Method pairs is a list of list of string, convert to list of tuples
        method_pairs = [
            (method_pair[0], method_pair[1]) for method_pair in method_pairs_in_config
        ]
        all_methods = list(
            {method for method_pair in method_pairs for method in method_pair}
        )
        # Check that the methods are in the list of methods to run
        method_results_paths = get_dge_methods_paths(
            all_methods=all_methods,
            dge_results_path=dge_results_path,
            paths=paths,
            dge_methods_to_run=dge_methods_to_run,
        )

        for heterogeneity_method_param in heterogeneity_method_params:
            build_test_vs_ref_cross_tables(
                method_pairs=method_pairs,
                method_results_paths=method_results_paths,
                cross_table_save_path=cross_table_path,
                dataset_names=dataset_names,
                small_samples=False,
                small_genes=False,
                only_two_centers=only_two_centers,
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                meta_analysis_parameters=meta_analysis_parameters,
                heterogeneity_method=heterogeneity_method,
                heterogeneity_method_param=heterogeneity_method_param,
                ref_with_heterogeneity=ref_with_heterogeneity,
                log2fc_threshold=log2fc_threshold,
                padj_threshold=padj_threshold,
                **pydeseq2_kwargs,
            )


def get_dge_methods_paths(
    all_methods: list[str],
    dge_results_path: Path,
    paths: dict,
    dge_methods_to_run: list[str],
) -> dict:
    """Get the paths to the results of the differential gene expression methods.

    Parameters
    ----------
    all_methods : list[str]
        The list of all differential gene expression methods to run.

    dge_results_path : Path
        The path to the differential gene expression results.

    paths : dict
        The paths dictionary, containing the remote results path
        if needed.

    dge_methods_to_run : list[str]
        The list of differential gene expression methods to run.

    Returns
    -------
    dict
        A dictionary containing the paths to the results of the
        differential gene expression methods.

    """
    # Check that the methods are in the list of methods to run
    for method in all_methods:
        if method not in dge_methods_to_run and method != "fedpydeseq2_remote":
            logger.warning(f"Method {method} is not in the list of methods to run")

    method_results_paths = {
        method: dge_results_path / method
        for method in all_methods
        if method != "fedpydeseq2_remote" and method != "pydeseq2_largest"
    }

    if "fedpydeseq2_remote" in all_methods:
        method_results_paths["fedpydeseq2_remote"] = paths["remote_results"]

    if "pydeseq2_largest" in all_methods:
        method_results_paths["pydeseq2_largest"] = (
            dge_results_path / "pydeseq2_per_center"
        )
    return method_results_paths


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

    logger.info("Creating plots...")
    run_plot_pipe(
        config=config,
        paths=paths,
        raw_data_path=raw_data_path,
    )
    logger.info("END ...")


if __name__ == "__main__":
    main()
