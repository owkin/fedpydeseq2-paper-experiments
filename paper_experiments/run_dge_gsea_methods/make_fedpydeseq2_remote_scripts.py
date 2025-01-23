import argparse
from pathlib import Path

from loguru import logger

from paper_experiments.utils.config_utils import load_config
from paper_experiments.utils.constants import EXPERIMENT_PATHS_FILE
from paper_experiments.utils.constants import REMOTE_SCRIPTS_DIR
from paper_experiments.utils.constants import SPECS_DIR


def create_bash_script(
    datasets: list[str],
    save_file_path: str,
    raw_data_path: str,
    credentials_path: str,
    bash_script_output_file: str,
    processed_data_path: str | None = None,
    design_factors: str | list[str] | None = None,
    continuous_factors: list[str] | None = None,
    ref_levels: dict[str, str] | None = None,
    contrast: list[str] | None = None,
    refit_cooks: bool | None = None,
    heterogeneity_method: str | None = None,
    heterogeneity_method_params: list[float | None] | None = None,
    fedpydeseq2_wheel_path: str | None = None,
):
    """Create a bash script for running experiments.

    Parameters
    ----------
    datasets : list[str]
        List of dataset names.
    save_file_path : str
        Path where the results will be saved.
    raw_data_path : str
        Path to the raw data.
    credentials_path : str
        Path to the credentials file.
    bash_script_output_file : str
        Path where the generated bash script will be saved.
    processed_data_path : str or None
        Path to the processed data.
        If None, the processed data will be created as
        a temporary directory.
    design_factors : Optional[Union[str, list[str]]]
        Design factors to use.
    continuous_factors : Optional[list[str]]
        Continuous factors to use.
    ref_levels : Optional[dict[str, str]]
        Reference levels to use.
    contrast : Optional[list[str]]
        Contrasts to use.
    refit_cooks : Optional[bool]
        Whether to refit cooks.
    heterogeneity_method : Optional[str]
        The heterogeneity method to use.
    heterogeneity_method_params : Optional[list[Optional[float]]]
        The heterogeneity method parameters to use.
    fedpydeseq2_wheel_path : Optional[str]
        Path to the FedPyDESeq2 wheel file.
    """
    additional_flags = ""
    if design_factors is not None:
        design_factors = (
            design_factors if isinstance(design_factors, list) else [design_factors]
        )
        design_factors_str = ", ".join(design_factors)
        additional_flags += f'    --design_factors "{design_factors_str}" \\\n'
    if continuous_factors is not None:
        continuous_factors_str = ", ".join(continuous_factors)
        additional_flags += f'    --continuous_factors "{continuous_factors_str}" \\\n'
    if ref_levels is not None:
        ref_levels_str = ",".join(f"{k}:{v}" for k, v in ref_levels.items())
        additional_flags += f'    --ref_levels "{ref_levels_str}" \\\n'
    if contrast is not None:
        contrast_str = ",".join(contrast)
        additional_flags += f'    --contrast "{contrast_str}" \\\n'
    if refit_cooks is not None:
        if refit_cooks:
            additional_flags += "    --refit_cooks \\\n"
        else:
            additional_flags += "    --no_refit_cooks \\\n"

    if fedpydeseq2_wheel_path is not None:
        additional_flags += (
            f'    --fedpydeseq2_wheel_path "{fedpydeseq2_wheel_path}" \\\n'
        )

    if processed_data_path is not None:
        additional_flags += f'    --processed_data_path "{processed_data_path}" \\\n'

    script_content_inside_for = (
        f"    fedpydeseq2-tcga-pipe \\\n"
        f"    --register_data \\\n"
        f'    --backend "remote" \\\n'
        f'    --dataset_name "${{dataset}}" \\\n'
        f"    --keep_original_centers \\\n"
        f'    --save_filepath "{save_file_path}"  \\\n'
        f'    --raw_data_path "{raw_data_path}" \\\n'
        f'    --credentials_path "{credentials_path}" \\\n'
    )
    # Add additional flags

    script_content_inside_for += additional_flags
    # Add heterogeneity flags
    if heterogeneity_method is not None:
        script_content_inside_for += (
            f'    --heterogeneity_method "{heterogeneity_method}" \\\n'
        )
        script_content_inside_for += (
            '    --heterogeneity_method_param "${heterogeneity_method_param}" \\\n'
        )
    # Add force preprocessing flag
    script_content_inside_for += "    --force_preprocessing\n"

    # If heterogeneity_method is None, do only a loop over the datasets
    datasets_str = " ".join(f"'{dataset}'" for dataset in datasets)

    if heterogeneity_method is None:
        script_content = (
            f"# You might need to run\n"
            f"# `chmod 777 {bash_script_output_file}`\n"
            f"# to give execution rights to the script\n"
            f"for dataset in {datasets_str}\n"
            f"do\n"
            f"{script_content_inside_for}"
            f"done\n"
        )
    # Do a loop over the datasets and the heterogeneity method parameters
    else:
        assert heterogeneity_method_params is not None
        heterogeneity_method_params_str = " ".join(
            f"'{str(param)}'" for param in heterogeneity_method_params
        )
        script_content = (
            f"# You might need to run\n"
            f"# `chmod 777 {bash_script_output_file}`\n"
            f"# to give execution rights to the script\n"
            f"for dataset in {datasets_str}\n"
            f"do\n"
            f"for heterogeneity_method_param in {heterogeneity_method_params_str}\n"
            f"do\n"
            f"{script_content_inside_for}"
            f"done\n"
            f"done\n"
        )

    with open(bash_script_output_file, "w") as file:
        file.write(script_content)


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
        "--fedpydeseq2_wheel_path",
        type=str,
        required=True,
        help="Path to the FedPyDESeq2 wheel file",
    )
    args = parser.parse_args()
    if args.config_file is None:
        config_file = SPECS_DIR / f"{args.experiment_name}_specs.yaml"
    else:
        config_file = Path(args.config_file)

    if args.paths_file is None:
        paths_file = EXPERIMENT_PATHS_FILE
    else:
        paths_file = Path(args.paths_file)

    logger.info("Loading the configuration file...")
    config = load_config(config_file)
    logger.success("Config successfully loaded !")

    logger.info("Loading the paths file...")

    paths = load_config(paths_file)["experiments"][args.experiment_name]
    raw_data_path = load_config(paths_file)["raw_data_path"]
    logger.success("Paths successfully loaded !")

    datasets = config["datasets"]

    pydeseq2_parameters = config["pydeseq2_parameters"]
    design_factors = pydeseq2_parameters["design_factors"]
    continuous_factors = pydeseq2_parameters["continuous_factors"]
    ref_levels = pydeseq2_parameters.get("ref_levels", None)
    contrast = pydeseq2_parameters["contrast"]
    refit_cooks = pydeseq2_parameters["refit_cooks"]

    heterogeneity_config = config["heterogeneity"]
    if heterogeneity_config is not None:
        heterogeneity_method = heterogeneity_config["heterogeneity_method"]
        heterogeneity_method_params = heterogeneity_config[
            "heterogeneity_method_params"
        ]
    else:
        heterogeneity_method = None
        heterogeneity_method_params = None

    fedpydeseq2_wheel_path = config.get("fedpydeseq2_wheel_path", None)

    save_file_path = paths["remote_results"]
    processed_data_path = paths.get("remote_processed", None)
    credentials_path = paths["remote_credentials"]

    bash_script_output_file = REMOTE_SCRIPTS_DIR / f"{args.experiment_name}_remote.sh"

    create_bash_script(
        datasets,
        save_file_path,
        raw_data_path,
        credentials_path,
        bash_script_output_file,
        processed_data_path=processed_data_path,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        ref_levels=ref_levels,
        contrast=contrast,
        refit_cooks=refit_cooks,
        heterogeneity_method=heterogeneity_method,
        heterogeneity_method_params=heterogeneity_method_params,
        fedpydeseq2_wheel_path=fedpydeseq2_wheel_path,
    )

    logger.success(f"Bash script successfully created at {bash_script_output_file}")
