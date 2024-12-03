import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal

import fedpydeseq2_datasets
import pandas as pd
import yaml  # type: ignore
from fedpydeseq2.fedpydeseq2_pipeline import run_fedpydeseq2_experiment
from fedpydeseq2.substra_utils.utils import get_n_centers_from_datasamples_file
from fedpydeseq2_datasets.constants import TCGADatasetNames
from fedpydeseq2_datasets.process_and_split_data import setup_tcga_dataset
from fedpydeseq2_datasets.utils import get_experiment_id
from loguru import logger

DEFAULT_ASSET_DIRECTORY = Path(fedpydeseq2_datasets.__file__).parent.resolve() / Path(
    "assets/tcga"
)

DEFAULT_PARAMETER_FILE = (
    Path(__file__).parent.parent.parent / "fedpydeseq2_parameters/default.yaml"
)


def setup_tcga_data_folder(
    raw_data_path: Path,
    processed_data_path: Path,
    dataset_name: TCGADatasetNames,
    only_two_centers: bool = True,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    force: bool = False,
    heterogeneity_method: str | None = None,
    heterogeneity_method_param: float | None = None,
    **pydeseq2_kwargs: Any,
) -> tuple[int, Path]:
    """
    Set the TCGA data folder.

    Parameters
    ----------
    raw_data_path : Path
        The path to the raw data.

    processed_data_path : Path
        The path to the processed data. The subdirectories will
        be created if needed.

    dataset_name : TCGADatasetNames
        The dataset name.

    only_two_centers : bool
        Whether to use only two centers.

    design_factors : str or list[str]
        The design factors.

    continuous_factors : list[str] or None
        The continuous factors.

    force : bool
        Whether to force the preprocessing step.

    heterogeneity_method : str or None
        The heterogeneity method.

    heterogeneity_method_param : float or None
        The heterogeneity method parameter.

    **pydeseq2_kwargs : Any
        Additional parameters to pass to the DESeq2Strategy.
        For example the contrast, the lfc_null, the alt_hypothesis.

    Returns
    -------
    n_centers : int
        The number of centers.

    centers_root_directory : Path
        The path to the centers root directory.

    """
    setup_tcga_dataset(
        raw_data_path,
        processed_data_path,
        dataset_name=dataset_name,
        small_samples=False,
        small_genes=False,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        heterogeneity_method=heterogeneity_method,
        heterogeneity_method_param=heterogeneity_method_param,
        force=force,
        **pydeseq2_kwargs,
    )
    experiment_id = get_experiment_id(
        dataset_name,
        small_samples=False,
        small_genes=False,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        heterogeneity_method=heterogeneity_method,
        heterogeneity_method_param=heterogeneity_method_param,
        **pydeseq2_kwargs,
    )
    metadata = pd.read_csv(
        processed_data_path / "pooled_data" / "tcga" / experiment_id / "metadata.csv"
    )
    n_centers = len(metadata.center_id.unique())
    centers_root_directory = (
        processed_data_path / "centers_data" / "tcga" / experiment_id
    )
    return n_centers, centers_root_directory


def run_fedpydeseq2_tcga_pipe(
    raw_data_path: Path,
    processed_data_path: Path | None = None,
    dataset_name: TCGADatasetNames = "TCGA-LUAD",
    backend: str = "remote",
    simulate: bool = False,
    asset_directory: Path = DEFAULT_ASSET_DIRECTORY,
    register_data: bool = False,
    design_factors: str | list[str] = "stage",
    ref_levels: dict[str, str] | None = None,
    continuous_factors: list[str] | None = None,
    contrast: list[str] | None = None,
    lfc_null: float | None = None,
    alt_hypothesis: Literal["greaterAbs", "lessAbs", "greater", "less"] | None = None,
    refit_cooks: bool | None = None,
    save_filepath: str | Path | None = None,
    compute_plan_name: str | None = None,
    compute_plan_name_suffix: str | None = None,
    remote_timeout: int = 86400,  # 24 hours
    credentials_path: str | Path | None = None,
    dataset_datasamples_keys_path: str | Path | None = None,
    keep_original_centers: bool = False,
    force: bool = False,
    cp_id_path: str | Path | None = None,
    parameter_file: str | Path | None = DEFAULT_PARAMETER_FILE,
    heterogeneity_method: str | None = None,
    heterogeneity_method_param: float | None = None,
    fedpydeseq2_wheel_path: str | Path | None = None,
    **kwargs: Any,
):
    """Run the experiment with DESeq2 strategy using remote backend.

    Parameters
    ----------
    raw_data_path : str
        The path to the root data.

    processed_data_path : Path or None
        The path to the processed data. The subdirectories will
        be created if needed.

    dataset_name : TCGADatasetNames
        The dataset name.

    backend : str
        The backend to use for the experiment. Should be one
        of 'subprocess', 'docker' or 'remote'.

    simulate : bool
        Whether to simulate the experiment when using the subprocess backend.

    asset_directory : str
        The path to the asset directory.

    register_data : bool
        Whether to register the data.

    design_factors : str or list[str]
        The design factors.

    ref_levels : dict[str, str] or None
        The reference levels for the design factors.

    continuous_factors : list[str] or None
        The continuous factors.

    contrast : list[str] or None
        The contrast.

    lfc_null : float or None
        The null hypothesis for the log fold change.
        If None, will be set to 0.0, and will not be registered in the
        experiment id.

    alt_hypothesis : Literal["greaterAbs", "lessAbs", "greater", "less"] or None
        The alternative hypothesis. If not provided, will be set to None.

    refit_cooks : bool or None
        Whether to refit the cooks. By default will be set to None.
        In that case, it will be set to True, but will not be registered in the
        experiment id.

    save_filepath : Path or None
        The path to save the results.

    compute_plan_name : str or None
        The name of the compute plan.
        If None, the name will be set to
        "FedPyDESeq2_{dataset_name}_{current_datetime}".

    compute_plan_name_suffix : str or None
        The suffix to add to the compute plan name.

    remote_timeout : int
        The timeout for the remote backend.
        Default is 86400 seconds (24 hours).

    credentials_path : str or Path or None
        The path to the credentials file.
        By default, will be set to

        This credentials file allows to connect to the different organizations.

    dataset_datasamples_keys_path : str or Path or None
        The path to the dataset datasamples keys file.
        By default, will be set to
        Path(fedpydeseq2.__file__).parent /
          "credentials/{experiment_id}-datasamples-keys.yaml"
        This file contains the datasamples keys for the dataset.
        It is created and filled in if register_data is True
        and the datasamples are registered. Otherwise, the datasamples are
        retrieved from the file.

    keep_original_centers : bool
        Whether to keep the original centers or to use only two centers.

    force : bool
        Whether to force the preprocessing step.

    cp_id_path : str or Path or None
        Path to the file containing the compute plan id.

    parameter_file : str or Path or None
        If not None, yaml file containing the parameters to pass to the DESeq2Strategy.
        By default, it will be set to
        Path(fedpydeseq2.__file__).parent /
          "substra_utils/fedpydeseq2_parameters/default.yaml"

    heterogeneity_method : str or None
        The method to used to define the heterogeneity
        of the center's attribution.

    heterogeneity_method_param : float or None
        The parameter of the heterogeneity method.

    fedpydeseq2_wheel_path : str or Path or None
        Path to the FedPyDESeq2 wheel file.

    **kwargs: Any
        Additional parameters to pass the FedPyDESeq2 strategy.

    """
    only_two_centers = not keep_original_centers
    if ref_levels is None:
        ref_levels = {"stage": "Advanced"}

    pydeseq2_kwargs: dict[str, Any] = {}
    if refit_cooks is not None:
        pydeseq2_kwargs["refit_cooks"] = refit_cooks
    if lfc_null is not None:
        pydeseq2_kwargs["lfc_null"] = lfc_null
    if alt_hypothesis is not None:
        pydeseq2_kwargs["alt_hypothesis"] = alt_hypothesis
    if contrast is not None:
        pydeseq2_kwargs["contrast"] = contrast

    pydeseq2_kwargs.update(kwargs)

    logger.info(f"Running experiment for dataset {dataset_name}")
    logger.info(f"Design factors: {design_factors}")
    logger.info(f"Continuous factors: {continuous_factors}")
    logger.info(f"PyDESeq2 kwargs: {pydeseq2_kwargs}")

    experiment_id = get_experiment_id(
        dataset_name,
        small_samples=False,
        small_genes=False,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        heterogeneity_method=heterogeneity_method,
        heterogeneity_method_param=heterogeneity_method_param,
        **pydeseq2_kwargs,
    )

    if save_filepath is not None:
        save_folder = Path(save_filepath) / experiment_id
        save_folder.parent.mkdir(parents=True, exist_ok=True)
        logger.add(save_folder / "logs.txt")
    else:
        save_folder = None

    if compute_plan_name is None:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if len(experiment_id) > 50:
            logger.warning(
                f"The experiment id {experiment_id} is too long. "
                f"It will be truncated to {experiment_id[:50]}."
            )

        compute_plan_name = f"FedPyDESeq2_{experiment_id[:50]}_{current_datetime}"

    if compute_plan_name_suffix is not None and compute_plan_name_suffix != "":
        compute_plan_name = f"{compute_plan_name}_{compute_plan_name_suffix}"

    logger.info(f"Running experiment with compute plan name: {compute_plan_name}")

    if dataset_datasamples_keys_path is None:
        dataset_datasamples_keys_path = (
            Path(__file__).parent.parent.parent
            / f"credentials/{experiment_id}-datasamples-keys.yaml"
        )
    else:
        dataset_datasamples_keys_path = Path(dataset_datasamples_keys_path)

    with tempfile.TemporaryDirectory() as processed_data_path_temp:
        if processed_data_path is None:
            processed_data_path = Path(processed_data_path_temp)

        if register_data:
            n_centers, centers_root_directory = setup_tcga_data_folder(
                raw_data_path=raw_data_path,
                processed_data_path=processed_data_path,
                dataset_name=dataset_name,
                only_two_centers=only_two_centers,
                design_factors=design_factors,
                continuous_factors=continuous_factors,
                heterogeneity_method=heterogeneity_method,
                heterogeneity_method_param=heterogeneity_method_param,
                force=force,
                **pydeseq2_kwargs,
            )
        else:
            centers_root_directory = None
            assert dataset_datasamples_keys_path.exists(), (
                f"The dataset datasamples keys file {dataset_datasamples_keys_path} "
                f"does not exist."
                "Please provide a valid path to the datasamples keys file."
            )
            n_centers = get_n_centers_from_datasamples_file(
                dataset_datasamples_keys_path
            )

        if parameter_file is not None:
            with open(parameter_file, "rb") as file:
                run_fedpydeseq2_experiment_kwargs = yaml.load(
                    file, Loader=yaml.FullLoader
                )
                logger.info(f"Loaded parameters from {parameter_file}")
                logger.info(f"Parameters: {run_fedpydeseq2_experiment_kwargs}")
        else:
            logger.info("No parameter file provided.")
            run_fedpydeseq2_experiment_kwargs = {}

        specified_kwargs = {
            "n_centers": n_centers,
            "backend": backend,
            "simulate": simulate,
            "register_data": register_data,
            "asset_directory": asset_directory,
            "centers_root_directory": centers_root_directory,
            "compute_plan_name": compute_plan_name,
            "dataset_name": experiment_id,
            "remote_timeout": remote_timeout,
            "clean_models": True,
            "save_filepath": save_folder,
            "credentials_path": credentials_path,
            "cp_id_path": cp_id_path,
            "dataset_datasamples_keys_path": dataset_datasamples_keys_path,
            "design_factors": design_factors,
            "ref_levels": ref_levels,
            "continuous_factors": continuous_factors,
            "contrast": contrast,
            "lfc_null": lfc_null,
            "alt_hypothesis": alt_hypothesis,
            "refit_cooks": refit_cooks,
            "fedpydeseq2_wheel_path": fedpydeseq2_wheel_path,
        }

        for key, value in specified_kwargs.items():
            if key in run_fedpydeseq2_experiment_kwargs and value is None:
                logger.info(
                    f"Using the value {run_fedpydeseq2_experiment_kwargs[key]}"
                    f" for the key {key} "
                    "as it was not specified in the arguments of the function."
                )
            elif key in run_fedpydeseq2_experiment_kwargs and value is not None:
                logger.warning(
                    f"The key {key} was specified in the arguments of the function"
                    f"with the value {value}, but was also provided in the "
                    "parameter file"
                    f"with the value {run_fedpydeseq2_experiment_kwargs[key]}."
                    "The value provided in the arguments will be used."
                )

            else:
                logger.info(
                    f"Using the value {value} for the key {key} "
                    "as it was not provided in the parameter file."
                )
                run_fedpydeseq2_experiment_kwargs[key] = value

        if len(kwargs) > 0:
            logger.info(f"Adding the following parameters to the experiment: {kwargs}")
        run_fedpydeseq2_experiment_kwargs.update(kwargs)

        # Update lfc_null and refit_cooks if they are not provided
        if run_fedpydeseq2_experiment_kwargs["lfc_null"] is None:
            logger.warning(
                "The lfc_null parameter was not provided "
                "in the arguments or in the parameter file."
                "It will be set to 0.0."
            )
            run_fedpydeseq2_experiment_kwargs["lfc_null"] = 0.0
        if run_fedpydeseq2_experiment_kwargs["refit_cooks"] is None:
            logger.warning(
                "The refit_cooks parameter was not provided in the"
                " arguments or in the parameter file."
                "It will be set to True."
            )
            run_fedpydeseq2_experiment_kwargs["refit_cooks"] = True

        if save_folder is not None:
            with open(save_folder / "experiment_parameters.json", "w") as f:
                json.dump(
                    {k: str(v) for k, v in run_fedpydeseq2_experiment_kwargs.items()}, f
                )
        logger.info(
            f"Running FedPyDESeq2 experiment with the following parameters:"
            f"{run_fedpydeseq2_experiment_kwargs}"
        )

        run_fedpydeseq2_experiment(**run_fedpydeseq2_experiment_kwargs)  # type: ignore


def main():
    """Run the main function."""
    parser = argparse.ArgumentParser()
    dataset_name_help = (
        "The dataset name. Must be one of the following: 'TCGA-LUAD', "
        "'TCGA-PAAD', 'TCGA-BRCA', 'TCGA-COAD', 'TCGA-LUSC', 'TCGA-READ', "
        "'TCGA-SKCM', 'TCGA-PRAD', 'TCGA-NSCLC', 'TCGA-CRC."
    )
    parser.add_argument(
        "--dataset_name", type=str, default="TCGA-LUAD", help=dataset_name_help
    )
    raw_data_path_help = (
        "The path to the root data."
        "Should contain a 'tcga' subdirectory with 2 "
        "files: <dataset_name>_clinical.tsv.gz"
        " and <dataset_name>_raw_RNAseq.tsv.gz."
    )
    parser.add_argument(
        "--raw_data_path",
        type=str,
        required=True,
        help=raw_data_path_help,
    )
    processed_data_path_help = (
        "The path to the processed data. The subdirectories "
        "will be created if needed."
    )
    parser.add_argument(
        "--processed_data_path",
        type=str,
        required=True,
        help=processed_data_path_help,
    )
    register_data_help = "Whether to register the data or not."
    parser.add_argument(
        "--register_data",
        dest="register_data",
        action="store_true",
        help=register_data_help,
    )
    parser.set_defaults(register_data=False)
    design_factors_help = (
        "The design factors. Should be a comma-separated list, such as "
        "'gender,stage' or 'gender' or 'gender, stage'."
    )
    parser.add_argument(
        "--design_factors", type=str, default="stage", help=design_factors_help
    )
    ref_levels_help = (
        "The reference levels for the design factors."
        " Should be a comma-separated list, "
        "where each element is a key-value pair separated by a colon, such as "
        "'stage:Advanced,gender:male'."
    )
    parser.add_argument("--ref_levels", type=str, required=False, help=ref_levels_help)
    continuous_factors_help = (
        "The continuous factors. Should be a comma-separated list, "
        "such as 'CCF' or 'age, CCF'."
    )
    parser.add_argument(
        "--continuous_factors", type=str, required=False, help=continuous_factors_help
    )
    contrast_help = (
        "The contrast. Should be a comma-separated list with three elements, such as "
        "'stage,Advanced,Non-advanced' or 'CCF, , '."
    )
    parser.add_argument("--contrast", type=str, required=False, help=contrast_help)
    lfc_null_help = "The null hypothesis for the log fold change."
    parser.add_argument("--lfc_null", type=float, help=lfc_null_help)
    alt_hypothesis_help = (
        "The alternative hypothesis. Must be one of the following: "
        "'greaterAbs', 'lessAbs', 'greater', 'less'."
    )
    parser.add_argument(
        "--alt_hypothesis",
        type=str,
        required=False,
        help=alt_hypothesis_help,
    )
    save_filepath_help = (
        "The path to save the results. If not provided, the results will not be saved."
    )
    parser.add_argument(
        "--save_filepath", type=str, required=False, help=save_filepath_help
    )
    compute_plan_name_help = (
        "The name of the compute plan. If not provided, it will be set to "
        "'FedPyDESeq2_{dataset_name}_{current_datetime}'."
    )
    parser.add_argument(
        "--compute_plan_name", type=str, required=False, help=compute_plan_name_help
    )
    compute_plan_name_suffix_help = "The compute plan name suffix."
    parser.add_argument(
        "--compute_plan_name_suffix",
        type=str,
        required=False,
        help=compute_plan_name_suffix_help,
        default="",
    )
    remote_timeout_help = (
        "The timeout for the remote backend. Default is 86400 seconds (24 hours)."
    )
    parser.add_argument(
        "--remote_timeout", type=int, default=86400, help=remote_timeout_help
    )
    credentials_path_help = "The path to the credentials file."
    parser.add_argument(
        "--credentials_path", type=str, required=False, help=credentials_path_help
    )
    dataset_datasamples_keys_path_help = (
        "The path to the dataset datasamples keys file. By default, will be set to "
        "Path(__file__).parent / 'credentials/<dataset>_datasamples_keys.yaml'. where"
        "<dataset> is the dataset name."
    )
    parser.add_argument(
        "--dataset_datasamples_keys_path",
        type=str,
        required=False,
        help=dataset_datasamples_keys_path_help,
    )
    fedpydeseq2_parameters_help = (
        "The path to the yaml file containing the parameters to pass "
        "to the DESeq2Strategy."
        "By default, it will be set to Path(fedpydeseq2.__file__).parent "
        "/ substra_utils/"
        "fedpydeseq2_parameters/default.yaml., except if small_cp is True, "
        "in which case"
        "it will be set to Path(fedpydeseq2.__file__).parent / "
        "substra_utils /"
        " fedpydeseq2_parameters/debug.yaml."
    )
    parser.add_argument(
        "--fedpydeseq2_parameters",
        type=str,
        required=False,
        help=fedpydeseq2_parameters_help,
    )

    parser.add_argument(
        "--asset_directory",
        type=str,
        help="The path to the asset directory. Required if --register-data",
        default=DEFAULT_ASSET_DIRECTORY,
        required=False,
    )
    parser.add_argument(
        "--backend",
        type=str,
        help=(
            "The backend to use for the experiment. Should be one "
            "of 'subprocess', 'docker' or 'remote'."
        ),
        default="remote",
        required=False,
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "Whether to simulate the experiment when using the " "subprocess backend."
        ),
    )

    parser.add_argument(
        "--force_preprocessing",
        action="store_true",
        help="Whether to force the preprocessing step.",
    )

    parser.add_argument(
        "--keep_original_centers",
        action="store_true",
        help="Whether to keep the original centers or to use only two centers.",
    )
    parser.add_argument(
        "--cp_id_path",
        type=str,
        required=False,
        help="Path to the file containing the compute plan id.",
    )
    parser.add_argument(
        "--no_refit_cooks",
        action="store_true",
        dest="no_refit_cooks",
        help="Whether not to refit cooks.",
    )
    parser.add_argument(
        "--refit_cooks",
        action="store_true",
        dest="do_refit_cooks",
        help="Whether to refit cooks.",
    )
    parser.add_argument(
        "--heterogeneity_method",
        type=str,
        required=False,
        help=(
            "The method to used to define the "
            "heterogeneity of the center's attribution."
        ),
    )
    parser.add_argument(
        "--heterogeneity_method_param",
        type=str,
        required=False,
        help="The parameter of the heterogeneity method.",
    )
    parser.add_argument(
        "--fedpydeseq2_wheel_path",
        type=str,
        required=False,
        help="Path to the FedPyDESeq2 wheel file.",
    )
    args = parser.parse_args()
    DATASET_NAME = args.dataset_name
    RAW_DATA_PATH = Path(args.raw_data_path)
    PROCESSED_DATA_PATH = Path(args.processed_data_path)
    ASSET_DIRECTORY = args.asset_directory
    REGISTER_DATA = args.register_data
    DESIGN_FACTORS_STR = args.design_factors
    BACKEND = args.backend
    SIMULATE = args.simulate
    FORCE_PREPROCESSING = args.force_preprocessing

    KEEP_ORIGINAL_CENTERS = args.keep_original_centers
    # Convert design_factors to list
    # Remove the spaces and split by comma
    DESIGN_FACTORS = DESIGN_FACTORS_STR.replace(" ", "").split(",")
    if "ref_levels" in args and args.ref_levels is not None:
        REF_LEVELS_STR = args.ref_levels
        REF_LEVELS = {}
        for level in REF_LEVELS_STR.replace(" ", "").split(","):
            key, value = level.split(":")
            REF_LEVELS[key] = value

    else:
        REF_LEVELS = None  # type: ignore
    if "continuous_factors" in args and args.continuous_factors is not None:
        CONTINUOUS_FACTORS_STR = args.continuous_factors
        # Convert continuous_factors to list
        # Remove the spaces and split by comma
        CONTINUOUS_FACTORS = CONTINUOUS_FACTORS_STR.replace(" ", "").split(",")
    else:
        CONTINUOUS_FACTORS = None
    if "contrast" in args and args.contrast is not None:
        CONTRAST_STR = args.contrast
        # Convert contrast to list
        # Remove the spaces and split by comma
        CONTRAST = CONTRAST_STR.replace(" ", "").split(",")
    else:
        CONTRAST = None
    if "lfc_null" in args and args.lfc_null is not None:
        LFC_NULL = args.lfc_null
    else:
        LFC_NULL = None
    if "alt_hypothesis" in args:
        ALT_HYPOTHESIS = args.alt_hypothesis
    else:
        ALT_HYPOTHESIS = None
    if "save_filepath" in args:
        SAVE_FILEPATH = args.save_filepath
    else:
        SAVE_FILEPATH = None

    if "compute_plan_name" in args:
        COMPUTE_PLAN_NAME = args.compute_plan_name
    else:
        COMPUTE_PLAN_NAME = None

    if "compute_plan_name_suffix" in args:
        COMPUTE_PLAN_NAME_SUFFIX = args.compute_plan_name_suffix
    else:
        COMPUTE_PLAN_NAME_SUFFIX = ""

    REMOTE_TIMEOUT = args.remote_timeout

    if "credentials_path" in args:
        CREDENTIALS_PATH = args.credentials_path
    else:
        CREDENTIALS_PATH = None

    if "cp_id_path" in args:
        CP_ID_PATH: str | Path | None = args.cp_id_path
    else:
        CP_ID_PATH = None

    if "dataset_datasamples_keys_path" in args:
        DATASET_DATASAMPLES_KEYS_PATH = args.dataset_datasamples_keys_path
    else:
        DATASET_DATASAMPLES_KEYS_PATH = None

    if "fedpydeseq2_parameters" in args and args.fedpydeseq2_parameters is not None:
        PARAMETER_FILE = args.fedpydeseq2_parameters
    else:
        PARAMETER_FILE = DEFAULT_PARAMETER_FILE

    if not args.no_refit_cooks and not args.do_refit_cooks:
        REFIT_COOKS = None
    elif args.no_refit_cooks and args.do_refit_cooks:
        raise ValueError("You cannot set both --no_refit_cooks and --refit_cooks.")
    elif args.no_refit_cooks:
        REFIT_COOKS = False
    elif args.do_refit_cooks:
        REFIT_COOKS = True

    if "heterogeneity_method" in args:
        HETEROGENEITY_METHOD = args.heterogeneity_method
    else:
        HETEROGENEITY_METHOD = None

    if (
        "heterogeneity_method_param" in args
        and args.heterogeneity_method_param is not None
    ):
        HETEROGENEITY_METHOD_PARAM = float(args.heterogeneity_method_param)
    else:
        HETEROGENEITY_METHOD_PARAM = None

    if "fedpydeseq2_wheel_path" in args:
        FEDPYDESEQ2_WHEEL_PATH = args.fedpydeseq2_wheel_path
    else:
        FEDPYDESEQ2_WHEEL_PATH = None

    run_fedpydeseq2_tcga_pipe(
        raw_data_path=RAW_DATA_PATH,
        processed_data_path=PROCESSED_DATA_PATH,
        asset_directory=ASSET_DIRECTORY,
        backend=BACKEND,
        simulate=SIMULATE,
        dataset_name=DATASET_NAME,
        register_data=REGISTER_DATA,
        design_factors=DESIGN_FACTORS,
        ref_levels=REF_LEVELS,
        continuous_factors=CONTINUOUS_FACTORS,
        contrast=CONTRAST,
        lfc_null=LFC_NULL,
        alt_hypothesis=ALT_HYPOTHESIS,
        refit_cooks=REFIT_COOKS,
        save_filepath=SAVE_FILEPATH,
        compute_plan_name=COMPUTE_PLAN_NAME,
        compute_plan_name_suffix=COMPUTE_PLAN_NAME_SUFFIX,
        remote_timeout=REMOTE_TIMEOUT,
        credentials_path=CREDENTIALS_PATH,
        dataset_datasamples_keys_path=DATASET_DATASAMPLES_KEYS_PATH,
        cp_id_path=CP_ID_PATH,
        keep_original_centers=KEEP_ORIGINAL_CENTERS,
        force=FORCE_PREPROCESSING,
        parameter_file=PARAMETER_FILE,
        heterogeneity_method=HETEROGENEITY_METHOD,
        heterogeneity_method_param=HETEROGENEITY_METHOD_PARAM,
        fedpydeseq2_wheel_path=FEDPYDESEQ2_WHEEL_PATH,
    )


if __name__ == "__main__":
    main()
