import json
import tempfile
from pathlib import Path
from typing import cast

import fedpydeseq2_datasets
import yaml
from fedpydeseq2.fedpydeseq2_pipeline import run_fedpydeseq2_experiment
from fedpydeseq2_datasets.constants import TCGADatasetNames
from fedpydeseq2_datasets.process_and_split_data import setup_tcga_dataset
from fedpydeseq2_datasets.utils import get_experiment_id

from fedpydeseq2_graphs.constants import PATHS_FILE

DESIGN_FACTORS = "stage"
DATASET_NAME = cast(TCGADatasetNames, "TCGA-LUAD")
PARAMETER_FILE = (
    Path(__file__).parent.parent.parent / "fedpydeseq2_parameters" / "default.yaml"
)

DEFAULT_ASSET_DIRECTORY = Path(fedpydeseq2_datasets.__file__).parent.resolve() / Path(
    "assets/tcga"
)


def setup_logging_config_file(logging_config_path: Path):
    """Set the logging configuration file.

    This function creates a logging configuration file at the provided path. The
    configuration file specifies that the workflow should be generated and cleaned
    after it is generated.
    """
    # Make sure the directory exists, and create it if not
    logging_config_path.parent.mkdir(parents=True, exist_ok=True)
    # Check that the workflow_file_path is in the PATHS_FILE and get it
    with open(PATHS_FILE) as f:
        paths = yaml.safe_load(f)
    assert "raw_workflow_file" in paths, (
        "The workflow_file_path is not in the PATHS_FILE"
    )
    workflow_file_path = Path(paths["raw_workflow_file"])
    # Get the workflow file path from
    logging_config = {
        "logger_configuration_ini_path": None,
        "generate_workflow": {
            "create_workflow": True,
            "workflow_file_path": str(workflow_file_path),
            "clean_workflow_file": True,
        },
        "log_shared_state_adata_content": False,
        "log_shared_state_size": False,
    }

    with logging_config_path.open("w") as f:
        json.dump(logging_config, f)


def get_processed_data_path() -> Path | None:
    """Get the processed data path from the PATHS_FILE.

    This function returns None if the processed_data_path is not in the PATHS_FILE.
    """
    with open(PATHS_FILE) as f:
        paths = yaml.safe_load(f)
    if "processed_data_path" not in paths:
        return None
    return Path(paths["processed_data_path"])


def get_raw_data_path() -> Path:
    """Get the raw data path from the PATHS_FILE."""
    with open(PATHS_FILE) as f:
        paths = yaml.safe_load(f)
    assert "raw_data_path" in paths, "The raw_data_path is not in the PATHS_FILE"
    return Path(paths["raw_data_path"])


def run_simple_fedpydeseq2_pipe(
    raw_data_path: Path,
    processed_data_path: Path,
    parameter_file: Path,
    logging_configuration_file_path: Path,
):
    """Run a simple FedPyDESeq2 pipeline.

    This function sets up a TCGA dataset with only two centers and the provided
    design factors. It then runs a FedPyDESeq2 pipeline with the provided
    parameters.

    raw_data_path : Path
        The path to the raw data.

    processed_data_path : Path
        The path to the processed data.

    parameter_file : Path
        The path to the parameter file.

    logging_configuration_file_path : Path
        The path to the logging configuration file.
    """
    setup_tcga_dataset(
        raw_data_path,
        processed_data_path,
        dataset_name=DATASET_NAME,
        small_samples=False,
        small_genes=False,
        only_two_centers=True,
        design_factors=DESIGN_FACTORS,
        continuous_factors=None,
        force=False,
    )
    experiment_id = get_experiment_id(
        DATASET_NAME,
        small_samples=False,
        small_genes=False,
        only_two_centers=True,
        design_factors=DESIGN_FACTORS,
        continuous_factors=None,
    )
    centers_root_directory = (
        processed_data_path / "centers_data" / "tcga" / experiment_id
    )

    run_fedpydeseq2_experiment(
        n_centers=2,
        backend="subprocess",
        design_factors=DESIGN_FACTORS,
        continuous_factors=None,
        simulate=True,
        centers_root_directory=centers_root_directory,
        logging_configuration_file_path=logging_configuration_file_path,
        parameter_file=parameter_file,
        register_data=True,
        asset_directory=DEFAULT_ASSET_DIRECTORY,
    )


def main():
    """Run the simple FedPyDESeq2 pipeline."""
    fixed_raw_data_path = get_raw_data_path()
    fixed_processed_data_path = get_processed_data_path()
    with tempfile.TemporaryDirectory() as tmp_dir:
        processed_data_path = (
            fixed_processed_data_path
            if fixed_processed_data_path is not None
            else Path(tmp_dir) / "processed_data"
        )
        logging_configuration_file_path = Path(tmp_dir) / "logging_configuration.json"
        setup_logging_config_file(logging_configuration_file_path)
        run_simple_fedpydeseq2_pipe(
            raw_data_path=fixed_raw_data_path,
            processed_data_path=processed_data_path,
            parameter_file=PARAMETER_FILE,
            logging_configuration_file_path=logging_configuration_file_path,
        )


if __name__ == "__main__":
    main()
