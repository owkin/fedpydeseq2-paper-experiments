from pathlib import Path
from typing import Any

import yaml  # type: ignore

from paper_experiments.utils.constants import HETEROGENEITY_PARAMETERS
from paper_experiments.utils.constants import META_ANALYSIS_PARAMETERS
from paper_experiments.utils.constants import META_ANALYSIS_PARAMETERS_ABLATION

PLACEHOLDERS = {
    "META_ANALYSIS_PARAMETERS": META_ANALYSIS_PARAMETERS,
    "META_ANALYSIS_PARAMETERS_ABLATION": META_ANALYSIS_PARAMETERS_ABLATION,
    "HETEROGENEITY_PARAMETERS": HETEROGENEITY_PARAMETERS,
}


def load_config(config_file: str | Path) -> dict:
    """Load the configuration file.

    Parameters
    ----------
    config_file : Union[str,Path]
        The path to the configuration file.

    Returns
    -------
    dict
        The configuration.
    """
    with open(config_file) as f:
        config = yaml.safe_load(f)
    config = replace_placeholders(config, PLACEHOLDERS)
    return config


def replace_placeholders(config: dict, placeholders: dict[str, Any]) -> dict:
    """Replace the placeholders in the configuration file.

    Parameters
    ----------
    config : dict
        The configuration file.
    placeholders : dict[str,Any]
        The placeholders to replace.

    Returns
    -------
    dict
        The configuration with the placeholders replaced.
    """
    for key, value in config.items():
        if isinstance(value, dict):
            replace_placeholders(value, placeholders)
        elif isinstance(value, str):
            if value in placeholders:
                config[key] = placeholders[value]
    return config
