from itertools import product
from pathlib import Path
from typing import Literal

SPECS_DIR = Path(__file__).parent.parent.parent.resolve() / "experiment_specifications"

EXPERIMENT_PATHS_FILE = (
    Path(__file__).parent.parent.parent.resolve() / "experiment_paths.yaml"
)

REMOTE_SCRIPTS_DIR = (
    Path(__file__).parent.parent.parent.resolve() / "fedpydeseq2_remote_scripts"
)

ExperimentNames = Literal[
    "heterogeneity",
    "single_factor",
    "multi_factor",
    "continuous_factor",
]

DGE_MODES = [
    "pydeseq2_per_center",
    "pydeseq2",
    "fedpydeseq2_simulated",
    "meta_analysis",
    "fedpydeseq2_remote",
    "pydeseq2_largest",
]

PLOT_TYPES = [
    "cross_table",
]

HETEROGENEITY_PARAMETERS = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
]


MetaAnalysisParameter = (
    tuple[str, str | None, str | None]
    | tuple[str, str | None, str | None, float | None]
)


META_ANALYSIS_PARAMETERS: list[MetaAnalysisParameter] = [
    ("random_effect", "dl", None),
    ("random_effect", "iterated", None),
    ("pvalue_combination", None, "stouffer"),
    ("pvalue_combination", None, "fisher"),
    ("fixed_effect", None, None),
]

META_ANALYSIS_PARAMETERS_ABLATION: list[MetaAnalysisParameter] = [
    (meta_analysis_type, method_random_effects, method_combination, stats_clip_value)
    for (
        meta_analysis_type,
        method_random_effects,
        method_combination,
    ), stats_clip_value in product(
        [
            ("random_effect", "dl", None),
            ("random_effect", "iterated", None),
            ("random_effect", "chi2", None),
            ("pvalue_combination", None, "stouffer"),
            ("pvalue_combination", None, "fisher"),
            ("fixed_effect", None, None),
        ],
        [1e6, 1e8, 1e10],
    )
]

HETEROGENEITY_METHOD_PARAM_ABLATION = 0.5
