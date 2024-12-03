import tempfile
from pathlib import Path
from typing import Any

from fedpydeseq2_datasets.constants import TCGADatasetNames
from fedpydeseq2_datasets.create_reference_dds import setup_tcga_ground_truth_dds
from fedpydeseq2_datasets.process_and_split_data import setup_tcga_dataset
from fedpydeseq2_datasets.utils import get_experiment_id
from fedpydeseq2_datasets.utils import get_ground_truth_dds_name
from fedpydeseq2_datasets.utils import get_n_centers

from paper_experiments.run_dge_gsea_methods.pydeseq2_pooled_tcga_pipe import (
    create_and_save_pydeseq2_stats_results,
)


def run_tcga_pydeseq2_per_center_experiments(
    dataset_name: TCGADatasetNames,
    raw_data_path: str | Path,
    results_path: str | Path,
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    reference_dds_ref_level: tuple[str, str] = ("stage", "Advanced"),
    heterogeneity_method: str | None = None,
    heterogeneity_method_param: float | None = None,
    **pydeseq2_kwargs: Any,
):
    """
    Run the TCGA experiments with per center data.

    Parameters
    ----------
    dataset_name : TCGADatasetNames
        The name of the dataset to use

    raw_data_path : str or Path
        The path to the raw data
        It is assumed that the raw data is in the following structure:
        ```
        <raw_data_path>
        ├── tcga
        │   ├── COHORT
        │   │   ├── Counts_raw.parquet
        │   │   └── recount3_metadata.tsv.gz
        │   ├── centers.csv
        │   ├── tumor_purity_metadata.csv
        │   └── cleaned_clinical_metadata.csv
        └──
        ```

    results_path : str or Path
        The path to save the results. At the end of this function, it will have the
        following structure:
        ```
        <results_path>
        ├── <experiment_id>
        │   ├── n_centers.txt
        │   └── center_0
        │       └── <ground_truth_dds_name>_stats_res.pkl
        └──
        ```

    small_samples : bool
        Whether to use a small number of samples. Default is False.

    small_genes : bool
        Whether to use a small number of genes. Default is False.

    only_two_centers : bool
        Whether to use only two centers. Default is False.

    design_factors : str or list[str]
        The design factors to use. Default is "stage".

    continuous_factors  : list[str] or None
        The continuous factors to use. Default is None.

    reference_dds_ref_level : tuple[str, str]
        The reference level of the design factor. Default is ("stage", "Advanced").

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
        Additional keyword arguments to pass to the pydeseq2 strategy.

    """
    with tempfile.TemporaryDirectory() as processed_data_path_str:
        processed_data_path = Path(processed_data_path_str)

        setup_tcga_dataset(
            raw_data_path,
            processed_data_path,
            dataset_name=dataset_name,
            small_samples=small_samples,
            small_genes=small_genes,
            only_two_centers=only_two_centers,
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            force=True,
            heterogeneity_method=heterogeneity_method,
            heterogeneity_method_param=heterogeneity_method_param,
            **pydeseq2_kwargs,
        )
        refit_cooks = pydeseq2_kwargs.get("refit_cooks", True)
        setup_tcga_ground_truth_dds(
            processed_data_path,
            dataset_name=dataset_name,
            small_samples=small_samples,
            small_genes=small_genes,
            only_two_centers=only_two_centers,
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            reference_dds_ref_level=reference_dds_ref_level,
            default_refit_cooks=True,
            force=True,
            pooled=False,
            heterogeneity_method=heterogeneity_method,
            heterogeneity_method_param=heterogeneity_method_param,
            **pydeseq2_kwargs,
        )
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

        n_centers = get_n_centers(
            processed_data_path=processed_data_path,
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
        # Save the number of centers in the results path, in the experiment id
        # as a simple text file
        n_centers_file = Path(results_path, experiment_id, "n_centers.txt")
        # make the parent directory
        n_centers_file.parent.mkdir(parents=True, exist_ok=True)
        with open(n_centers_file, "w") as f:
            f.write(str(n_centers))

        for center_id in range(n_centers):
            dds_filepath = Path(
                processed_data_path,
                "centers_data",
                "tcga",
                experiment_id,
                f"center_{center_id}",
                ground_truth_dds_name + ".pkl",
            )

            stats_res_file = Path(
                results_path,
                experiment_id,
                f"center_{center_id}",
                ground_truth_dds_name + "_stats_res.pkl",
            )

            create_and_save_pydeseq2_stats_results(
                dds_filepath,
                pydeseq2_kwargs,
                stats_res_file,
                center_id=center_id,
            )
