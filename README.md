
# FedPyDESeq2 paper experiments

The goal of this repository is to provide all code and utils necessary to generate the
FedPyDESeq2 paper plots.

## Setup

### 1. Clone the repository
Start by cloning this repository and navigating to its root directory.

```sh
git clone https://github.com/owkin/fedpydeseq2-paper-experiments.git
cd fedpydeseq2-paper-experiments
```

### 2. Create a conda environment
Create a Conda environment with Python 3.11 to ensure reproducibility.

```sh
conda create -n fedpydeseq2_paper_experiments python=3.11
conda activate fedpydeseq2_paper_experiments
```

### 3. Install poetry
Install Poetry to manage dependencies.

```sh
conda install pip
pip install poetry==1.8.2
```

### 4. Install dependencies
Install the necessary dependencies using Poetry, including the linting group if needed.

```sh
poetry install --with linting
```

### 5. Download the data

To download the raw data, please use the provided command `fedpydeseq2-download-data` to download it to the
desired path.

```bash
fedpydeseq2-download-data --raw_data_output_path <path>
```

You can pass the `conda` activation path as an argument as well, for example:

```bash
fedpydeseq2-download-data --raw_data_output_path <path> --conda_activate_path /opt/miniconda/bin/activate
```

## How to use this repository

This repository provides three main commands once installed:

1. **`run-dge-gsea`**:
   - Executes Differential Gene Expression (DGE) experiments and performs Gene Set Enrichment Analysis (GSEA) on the DGE results.
   - Requires an experiment configuration file and a paths configuration file.

2. **`make-fedpydeseq2-remote-script`**:
   - Generates the necessary scripts to run `fedpydeseq2` on a remote server for real Federated Learning (FL) execution.
   - Requires an experiment configuration file and a paths configuration file.

3. **`make-plots`**:
   - Creates the desired figures based on the DGE and GSEA results, as specified in the configuration file.
   - Requires an experiment configuration file and a paths configuration file.

Each of these commands can run on an experiment configuration file and a paths configuration file. Alternatively, you can provide just the experiment name which will point to the configuration file.

### Configuration files

1. **Experiment configuration file**:
   - Sets up an experiment to compare different DGE methods using the same settings defined by `pydeseq2_parameters`.
   - Configures the GSEA run for this comparison.
   - Specifies the figures to be created from the DGE and GSEA results.

2. **Paths configuration file**:
   - Specifies the paths where the raw data is stored, which is used to run the DGE experiments.
   - Defines where to save the results for each experiment.

#### Differential expression methods

In the package, we run several Differential Gene Expression (DGE) methods to compare their performance and results. The available methods are:

- **`fedpydeseq2_simulated`**: Simulates federated learning by splitting the data into multiple subsets and run the federated pipeline locally.
- **`meta_analysis`**: Can be run using various techniques, including:
  - P-value combination methods: Stouffer's and Fisher's methods.
  - Fixed effect model.
  - Random effect model with either a one-step DerSimonian-Laird estimator or an iterated method.
- **`pydeseq2_largest`**: Runs the DGE analysis on the largest cohort available, providing a baseline for comparison.
- **`pydeseq2`**: The standard approach, running the analysis on the entire dataset.
- **`fedpydeseq2_remote`**: Designed for real federated learning scenarios, where the analysis is performed on remote servers, and each dataset is stored only on its corresponding remote server.

#### Figures

For now, we implement three types of figures to visualize the results:

- **Heterogeneity plots**: Visualize the variability in gene expression across different centers or batches.
- **Cross tables**: Compare the results of different DGE methods by displaying the overlap and differences in identified genes.
- **GSEA plots**: Visualize the results of Gene Set Enrichment Analysis, showing the enrichment scores and significance of gene sets.





### Example experiment configuration

```yaml
pydeseq2_parameters:
  design_factors:
  - stage
  continuous_factors: null
  refit_cooks: null
  contrast: null
keep_original_centers: true
datasets:
- TCGA-LUAD
- TCGA-LUSC
- TCGA-READ
- TCGA-COAD
- TCGA-SKCM
- TCGA-PAAD
- TCGA-BRCA
- TCGA-PRAD
heterogeneity: null
meta_analysis: META_ANALYSIS_PARAMETERS
run_dge:
- pydeseq2
- fedpydeseq2_simulated
- meta_analysis
- pydeseq2_largest
run_gsea:
- fedpydeseq2_simulated
- meta_analysis
- pydeseq2_largest
- pydeseq2
- fedpydeseq2_remote
plots:
  cross_table:
    log2fc_threshold: 2.0
    padj_threshold: 0.05
    method_pairs:
    - ['fedpydeseq2_simulated', 'pydeseq2']
    - ['fedpydeseq2_remote', 'pydeseq2']
    - ['fedpydeseq2_simulated', 'fedpydeseq2_remote']
    - ['meta_analysis', 'pydeseq2']
    - ['pydeseq2_largest', 'pydeseq2']
  gsea_plots:
    method_pairs:
    - ['fedpydeseq2_simulated', 'pydeseq2']
    - ['fedpydeseq2_remote', 'pydeseq2']
    - ['meta_analysis', 'pydeseq2']
    - ['pydeseq2_largest', 'pydeseq2']
    plot_parameters:
    - with_diff: true
      padj_threshold: 0.05
    - with_diff: false
      padj_threshold: 0.05
    - with_diff: true
      padj_threshold: 0.05
      with_pvalues: true
    - with_diff: false
      padj_threshold: 0.05
      with_pvalues: true
```

### Configuration breakdown

1. **pydeseq2_parameters**:
   - **design_factors**: Specifies the factors used in the design matrix for the analysis. Here, it includes `stage`.
   - **continuous_factors**: Specifies continuous factors used in the analysis. Set to `null` indicating no continuous factors are used.
   - **refit_cooks**: Parameter for refitting Cook's distances. Set to `null`.
   - **contrast**: Specifies the contrast parameter for the analysis. Set to `null`.

2. **keep_original_centers**:
   - A boolean flag indicating whether to keep the original centers. Set to `true`.

3. **datasets**:
   - Lists the datasets to be used in the analysis. Examples include `TCGA-LUAD`, `TCGA-LUSC`, `TCGA-READ`, etc.

4. **heterogeneity**:
   - Parameter for heterogeneity analysis. Set to `null`.

5. **meta_analysis**:
   - Placeholder for meta-analysis parameters. Set to `META_ANALYSIS_PARAMETERS`.

6. **run_dge**:
   - Specifies the DGE methods to be run. The methods listed are:
     - `pydeseq2`
     - `fedpydeseq2_simulated`
     - `meta_analysis`
     - `pydeseq2_largest`

7. **run_gsea**:
   - Specifies the Gene Set Enrichment Analysis (GSEA) methods to be run. The methods listed are:
     - `fedpydeseq2_simulated`
     - `meta_analysis`
     - `pydeseq2_largest`
     - `pydeseq2`
     - `fedpydeseq2_remote`

8. **plots**:
   - Specifies the plotting configurations for visualizing the results of the DGE and GSEA analyses.

   - **cross_table**:
     - **log2fc_threshold**: Log2 fold change threshold for filtering results. Set to `2.0`.
     - **padj_threshold**: Adjusted p-value threshold for filtering results. Set to `0.05`.
     - **method_pairs**: Lists pairs of methods to be compared in the cross table. Examples include:
       - `['fedpydeseq2_simulated', 'pydeseq2']`
       - `['fedpydeseq2_remote', 'pydeseq2']`
       - `['fedpydeseq2_simulated', 'fedpydeseq2_remote']`
       - `['meta_analysis', 'pydeseq2']`
       - `['pydeseq2_largest', 'pydeseq2']`

   - **gsea_plots**:
     - **method_pairs**: Lists pairs of methods to be compared in the GSEA plots. Examples include:
       - `['fedpydeseq2_simulated', 'pydeseq2']`
       - `['fedpydeseq2_remote', 'pydeseq2']`
       - `['meta_analysis', 'pydeseq2']`
       - `['pydeseq2_largest', 'pydeseq2']`
     - **plot_parameters**: Specifies parameters for generating the GSEA plots. Examples include:
       - `with_diff: true, padj_threshold: 0.05`
       - `with_diff: false, padj_threshold: 0.05`
       - `with_diff: true, padj_threshold: 0.05, with_pvalues: true`
       - `with_diff: false, padj_threshold: 0.05, with_pvalues: true`

### Paths configuration and file structure

You must configure the paths in the `experiment_paths.yaml` in the repository. This specifies where you want to save the different experiment results, and where you want to store the remote experiment results if you want to run them. Note that these are not run using the `run-dge-gsea` method, and must be run separately using the helper functions defined in the next section.

#### YAML template

```yaml
experiments:
  single_factor:
    results: "path/to/local/results"
    remote_results: "path/to/remote/results"
    remote_processed: "path/to/remote/processed"
    remote_credentials: "path/to/remote/credentials"
  multi_factor:
    results: "path/to/local/results"
    remote_results: "path/to/remote/results"
    remote_processed: "path/to/remote/processed"
    remote_credentials: "path/to/remote/credentials"
  continuous_factor:
    results: "path/to/local/results"
    remote_results: "path/to/remote/results"
    remote_processed: "path/to/remote/processed"
    remote_credentials: "path/to/remote/credentials"
  heterogeneity:
    results: "path/to/local/results"
    remote_results: "path/to/remote/results"
    remote_processed: "path/to/remote/processed"
    remote_credentials: "path/to/remote/credentials"
raw_data_path: "path/to/raw/data"
```

For a given experiment, the following directories will be created.

```
experiments[experiment_name][results]/
    ├── dge_results/
    │   ├── pydeseq2/
    │   │   ├── <experiment_id>/
    │   │   └── artefacts/
    │   ├── fedpydeseq2_simulated/
    │   │   ├── <experiment_id>/
    │   │   └── artefacts/
    │   ├── meta_analysis/
    │   │   ├── <experiment_id>/
    │   │   └── artefacts/
    │   ├── pydeseq2_largest/
    │   │   ├── <experiment_id>/
    │   │   └── artefacts/
    ├── gsea_results/
    │   ├── pydeseq2/
    │   │   └── <experiment_id>/
    │   ├── fedpydeseq2_simulated/
    │   │   └── <experiment_id>/
    │   ├── meta_analysis/
    │   │   └── <experiment_id>/
    │   ├── pydeseq2_largest/
    │   │   └── <experiment_id>/
    │   └── fedpydeseq2_remote/
    │       └── <experiment_id>/
    ├── figures/
    │   ├── heterogeneity_plots/
    │   ├── cross_tables/
    │   └── gsea_plots/
```


### Experiments performed in this repository

We perform four main experiments in this repository, each configured with a file located in the [`experiment_specifications`](TODO) directory at the root of the repository:

#### 1. Single factor analysis
This experiment focuses on analyzing the effect of a single factor on gene expression. It uses the DESeq2 model to identify differentially expressed genes based on a single condition or factor. The configuration for this experiment is specified in the `experiment_specifications/single_factor_specs.yaml` file. The datasets used include various TCGA datasets such as TCGA-LUAD, TCGA-LUSC, and others.

#### 2. Multi-factor analysis
This experiment extends the single factor analysis by considering multiple factors simultaneously. It uses the DESeq2 model to identify differentially expressed genes while accounting for the combined effects of multiple conditions or factors. The configuration for this experiment is specified in the `experiment_specifications/multi_factor_specs.yaml` file. Factors such as stage and gender are considered, and datasets include TCGA-LUAD, TCGA-LUSC, TCGA-READ, and others.

#### 3. Continuous factor analysis
This experiment investigates the effect of continuous factors on gene expression. It uses the DESeq2 model to identify differentially expressed genes based on continuous variables. The configuration for this experiment is specified in the `experiment_specifications/continuous_factor_specs.yaml` file. Continuous factors such as CPE are considered, and datasets include TCGA-LUAD, TCGA-LUSC, TCGA-READ, and others.

#### 4. Heterogeneity analysis
This experiment investigates the heterogeneity in gene expression across different centers or batches. It uses the DESeq2 model to identify differentially expressed genes while accounting for batch effects and other sources of heterogeneity. The configuration for this experiment is specified in the `experiment_specifications/heterogeneity_specs.yaml` file. The heterogeneity method used is "binomial", and datasets include TCGA-NSCLC and TCGA-CRC.


## Running Differential Gene Expression with different methods and Gene Set Enrichment Analysis (GSEA) on top

The `dge_gsea_pipe.py` script is used to run Differential Gene Expression (DGE) for different methods to be compared and Gene Set Enrichment Analysis (GSEA) methods on top on specified datasets. This script is located in the `fedpydeseq2-paper-experiments/paper_experiments/run_dge_gsea_methods/` directory.

NB: this does not concern the running of fedpydeseq2 on a remote server. The dedicated section can be found below.

### Overview
The script performs the following tasks:

1. Loads the configuration and paths files.
2. Runs the specified DGE methods on the datasets (but not the remote fedpydeseq2).
3. Runs GSEA on the results of the DGE methods (including the remote fedpydeseq2 results, if specified).

### Experiment configuration
The script requires a configuration file that specifies the parameters for the DGE and GSEA methods. This configuration file is typically located in the `experiment_specifications` directory and is named according to the experiment being run (e.g., `single_factor_specs.yaml`, `multi_factor_specs.yaml`).

### Paths configuration
Ensure that the `experiment_paths.yaml` file is up to date. This file specifies where the raw data is stored and where to save the results for each experiment.

### Running the script

To run the `run-dge-gsea` command, follow these steps:

1. **Ensure Configuration Files are Ready**:
   - Prepare the experiment configuration file in the [`experiment_specifications`]() directory.
   - Ensure the `experiment_paths.yaml` file is correctly set up.

2. **Execute the Command**:
   - Use the following command to run the script, replacing `<experiment_name>` with the name of your experiment configuration file (without the `.yaml` extension).

```sh
run-dge-gsea --experiment_name <experiment_name>
```

This command will load the specified configuration, run the DGE methods on the datasets (excluding the remote fedpydeseq2), and perform GSEA on the results (including the remote fedpydeseq2 results, if specified).

## Running fedpydeseq2 remotely

### Configuration

To run one of the experiments remotely, you must fill in your `experiment_paths.yaml` with the fields related to the remote setting. For example, for the heterogeneity experiment:

```yaml
experiments:
  heterogeneity:
    results: ""
    remote_results: "my path"
    remote_processed: "path to processed data"
    remote_credentials: "path to remote credentials"
raw_data_path: "important"
```

- **results**: Local path to save the results.
- **remote_results**: Remote path to save the results.
- **remote_processed**: Remote path to the processed data.
- **remote_credentials**: Path to the credentials for accessing remote resources.

The credentials are what allow you to access the Substra client (see Substra documentation).

### Creating a bash script

After you fill in the configuration file, run the following command:

```sh
make-fedpydeseq2-remote-scripts --experiment_name <experiment_name>
```

- **experiment_name**: Must be a key in the [`experiments`]() part of the `experiment_paths.yaml` file and have a corresponding `<experiment_name>_specs.yaml` file in the `experiment_specifications` directory.

This command will create a bash script in the [`fedpydeseq2_remote_scripts`]() subfolder of this repository.

### Example output script

Here is an example of what the generated script might look like. Note that all explicit paths have been replaced with placeholders:

```shell
# You might need to run
# `chmod 777 /path/to/fedpydeseq2_remote_scripts/single_factor_remote.sh`
# to give execution rights to the script
for dataset in 'TCGA-LUAD' 'TCGA-LUSC' 'TCGA-READ' 'TCGA-COAD' 'TCGA-SKCM' 'TCGA-PAAD' 'TCGA-BRCA' 'TCGA-PRAD'
do
    fedpydeseq2-tcga-pipe \
    --register_data \
    --backend "remote" \
    --dataset_name "${dataset}" \
    --keep_original_centers \
    --save_filepath "/path/to/results"  \
    --raw_data_path "/path/to/raw_data" \
    --processed_data_path "/path/to/processed_data"  \
    --credentials_path "/path/to/credentials.yaml" \
    --design_factors "stage" \
    --force_preprocessing
done
```

### Executing the bash script

1. Ensure you are in the Conda environment created during setup.
2. Navigate to the [`fedpydeseq2_remote_scripts`]() subfolder.
3. Execute the generated bash script.

```sh
cd fedpydeseq2_remote_scripts
bash <generated_script>.sh
```

This will run the `fedpydeseq2` experiment remotely using the specified configuration.

## Making Plots

The `make-plots` command is configured to run the `main` function from the `plot_pipe` module located in the `paper_experiments.figures` package.

### Usage

Run the `make-plots` command to generate the new plot:

```sh
make-plots --experiment_name your_experiment_name
```

### How `make-plots` Works

1. **Command Execution**:
   - When you run `make-plots` from the command line, it triggers the `main` function in the `plot_pipe` module.

2. **Loading Configuration**:
   - The `main` function typically starts by loading the experiment configuration file and the paths configuration file. These files contain the necessary parameters and paths for generating the plots.

3. **Generating Plots**:
   - The `main` function then processes the DGE and GSEA results based on the configurations specified in the YAML file.
   - It uses the plotting parameters defined in the `plots` section of the configuration file to generate the desired figures.

4. **Saving Plots**:
   - The generated plots are saved to the specified locations, as defined in the paths configuration file.

### Cross Tables

The `cross_table` plot is designed to compare the results of different Differential Gene Expression (DGE) methods. Here's a detailed breakdown of how it works:

#### Configuration

In the YAML configuration file, the `cross_table` section specifies the parameters for generating the cross table plots:

```yaml
plots:
  cross_table:
    log2fc_threshold: 2.0
    padj_threshold: 0.05
    method_pairs:
    - ['fedpydeseq2_simulated', 'pydeseq2']
    - ['fedpydeseq2_remote', 'pydeseq2']
    - ['fedpydeseq2_simulated', 'fedpydeseq2_remote']
    - ['meta_analysis', 'pydeseq2']
    - ['pydeseq2_largest', 'pydeseq2']
```

- **log2fc_threshold**: Log2 fold change threshold for filtering results.
- **padj_threshold**: Adjusted p-value threshold for filtering results.
- **method_pairs**: List of method pairs to be compared in the cross table.

#### Implementation in `plot_pipe.py`

The `run_plot_pipe` function in `plot_pipe.py` handles the generation of the cross table plots:

1. **Load Configuration**: Extracts the `cross_table` configuration from the YAML file.
2. **Prepare Paths**: Creates the directory for saving cross table plots.
3. **Filter Results**: Filters the DGE results based on the specified thresholds.
4. **Generate Cross Tables**: Compares the results of the specified method pairs and generates the cross tables.
5. **Save Cross Tables**: Saves the generated cross tables to the specified path.

#### Utility Function: `build_test_vs_ref_cross_tables`

The `build_test_vs_ref_cross_tables` function is responsible for generating the cross table plots. It:

1. Loads results for the specified method pairs.
2. Filters the results based on the log fold change and adjusted p-value thresholds.
3. Creates the cross table by comparing the filtered results.
4. Saves the cross table to the specified path.

### GSEA Plots

The `gsea_plots` section in the configuration file specifies the parameters for generating Gene Set Enrichment Analysis (GSEA) plots. Here's a detailed breakdown of how it works:

#### Configuration

In the YAML configuration file, the `gsea_plots` section specifies the parameters for generating the GSEA plots:

```yaml
plots:
  gsea_plots:
    method_pairs:
    - ['fedpydeseq2_simulated', 'pydeseq2']
    - ['fedpydeseq2_remote', 'pydeseq2']
    - ['meta_analysis', 'pydeseq2']
    - ['pydeseq2_largest', 'pydeseq2']
    plot_parameters:
    - with_diff: true
      padj_threshold: 0.05
    - with_diff: false
      padj_threshold: 0.05
    - with_diff: true
      padj_threshold: 0.05
      with_pvalues: true
    - with_diff: false
      padj_threshold: 0.05
      with_pvalues: true
```

- **method_pairs**: List of method pairs to be compared in the GSEA plots.
- **plot_parameters**: List of parameters for generating the GSEA plots, such as whether to include differential expression (`with_diff`) and the adjusted p-value threshold (`padj_threshold`).

#### Implementation in `plot_pipe.py`

The `run_plot_pipe` function in `plot_pipe.py` handles the generation of the GSEA plots:

1. **Load Configuration**: Extracts the `gsea_plots` configuration from the YAML file.
2. **Prepare Paths**: Creates the directory for saving GSEA plots.
3. **Check Methods**: Ensures that the methods specified in `method_pairs` are included in the list of methods to run for GSEA.
4. **Generate GSEA Plots**: Calls the utility function to generate the GSEA plots for each method pair and set of plot parameters.
5. **Save GSEA Plots**: Saves the generated GSEA plots to the specified path.

#### Utility Function: `make_gsea_plot_method_pairs`

The `make_gsea_plot_method_pairs` function is responsible for generating the GSEA plots. It:

1. **Load Results**: Loads the GSEA results for the specified method pairs.
2. **Filter Results**: Filters the results based on the specified plot parameters.
3. **Generate Plots**: Creates the GSEA plots by comparing the filtered results of the method pairs.
4. **Save Plots**: Saves the generated plots to the specified path.

### Heterogeneity Plot

The `heterogeneity` plot is designed to analyze and visualize the effect of the heterogeneity of centers in the results of different Differential Gene Expression (DGE) methods. Here's a detailed breakdown of how it works:

#### Configuration

In the YAML configuration file, the `heterogeneity` section specifies the parameters for generating the heterogeneity plots:

```yaml
plots:
  heterogeneity:
    method_test: ['fedpydeseq2_simulated', 'pydeseq2']
    method_ref: 'pydeseq2'
    scoring_function_names: ['scoring_function_1', 'scoring_function_2']
```

- **method_test**: List of methods to be tested for heterogeneity.
- **method_ref**: Reference method against which the test methods are compared.
- **scoring_function_names**: List of scoring functions used to evaluate heterogeneity.

#### Implementation in `plot_pipe.py`

The `run_plot_pipe` function in `plot_pipe.py` handles the generation of the heterogeneity plots:

1. **Load Configuration**: Extracts the `heterogeneity` configuration from the YAML file.
2. **Prepare Paths**: Creates the directory for saving heterogeneity plots.
3. **Generate Heterogeneity Plots**: Calls the utility function to generate the heterogeneity plots for each dataset and scoring function.
4. **Save Heterogeneity Plots**: Saves the generated heterogeneity plots to the specified path.

#### Utility Function: `build_test_vs_ref_heterogeneity_plot`

The `build_test_vs_ref_heterogeneity_plot` function is responsible for generating the heterogeneity plots. It:

1. **Load Results**: Loads the DGE results for the test methods and the reference method.
2. **Score Methods**: Uses the specified scoring functions to score the test methods vs the reference methods.
3. **Generate Plots**: Creates the heterogeneity plots based on the calculated scores.
4. **Save Plots**: Saves the generated plots to the specified path.

### Example Configuration for Remote Execution

To run the heterogeneity experiment remotely, you need to configure the `experiment_paths.yaml` file with the necessary paths and credentials:

```yaml
experiments:
  heterogeneity:
    results: ""
    remote_results: "my path"
    remote_processed: "path to processed data"
    remote_credentials: "path to remote credentials"
raw_data_path: "important"
```

- **results**: Path to save the local results.
- **remote_results**: Path to save the remote results.
- **remote_processed**: Path to the processed data on the remote server.
- **remote_credentials**: Path to the credentials for accessing the remote server.
- **raw_data_path**: Path to the raw data.

### Adding a New Plot to the `make-plots` Command

To add a new plot to the `plot_pipe.py` script, follow these steps:

1. **Define the New Plot Configuration**:
   - Add a new section in the configuration file (YAML) for the new plot.

2. **Update the `run_plot_pipe` Function**:
   - Add logic to handle the new plot configuration.
   - Call the appropriate utility function to generate the new plot.

3. **Create the Utility Function**:
   - Implement a new utility function to generate the plot.

4. **Update the Documentation**:
   - Update the README.md file to include information about the new plot.

#### 1. Define the New Plot Configuration

Add a new section in your YAML configuration file for the new plot. For example:

```yaml
plots:
  new_plot:
    parameter1: value1
    parameter2: value2
```

#### 2. Update the `run_plot_pipe` Function

Modify the `run_plot_pipe` function to handle the new plot configuration:

```python
def run_plot_pipe(
    config: dict,
    paths: dict,
    raw_data_path: str | Path,
) -> None:
    ...
    plots_config = config["plots"]

    ...

    # Generate new plot
    if "new_plot" in plots_config:
        new_plot_config = plots_config["new_plot"]
        new_plot_path = plot_results_path / "new_plot"
        new_plot_path.mkdir(parents=True, exist_ok=True)

        # Call the utility function to generate the new plot
        generate_new_plot(
            config=new_plot_config,
            save_path=new_plot_path,
            dataset_names=dataset_names,
            design_factors=design_factors,
            continuous_factors=continuous_factors,
            meta_analysis_parameters=meta_analysis_parameters,
            **pydeseq2_kwargs,
        )
```

#### 3. Create the Utility Function

Implement the utility function to generate the new plot:

```python
def generate_new_plot(
    config: dict,
    save_path: Path,
    dataset_names: list,
    design_factors: list,
    continuous_factors: list,
    meta_analysis_parameters: dict,
    **kwargs,
) -> None:
    """Generate the new plot.

    Parameters
    ----------
    config : dict
        The configuration dictionary for the new plot.
    save_path : Path
        The path to save the new plot.
    dataset_names : list
        The list of dataset names.
    design_factors : list
        The list of design factors.
    continuous_factors : list
        The list of continuous factors.
    meta_analysis_parameters : dict
        The meta analysis parameters.
    kwargs : dict
        Additional keyword arguments.
    """
    # Implement the logic to generate the new plot
    ...
```

#### 4. Update the Documentation

Update the README.md file to include information about the new plot:

```markdown
### New Plot

This section describes how to configure and generate the new plot.

#### Configuration

Add the following section to your YAML configuration file:


```
