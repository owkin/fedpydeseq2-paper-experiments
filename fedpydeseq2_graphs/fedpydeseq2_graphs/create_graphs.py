import pathlib
import pickle
from pathlib import Path
from typing import Literal

import yaml  # type: ignore

from fedpydeseq2_graphs.constants import CONFIG_FILE
from fedpydeseq2_graphs.constants import PATHS_FILE
from fedpydeseq2_graphs.utils.data_classes import FunctionBlock
from fedpydeseq2_graphs.utils.data_classes import SharedState
from fedpydeseq2_graphs.utils.graph_utils import create_workflow_graph
from fedpydeseq2_graphs.utils.table_utils import create_table_from_dico


def get_workflow_name(
    max_depth: int | None, function_block_name: str | None, rank: int = 0
) -> str:
    """Get the name of the workflow."""
    if max_depth is not None:
        if function_block_name is not None:
            return (
                f"workflow_depth_{max_depth}_"
                f"function_block_{function_block_name}_rank_{rank}"
            )
        return f"workflow_depth_{max_depth}_rank_{rank}"
    if function_block_name is not None:
        return f"workflow_function_block_{function_block_name}_rank_{rank}"
    return f"entire_workflow_rank_{rank}"


def create_workflow_graph_and_tables(
    shared_states: dict[int, SharedState],
    function_blocks: list[FunctionBlock],
    output_directory: str | Path,
    max_depth: int | None = None,
    function_block_name: str | None = None,
    rank: int = 0,
    flatten_first_depth: bool = True,
    formats: list[Literal["png", "eps", "pdf", "svg"]] | None = None,
):
    """Create workflow graphs and tables from shared states and blocks.

    This function generates workflow graphs and tables based on
    the provided shared states and function blocks.
    It creates directories for storing the generated graphs
    and tables, and renders the workflow graphs with different
    naming conventions for shared states. It also creates a
    table from the shared states and saves it in CSV and LaTeX formats.

    Parameters
    ----------
    shared_states : dict[int, SharedState]
        A dictionary mapping shared state IDs to SharedState objects.

    function_blocks : list[FunctionBlock]
        A list of function blocks.

    output_directory : str | Path
        The directory where the output graphs and tables will be saved.

    max_depth : int | None, optional
        The maximum depth to plot, by default None.

    function_block_name : str | None, optional
        The name of the function block to plot, by default None.

    rank : int, optional
        The rank to search for, by default 0.

    flatten_first_depth : bool, optional
        Whether to flatten the first depth of the graph, by default True.

    formats : list[Literal["png", "eps", "pdf", "svg"]] | None, optional
        The formats to save the graphs in, by default None.

    Returns
    -------
    None
    """
    # Convert shared states dictionary to a list
    shared_states_list = [shared_states[i] for i in range(len(shared_states))]

    # Ensure the output directory exists
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    # Create a subdirectory for this function_block_name and max_depth
    workflow_name = get_workflow_name(max_depth, function_block_name, rank)
    workflow_dir = output_directory / workflow_name
    workflow_dir.mkdir(exist_ok=True, parents=True)

    # Create directories for tables and graphs
    tables_dir = workflow_dir / "tables"
    tables_dir.mkdir(exist_ok=True, parents=True)
    graphs_dir = workflow_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True, parents=True)

    # Set default formats if None
    if formats is None:
        formats = ["eps", "png"]

    # Create the initial version of the workflow graph
    nodes_info = create_workflow_graph(
        shared_states=shared_states_list,
        function_blocks=function_blocks,
        render_path=None,
        max_depth=max_depth,
        function_block_name=function_block_name,
        shared_state_naming="id",
        render=False,
        shared_state_mapping=None,
        rank=rank,
        formats=formats,
    )

    # Create a mapping from shared state IDs to new IDs
    present_shared_state_ids = sorted(nodes_info[0])
    old_ids_to_new_ids = {
        old_id: new_id for new_id, old_id in enumerate(present_shared_state_ids)
    }

    # Create a new version of the graph with the new IDs
    create_workflow_graph(
        shared_states=shared_states_list,
        function_blocks=function_blocks,
        render_path=str(graphs_dir / "workflow_graph"),
        max_depth=max_depth,
        function_block_name=function_block_name,
        shared_state_naming="id",
        render=True,
        shared_state_mapping=old_ids_to_new_ids,
        rank=rank,
        flatten_first_depth=flatten_first_depth,
        formats=formats,
    )

    # Create the workflow graph with shared_state_naming="content"
    create_workflow_graph(
        shared_states=shared_states_list,
        function_blocks=function_blocks,
        render_path=str(graphs_dir / "workflow_graph_with_naming"),
        max_depth=max_depth,
        function_block_name=function_block_name,
        shared_state_naming="content",
        render=True,
        shared_state_mapping=None,
        flatten_first_depth=flatten_first_depth,
        formats=formats,
    )

    # Create a table from the shared states and save it in CSV and LaTeX formats
    table, md_table, df = create_table_from_dico(
        shared_states,
        shared_with_aggregator=None,
        shared_state_mapping=old_ids_to_new_ids,
    )
    df.to_csv(tables_dir / "shared.csv")
    # For debugging purposes, save the Description column of the table to a latex
    # file
    description_str = "\n".join(df["Description"].dropna().tolist())
    with open(tables_dir / "descriptions.tex", "w") as f:
        f.write(description_str)
    with open(tables_dir / "shared.tex", "w") as f:
        f.write(table)
    with open(tables_dir / "shared.md", "w") as f:
        f.write(md_table)


def main():
    """Create the graph of the workflow."""
    # Loath the table_dir and processed_workflow_dir
    with open(PATHS_FILE) as f:
        paths = yaml.safe_load(f)

    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    output_dir = pathlib.Path(paths["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    processed_workflow_dir = pathlib.Path(paths["processed_workflow_dir"])

    # Load the shared_states.pkl file
    with open(processed_workflow_dir / "shared_states.pkl", "rb") as f:
        shared_states = pickle.load(f)

    # Load the function_blocks.pkl file
    with open(processed_workflow_dir / "function_blocks.pkl", "rb") as f:
        function_blocks = pickle.load(f)

    # Create the graphs
    graphs_info = config["graphs_to_build"]

    for function_block_name, max_depth, rank, flatten_first_depth in graphs_info:
        create_workflow_graph_and_tables(
            shared_states,
            function_blocks,
            output_directory=output_dir,
            function_block_name=function_block_name,
            max_depth=max_depth,
            rank=rank,
            flatten_first_depth=flatten_first_depth,
            formats=["eps", "png"],
        )


if __name__ == "__main__":
    main()
