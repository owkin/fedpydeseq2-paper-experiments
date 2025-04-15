import re
from pathlib import Path

COMMANDS_FILE = Path(__file__).parent.parent.parent / "macros.tex"


def load_latex_commands() -> dict[str, str]:
    """Extract LaTeX commands and their definitions from a file.

    Returns
    -------
    dict[str, str]
        A dictionary mapping LaTeX command names to their definitions.
    """
    commands = {}
    with open(COMMANDS_FILE) as f:
        for line in f:
            if line.startswith("\\newcommand{"):
                # Extract command name and replacement
                match = re.match(r"\\newcommand{\\(\w+)}{(.*)}", line)
                if match:
                    cmd_name, replacement = match.groups()
                    commands[cmd_name] = replacement
            elif line.startswith("\\DeclareMathOperator{"):
                # Handle DeclareMathOperator
                match = re.match(r"\\DeclareMathOperator{\\(\w+)}{(.*)}", line)
                if match:
                    cmd_name, operator = match.groups()
                    # In math mode, operators should have proper spacing
                    commands[cmd_name] = f"\\operatorname{{{operator}}}"
    return commands


def replace_latex_commands(text: str, commands: dict[str, str]) -> str:
    r"""Replace LaTeX commands with their definitions.

    Replaces commands in order of length (longest first) to ensure that longer commands
    (like \countsmean) are replaced before their substrings (like \counts).

    Parameters
    ----------
    text : str
        The text containing LaTeX commands to be replaced.
    commands : dict[str, str]
        A dictionary mapping LaTeX command names to their definitions.

    Returns
    -------
    str
        The text with all LaTeX commands replaced by their definitions.
    """
    # Sort commands by length (longest first) to handle nested commands properly
    sorted_commands = sorted(commands.items(), key=lambda x: len(x[0]), reverse=True)

    # Keep replacing commands until no more replacements can be made
    prev_text = None
    while prev_text != text:
        prev_text = text
        for cmd, repl in sorted_commands:
            text = text.replace(f"\\{cmd}", repl)
    return text


def latex_table_to_markdown(latex_table: str, commands: dict[str, str]) -> str:
    r"""Convert LaTeX table to Markdown format.

    This function converts a LaTeX table to Markdown format, handling various LaTeX
    commands, math mode expressions, and table structure. It preserves the table's
    structure while converting LaTeX-specific syntax to Markdown-compatible format.

    Parameters
    ----------
    latex_table : str
        The LaTeX table as a string.
    commands : dict[str, str]
        A dictionary mapping LaTeX command names to their definitions.

    Returns
    -------
    str
        The table in Markdown format.

    Notes
    -----
    The function handles:
    - Table headers and separators
    - Math mode expressions
    - Multirow cells
    - Line breaks
    - Special characters in math mode
    """
    lines = latex_table.split("\n")
    markdown_lines = []
    header_found = False
    current_row = []
    last_id = None
    in_header = False

    # Define the fixed header
    header = "| ID | Name | Type | Shape | Description | Computed by | Sent to |"
    separator = "|---|---|---|---|---|---|---|"

    for line in lines:
        # Skip LaTeX-specific commands
        if any(
            cmd in line
            for cmd in ["\\begin", "\\end", "\\toprule", "\\bottomrule", "\\cline"]
        ):
            continue

        # Check if we're in the header section
        if (
            "Type" in line
            and "Shape" in line
            and "Description" in line
            and "Computed by" in line
            and "Sent to" in line
        ) or ("ID" in line and "Name" in line):
            in_header = True
            continue

        if "\\midrule" in line:
            if not header_found:
                # Add our fixed header and separator
                markdown_lines.append(header)
                markdown_lines.append(separator)
                header_found = True
            in_header = False  # End of header section
            continue

        # Skip processing if we're in the header section
        if in_header:
            continue

        # Remove \rowcolor commands
        line = re.sub(r"\\rowcolor{\w+}", "", line)

        # Handle multirow
        line = re.sub(r"\\multirow\[t\]{\d+}{\*}{([^}]+)}", r"\1", line)

        # Split the line into cells
        cells = line.split("&")

        # Process each cell
        processed_cells = []
        for cell in cells:
            # Remove leading/trailing whitespace and \\
            cell = cell.strip().rstrip("\\").strip()

            # Replace LaTeX commands with their definitions
            cell = replace_latex_commands(cell, commands)

            # Replace \newline with <br> for markdown line breaks
            cell = cell.replace("\\newline", "<br>")

            # Preserve math mode and ensure it's on a single line
            def replace_newlines_in_math(match):
                math_content = match.group(1)
                # Escape | characters in math mode
                math_content = math_content.replace("|", "\\|")
                return "$" + math_content.replace("\n", " ") + "$"

            cell = re.sub(r"\$([^$]+)\$", replace_newlines_in_math, cell)

            # Handle \operatorname in math mode
            cell = re.sub(r"\\operatorname{([^}]+)}", r"\1", cell)

            processed_cells.append(cell)

        # If we have cells and this is a data row
        if processed_cells:
            # Check if this row starts with an ID (a number)
            if processed_cells and processed_cells[0].strip().isdigit():
                last_id = processed_cells[0]
                current_row = processed_cells
            elif processed_cells:  # This is a continuation row
                # For continuation rows, we want to keep the ID but not
                # add an extra empty cell
                if last_id is not None:
                    # Start with just the ID and the rest of the cells
                    current_row = (
                        [last_id] + processed_cells[1:]
                        if len(processed_cells) > 1
                        else [last_id]
                    )

            # Ensure we have all 7 columns, filling empty ones with empty strings
            while len(current_row) < 7:
                current_row.append("")

            # End of row
            if "\\\\" in line:
                markdown_lines.append("| " + " | ".join(current_row) + " |")
                current_row = []

    # Don't add the last row if it's empty or incomplete
    if current_row and all(cell.strip() for cell in current_row):
        # Ensure we have all 7 columns for the last row too
        while len(current_row) < 7:
            current_row.append("")
        markdown_lines.append("| " + " | ".join(current_row) + " |")

    return "\n".join(markdown_lines)


def convert_latex_table_to_markdown(latex_table: str) -> str:
    """Convert a LaTeX table to Markdown format.

    This is a convenience function that loads LaTeX commands and converts a table
    in one step.

    Parameters
    ----------
    latex_table : str
        The LaTeX table as a string.

    Returns
    -------
    str
        The table in Markdown format.
    """
    # Load LaTeX commands from string
    commands = load_latex_commands()

    # Convert table
    markdown_table = latex_table_to_markdown(latex_table, commands)

    return markdown_table
