from pathlib import Path
import re
import plotly.graph_objects as go


ROOT_DIR = Path(".")

# -----------------------------------------------------------------------------
# Participant metadata
# -----------------------------------------------------------------------------
# This table is displayed at the top of the website and summarizes the
# participants included in the current post-processing.
PARTICIPANTS = [
    {
        "pid": "001",
        "organization": "Polytechnique Montréal",
        "solvers": "CHAMPS",
        "names": "XXX",
        "tfgs": "2",
    },
]

# -----------------------------------------------------------------------------
# Dataset configuration
# -----------------------------------------------------------------------------
# The clean and experimental files are only used for the ice-shape plot.
#
# Participant datasets now contain two paths:
#   iceShapePath:
#       Used for the ice-shape plot.
#
#   variablesPath:
#       Used for surface variables such as collection efficiency, Cp, HTC, etc.
#
# For now, variablesPath points to the same file as iceShapePath, but this can be
# changed later without modifying the plotting functions.
DATASETS = [
    {
        "kind": "clean",
        "name": "Clean shape",
        "path": "C00_Clean-Shape/C00_TC241_Airfoil/Case_241_Clean_shape.dat",
        "x_variable": "CoordinateX_Iced",
        "y_variable": "CoordinateY_Iced",
        "line_width": 2,
        "dash": "solid",
        "color": "black",
        "has_markers": False,
    },
    {
        "kind": "experimental",
        "name": "Experimental",
        "path": "E00_Experimental-Data/E00_TC241_MCCS/Case-241-MCCS_MODIFIED.dat",
        "x_variable": "CoordinateX_Iced",
        "y_variable": "CoordinateY_Iced",
        "line_width": 3,
        "dash": "solid",
        "color": "black",
        "has_markers": True,
        "marker_symbol": "diamond",
        "marker_color": "pink",
        "marker_size": 10,
        "marker_line_color": "black",
        "marker_line_width": 1,
        "marker_every": 15,
    },
    {
        "kind": "participant",
        "iceShapePath": "001_POLIMO_CHAMPS/001_TC241_01/SOLUTION_ICE_SHAPE_CASE_IPW1_241.dat",
        "variablesPath": "001_POLIMO_CHAMPS/001_TC241_01/SOLUTION_VARIABLES_CASE_IPW1_241.dat",
        "x_variable": "CoordinateX_Iced",
        "y_variable": "CoordinateY_Iced",
        "line_width": 3,
        "dash": "solid",
        "color": "blue",
        "has_markers": False,
    },
]


# -----------------------------------------------------------------------------
# Participant naming utilities
# -----------------------------------------------------------------------------

def get_dataset_path(dataset: dict, path_key: str = "path") -> str:
    """
    Return the correct path for a dataset.

    Clean and experimental datasets use:
        path

    Participant datasets can use:
        iceShapePath
        variablesPath

    The fallback to "path" keeps the function compatible with older dataset
    definitions.
    """
    if path_key in dataset:
        return dataset[path_key]

    if "path" in dataset:
        return dataset["path"]

    raise KeyError(f"Dataset does not contain path key: {path_key}")


def get_participant_id_from_path(path: str) -> str:
    """
    Extract the Participant ID, PID, from the first folder.

    Example:
        001_POLIMO_CHAMPS/... -> 001
    """
    folder_name = Path(path).parts[0]
    match = re.match(r"^(\d{3})", folder_name)
    return match.group(1) if match else "Unknown"


def get_dataset_id_from_path(path: str) -> str:
    """
    Extract the Dataset ID, DID, from the case/submission folder.

    Example:
        001_TC241_01/... -> 01
    """
    parts = Path(path).parts

    if len(parts) < 2:
        return "Unknown"

    dataset_folder = parts[1]
    match = re.match(r"^\d{3}_TC\d+_(\d{2})", dataset_folder)

    return match.group(1) if match else "Unknown"


def get_dataset_name(dataset: dict) -> str:
    """
    Return the legend name using the format PID.DID.

    Example:
        PID = 001
        DID = 01
        legend = 001.01
    """
    if dataset["kind"] == "participant":
        path = get_dataset_path(dataset, "iceShapePath")
        pid = get_participant_id_from_path(path)
        did = get_dataset_id_from_path(path)
        return f"{pid}.{did}"

    return dataset["name"]


# -----------------------------------------------------------------------------
# Tecplot .dat parsing utilities
# -----------------------------------------------------------------------------

def extract_variables(text: str) -> list[str]:
    """
    Extract Tecplot variable names from the VARIABLES section.

    This searches only between VARIABLES and ZONE to avoid reading quoted strings
    from the TITLE section.
    """
    match = re.search(r"VARIABLES\s*=\s*([\s\S]*?)\bZONE\b", text, re.IGNORECASE)

    if not match:
        raise ValueError("Could not find VARIABLES section.")

    variables_text = match.group(1)
    return [variable.strip() for variable in re.findall(r'"([^"]+)"', variables_text)]


def extract_numbers(text: str) -> list[float]:
    """
    Extract all numerical values from a text block.

    Supports integers, decimals, and scientific notation.
    """
    pattern = r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?"
    return [float(value) for value in re.findall(pattern, text)]


def get_data_section(text: str) -> str:
    """
    Return the numerical data section of a Tecplot file.

    The function skips the header and starts from the first line beginning with
    a number.
    """
    lines = text.splitlines()

    for index, line in enumerate(lines):
        if re.match(r"^\s*[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?", line):
            return "\n".join(lines[index:])

    raise ValueError("No numeric data found.")


def get_ordered_point_count(text: str) -> int | None:
    """
    Extract the number of ordered points from the Tecplot ZONE header.

    Example:
        ZONE I=2001, DATAPACKING=BLOCK

    returns:
        2001
    """
    match = re.search(r"\bI\s*=\s*(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def is_block_packing(text: str) -> bool:
    """
    Check whether the Tecplot file uses BLOCK data packing.

    In BLOCK format, values are stored variable by variable:
        x1 x2 x3 ... xN
        y1 y2 y3 ... yN

    In POINT-like format, values are stored point by point:
        x1 y1
        x2 y2
        x3 y3
    """
    return re.search(r"DATAPACKING\s*=\s*BLOCK", text, re.IGNORECASE) is not None


def parse_tecplot_dat(path: str, x_variable: str, y_variable: str) -> tuple[list[float], list[float]]:
    """
    Parse a Tecplot .dat file and extract two variables.

    Parameters
    ----------
    path:
        Path to the Tecplot .dat file.

    x_variable:
        Variable to use on the x-axis.

    y_variable:
        Variable to use on the y-axis.

    Returns
    -------
    x, y:
        Lists of extracted values.
    """
    text = Path(path).read_text()

    variables = extract_variables(text)

    if x_variable not in variables:
        raise ValueError(f"Variable not found: {x_variable} in {path}. Available variables: {variables}")

    if y_variable not in variables:
        raise ValueError(f"Variable not found: {y_variable} in {path}. Available variables: {variables}")

    x_index = variables.index(x_variable)
    y_index = variables.index(y_variable)

    data_text = get_data_section(text)
    values = extract_numbers(data_text)
    n_variables = len(variables)

    if is_block_packing(text):
        n_points = get_ordered_point_count(text)

        if n_points is None:
            raise ValueError(f"BLOCK Tecplot file found, but I=... could not be read in {path}")

        x_start = x_index * n_points
        y_start = y_index * n_points

        x = values[x_start:x_start + n_points]
        y = values[y_start:y_start + n_points]

        return x, y

    n_rows = len(values) // n_variables

    x = []
    y = []

    for row in range(n_rows):
        x.append(values[row * n_variables + x_index])
        y.append(values[row * n_variables + y_index])

    return x, y


# -----------------------------------------------------------------------------
# Generic Plotly trace construction
# -----------------------------------------------------------------------------

def add_dataset_to_figure(fig: go.Figure, dataset: dict, path_key: str = "path") -> None:
    path = get_dataset_path(dataset, path_key)
    x, y = parse_tecplot_dat(path, dataset["x_variable"], dataset["y_variable"])
    name = get_dataset_name(dataset)

    legend_group = name
    has_markers = dataset.get("has_markers", False)

    # Case 1: line only, no markers.
    if not has_markers:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                legendgroup=legend_group,
                showlegend=True,
                line=dict(
                    color=dataset["color"],
                    width=dataset["line_width"],
                    dash=dataset["dash"],
                ),
            )
        )
        return

    # Case 2: line with markers on every point.
    # This is used if has_markers=True but marker_every is not defined.
    if "marker_every" not in dataset:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=name,
                legendgroup=legend_group,
                showlegend=True,
                line=dict(
                    color=dataset["color"],
                    width=dataset["line_width"],
                    dash=dataset["dash"],
                ),
                marker=dict(
                    symbol=dataset.get("marker_symbol", "circle"),
                    color=dataset.get("marker_color", dataset["color"]),
                    size=dataset.get("marker_size", 6),
                    line=dict(
                        color=dataset.get("marker_line_color", dataset["color"]),
                        width=dataset.get("marker_line_width", 0),
                    ),
                ),
            )
        )
        return

    # Case 3: line using all points + markers every N points.
    # The real line and real markers are hidden from the legend.
    # A dummy trace is added only to display one clean legend item with line+marker.
    marker_every = dataset["marker_every"]

    # Full line using all points.
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            legendgroup=legend_group,
            showlegend=False,
            line=dict(
                color=dataset["color"],
                width=dataset["line_width"],
                dash=dataset["dash"],
            ),
        )
    )

    # Real markers, drawn only every marker_every points.
    fig.add_trace(
        go.Scatter(
            x=x[::marker_every],
            y=y[::marker_every],
            mode="markers",
            name=f"{name} markers",
            legendgroup=legend_group,
            showlegend=False,
            marker=dict(
                symbol=dataset.get("marker_symbol", "circle"),
                color=dataset.get("marker_color", dataset["color"]),
                size=dataset.get("marker_size", 6),
                line=dict(
                    color=dataset.get("marker_line_color", dataset["color"]),
                    width=dataset.get("marker_line_width", 0),
                ),
            ),
        )
    )

    # Legend-only trace, not drawn in the plot area.
    # This creates one legend item showing both the line and the marker.
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines+markers",
            name=name,
            legendgroup=legend_group,
            showlegend=True,
            line=dict(
                color=dataset["color"],
                width=dataset["line_width"],
                dash=dataset["dash"],
            ),
            marker=dict(
                symbol=dataset.get("marker_symbol", "circle"),
                color=dataset.get("marker_color", dataset["color"]),
                size=dataset.get("marker_size", 6),
                line=dict(
                    color=dataset.get("marker_line_color", dataset["color"]),
                    width=dataset.get("marker_line_width", 0),
                ),
            ),
        )
    )


# -----------------------------------------------------------------------------
# Ice-shape figure
# -----------------------------------------------------------------------------

def build_ice_shape_figure() -> go.Figure:
    """
    Build the interactive Plotly figure for the ice-shape comparison.

    This figure includes:
        - clean shape,
        - experimental shape,
        - participant ice shape.
    """
    fig = go.Figure()

    for dataset in DATASETS:
        if dataset["kind"] == "participant":
            add_dataset_to_figure(fig, dataset, path_key="iceShapePath")
        else:
            add_dataset_to_figure(fig, dataset, path_key="path")

    fig.update_layout(
        font=dict(
            family="Arial, Helvetica, sans-serif",
            size=18,
        ),
        autosize=True,
        height=760,

        # No Plotly title here because the subsection title already identifies the plot.
        title=None,

        xaxis=dict(
            title=dict(text="X [m]", font=dict(size=22)),
            range=[-0.0173187471892, 0.0349345794985],
            tickmode="linear",
            dtick=0.01,
            tickformat=".3f",

            visible=True,
            showticklabels=True,
            ticks="outside",
            ticklen=8,
            tickwidth=2,
            tickcolor="black",
            tickfont=dict(size=18, color="black"),
            showline=True,
            linecolor="black",
            linewidth=2,
            mirror=True,

            showgrid=True,
            gridcolor="lightgray",
            zeroline=False,
        ),

        yaxis=dict(
            title=dict(text="Y [m]", font=dict(size=22)),
            range=[-0.0183502382957, 0.0280971632045],
            tickmode="linear",
            dtick=0.01,
            tickformat=".3f",

            visible=True,
            showticklabels=True,
            ticks="outside",
            ticklen=8,
            tickwidth=2,
            tickcolor="black",
            tickfont=dict(size=18, color="black"),
            showline=True,
            linecolor="black",
            linewidth=2,
            mirror=True,

            showgrid=True,
            gridcolor="lightgray",
            zeroline=False,

            # Equal aspect ratio for geometric comparison.
            scaleanchor="x",
            scaleratio=1,
        ),

        legend=dict(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
            groupclick="togglegroup",
        ),

        margin=dict(l=90, r=40, t=30, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


# -----------------------------------------------------------------------------
# Collection-efficiency figure
# -----------------------------------------------------------------------------

def build_collection_efficiency_figure() -> go.Figure:
    """
    Build the collection-efficiency plot for participant datasets.

    For now, this figure only includes participant submissions. Clean and
    experimental datasets are ignored because they do not contain beta data.

    The expected variables are:
        x-axis: Surface Distance from Highlight
        y-axis: Beta

    If your Tecplot file uses different variable names, modify the two strings
    in parse_tecplot_dat() below.
    """
    fig = go.Figure()

    for dataset in DATASETS:
        if dataset["kind"] != "participant":
            continue

        path = get_dataset_path(dataset, "variablesPath")

        # Modify these variable names if your file uses a different convention.
        s, beta = parse_tecplot_dat(
            path=path,
            x_variable="s",
            y_variable="CollectionEfficiency",
        )

        name = get_dataset_name(dataset)

        fig.add_trace(
            go.Scatter(
                x=s,
                y=beta,
                mode="lines",
                name=name,
                showlegend=True,
                line=dict(
                    color=dataset["color"],
                    width=dataset["line_width"],
                    dash=dataset["dash"],
                ),
            )
        )

    fig.update_layout(
        font=dict(
            family="Arial, Helvetica, sans-serif",
            size=18,
        ),
        autosize=True,
        height=620,

        # No Plotly title here because the subsection title already identifies the plot.
        title=None,

        xaxis=dict(
            title=dict(text="Surface Distance from Highlight [m]", font=dict(size=22)),
            tickmode="auto",
            tickformat=".3f",

            visible=True,
            showticklabels=True,
            ticks="outside",
            ticklen=8,
            tickwidth=2,
            tickcolor="black",
            tickfont=dict(size=18, color="black"),
            showline=True,
            linecolor="black",
            linewidth=2,
            mirror=True,

            showgrid=True,
            gridcolor="lightgray",
            zeroline=False,
        ),

        yaxis=dict(
            title=dict(text="Collection Efficiency [-]", font=dict(size=22)),
            tickmode="auto",
            tickformat=".3f",

            visible=True,
            showticklabels=True,
            ticks="outside",
            ticklen=8,
            tickwidth=2,
            tickcolor="black",
            tickfont=dict(size=18, color="black"),
            showline=True,
            linecolor="black",
            linewidth=2,
            mirror=True,

            showgrid=True,
            gridcolor="lightgray",
            zeroline=False,
        ),

        legend=dict(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
        ),

        margin=dict(l=100, r=40, t=30, b=90),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


# -----------------------------------------------------------------------------
# Plotly HTML conversion
# -----------------------------------------------------------------------------

def plotly_config(filename: str) -> dict:
    """
    Plotly configuration shared by all figures.

    The filename is used when the user downloads the figure as PNG from the
    Plotly toolbar.
    """
    return {
        "responsive": True,
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "height": 900,
            "width": 1200,
            "scale": 3,
        },
    }


def figure_to_html_div(fig: go.Figure, filename: str) -> str:
    """
    Convert a Plotly figure to an HTML div.

    full_html=False means the figure is inserted inside our custom index.html
    rather than generating a standalone HTML page.
    """
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config=plotly_config(filename),
    )


# -----------------------------------------------------------------------------
# Website generation
# -----------------------------------------------------------------------------
def build_participants_table(participants: list[dict]) -> str:
    """
    Build a non-collapsible participant information table.

    Columns:
        - Participant ID (PID)
        - Organization
        - Solver(s)
        - Name(s)
        - TFG(s)
    """
    rows_html = ""

    for participant in participants:
        rows_html += f"""
          <tr>
            <td>{participant["pid"]}</td>
            <td>{participant["organization"]}</td>
            <td>{participant["solvers"]}</td>
            <td>{participant["names"]}</td>
            <td>{participant["tfgs"]}</td>
          </tr>
        """

    return f"""
    <section class="info-section">
      <h2>Participant Information</h2>

      <div class="table-wrapper">
        <table class="participant-table">
          <thead>
            <tr>
              <th>Participant ID (PID)</th>
              <th>Organization</th>
              <th>Solver(s)</th>
              <th>Name(s)</th>
              <th>TFG(s)</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
      </div>
    </section>
    """

def build_case_section(case_title: str, subsections: list[dict]) -> str:
    """
    Build one collapsible test-case section.

    Each subsection corresponds to one figure type, for example:
        - Ice Shape Plot
        - Collection Efficiency Plot
        - Cp Plot
        - CL Grid Convergence
    """
    subsection_html = ""

    for subsection in subsections:
        subsection_html += f"""
        <section class="plot-subsection">
          <h3>{subsection["title"]}</h3>
          <p class="plot-description">{subsection["description"]}</p>
          <div class="plot-container">
            {subsection["figure_html"]}
          </div>
        </section>
        """

    return f"""
    <details class="case-section" open>
      <summary>{case_title}</summary>
      {subsection_html}
    </details>
    """


def build_index_html(participants_table_html: str, case_sections_html: str) -> str:
    """
    Build the complete website HTML.

    The participant table is non-collapsible.
    The test-case sections below it are collapsible.
    The visual styling is handled by the static style.css file.
    """
    return f"""
        <!DOCTYPE html>

        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>IPW3 Post-Processing</title>
            <link rel="stylesheet" href="style.css" />
        </head>
            <body>
                <header class="page-header">
                    <div class="header-title-row">
                        <img class="site-logo" src="assets/ipw3_logo_small.png" alt="IPW logo" />
                        <div class="header-text">
                        <h1>IPW3 Post-Processing</h1>
                        </div>
                    </div>
                </header>
                <main class="page-content">
                    {participants_table_html}
                    {case_sections_html}
                </main>
            </body>
        </html>
    """


def main() -> None:
    """
    Main entry point.

    This function:
        - builds the ice-shape figure,
        - builds the collection-efficiency figure,
        - inserts both figures under Test Case 241,
        - writes the generated index.html at the repository root.
    """
    ice_shape_fig = build_ice_shape_figure()
    collection_efficiency_fig = build_collection_efficiency_figure()

    test_case_241_html = build_case_section(
        case_title="Test Case 241",
        subsections=[
            {
                "title": "Ice Shape Plot",
                "description": "Legend Correponds to Participant ID . Dataset ID",
                "figure_html": figure_to_html_div(
                    ice_shape_fig,
                    filename="IPW1_Case_241_ice_shape",
                ),
            },
            {
                "title": "Collection Efficiency Plot",
                "description": "Legend Correponds to Participant ID . Dataset ID",
                "figure_html": figure_to_html_div(
                    collection_efficiency_fig,
                    filename="IPW1_Case_241_collection_efficiency",
                ),
            },
        ],
    )

    participants_table_html = build_participants_table(PARTICIPANTS)
    index_html = build_index_html(participants_table_html, test_case_241_html)

    (ROOT_DIR / "index.html").write_text(index_html)


if __name__ == "__main__":
    main()