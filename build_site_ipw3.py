"""
Build the IPW3 post-processing comparison website.

This file is the main site builder. Plot-specific logic is split into:
    - cutdata_builder.py
    - iceshape_builder.py
    - convergence_data_builder.py

The data reader/scanner remains gatherParticipantData.py.
"""

from __future__ import annotations

from pathlib import Path
from html import escape
import html

from tools.gatherParticipantData import CASE_SLICES, VALID_CASES, VALID_GRID_LEVELS, collect_case_ids, scan_all_participants
from tools.cutdata_builder import build_grid_level_cutdata_plots
from tools.iceshape_builder import build_ice_shape_section
from tools.convergence_data_builder import build_grid_convergence_section

ROOT_DIR = Path(".")
OUTPUT_HTML = ROOT_DIR / "index.html"

def load_participants(root_dir: Path):
    participants = scan_all_participants(root_dir)
    for participant in participants:
        participant.read_files()
    return participants


def get_case_ids(participants) -> list[str]:
    """Return the cases that should be printed on the website.

    The restructured gatherParticipantData hierarchy is:
        participant.cases[case_id].grid_levels[grid_level].datasets[dataset_id]

    Therefore, case IDs are collected directly from participant.cases instead of
    from the old participant.datasets[...] structure.

    VALID_CASES are always printed first so cases with no data still appear with
    the "No matching ..." warning boxes. Any extra detected case folder is
    appended afterwards.
    """
    discovered = collect_case_ids(participants)

    ordered_valid_cases = sorted(VALID_CASES)
    extra_detected_cases = sorted(discovered - set(ordered_valid_cases))

    return ordered_valid_cases + extra_detected_cases


def build_participants_table() -> str:
    readme_path = Path("README.md")

    if not readme_path.exists():
        return """
        <section class="participant-section">
            <h2>Participants</h2>
        </section>
        """

    lines = readme_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    table_lines = []
    inside_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.append(stripped)
            inside_table = True
        else:
            if inside_table:
                break

    if len(table_lines) < 3:
        return """
        <section class="participant-section">
            <h2>Participants</h2>
        </section>
        """

    header_cells = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    data_lines = table_lines[2:]
    rows = []

    for line in data_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header_cells):
            continue
        row_html = "<tr>"
        for cell in cells:
            row_html += f"<td>{html.escape(cell)}</td>"
        row_html += "</tr>"
        rows.append(row_html)

    header_html = "<tr>"
    for header in header_cells:
        header_html += f"<th>{html.escape(header)}</th>"
    header_html += "</tr>"

    if not rows:
        rows_html = f"""
        <tr>
            <td colspan="{len(header_cells)}">No participants found in README.md.</td>
        </tr>
        """
    else:
        rows_html = "\n".join(rows)

    return f"""
    <section class="participant-section">
        <h2>Participants</h2>
        <p class="plot-description">
            IDs are for example only.
        </p>

        <div class="participant-table-wrapper">
            <table class="participant-table">
                <thead>
                    {header_html}
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </section>
    """


def build_grid_level_plots(participants, case_id: str, grid_level: str) -> str:
    html_output = ""
    html_output += build_grid_level_cutdata_plots(participants, case_id, grid_level)
    html_output += build_ice_shape_section(participants, case_id, grid_level)
    return html_output


def build_grid_level_section(participants, case_id: str, grid_level: str) -> str:
    grid_html = build_grid_level_plots(participants, case_id, grid_level)
    return f"""
    <details class="grid-section">
      <summary>{escape(grid_level)}</summary>
      {grid_html}
    </details>
    """


def build_case_section(participants, case_id: str) -> str:
    expected_slices = CASE_SLICES.get(case_id, [])
    expected_slices_text = ", ".join(f"Y = {value:g} m" for value in expected_slices) if expected_slices else "not specified"

    data_sections_html = ""

    data_sections_html += f"""
        <details class="grid-section grid-convergence-section">
          <summary>Grid convergence</summary>
          {build_grid_convergence_section(participants, case_id)}
        </details>
    """

    for grid_level in sorted(VALID_GRID_LEVELS):
        data_sections_html += build_grid_level_section(participants, case_id, grid_level)

    return f"""
    <details class="case-section">
      <summary>{escape(case_id)}</summary>

      <p class="case-meta">
        Expected slice location(s): {escape(expected_slices_text)}
      </p>

      <section class="grid-levels-wrapper">
        {data_sections_html}
      </section>
    </details>
    """


def build_index_html(participants_table_html: str, case_sections_html: str) -> str:
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
    participants = load_participants(ROOT_DIR)
    case_ids = get_case_ids(participants)

    participants_table_html = build_participants_table()
    case_sections_html = ""

    for case_id in case_ids:
        case_sections_html += build_case_section(participants, case_id)

    index_html = build_index_html(participants_table_html, case_sections_html)
    OUTPUT_HTML.write_text(index_html, encoding="utf-8")

    print(f"Wrote {OUTPUT_HTML}")
    print(f"Participants found: {len(participants)}")
    print(f"Cases included: {', '.join(case_ids)}")


if __name__ == "__main__":
    main()
