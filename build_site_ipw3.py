"""
Build the IPW3 post-processing comparison website.

This file is the main site builder. Plot-specific logic is split into:
    - cutdata_builder.py
    - iceshape_builder.py
    - convergence_data_builder.py

The data reader/scanner remains gatherParticipantData.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from html import escape
import html
import re

from tools.gatherParticipantData import CASE_SLICES, VALID_CASES, VALID_GRID_LEVELS, HighlightPointsByCase, collect_case_ids, scan_all_participants
from tools.cutdata_builder import build_grid_level_cutdata_plots
from tools.iceshape_builder import build_ice_shape_section
from tools.convergence_data_builder import build_grid_convergence_section

ROOT_DIR = Path(".")
OUTPUT_HTML = ROOT_DIR / "index.html"
PAGES_DIR = ROOT_DIR / "PAGES"

def load_participants(root_dir: Path, highlight_points_by_case: HighlightPointsByCase | None = None, clean_s_cache: bool = False):
    participants = scan_all_participants(root_dir)
    for participant in participants:
        participant.read_files(highlight_points_by_case=highlight_points_by_case, clean_s_cache=clean_s_cache)
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


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "page"


def case_page_path(case_id: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}.html"


def convergence_page_path(case_id: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_grid_convergence.html"


def grid_page_path(case_id: str, grid_level: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_{grid_level.lower()}.html"


def link_from_root(path: Path) -> str:
    return path.as_posix()


def link_from_pages(path: Path) -> str:
    return path.relative_to(PAGES_DIR).as_posix()


def build_link_list(items: list[tuple[str, str, str]]) -> str:
    rows = ""
    for title, href, description in items:
        rows += f"""
        <a class="page-link-card" href="{escape(href)}">
          <strong>{escape(title)}</strong>
          <span>{escape(description)}</span>
        </a>
        """

    return f"""
    <nav class="page-link-grid">
      {rows}
    </nav>
    """


def build_nav_chip(label: str, href: str, active: bool = False) -> str:
    active_class = " active" if active else ""
    return f'<a class="nav-chip{active_class}" href="{escape(href)}">{escape(label)}</a>'


def build_page_navigation(case_ids: list[str], current_case_id: str | None = None, current_view: str | None = None) -> str:
    case_links = [build_nav_chip("Index", "../index.html" if current_case_id is not None else "index.html", active=current_case_id is None)]

    for case_id in case_ids:
        href = link_from_pages(case_page_path(case_id)) if current_case_id is not None else link_from_root(case_page_path(case_id))
        case_links.append(build_nav_chip(case_id, href, active=current_case_id == case_id and current_view == "case"))

    level_links = ""
    if current_case_id is not None:
        items = [build_nav_chip("Grid convergence", link_from_pages(convergence_page_path(current_case_id)), active=current_view == "grid_convergence")]
        for grid_level in sorted(VALID_GRID_LEVELS):
            items.append(build_nav_chip(grid_level, link_from_pages(grid_page_path(current_case_id, grid_level)), active=current_view == grid_level))
        level_links = f"""
        <div class="top-nav-row">
          <span class="top-nav-label">Views</span>
          {''.join(items)}
        </div>
        """

    return f"""
    <nav class="top-nav" aria-label="Page navigation">
      <div class="top-nav-row">
        <span class="top-nav-label">Cases</span>
        {''.join(case_links)}
      </div>
      {level_links}
    </nav>
    """


def build_case_index_section(case_ids: list[str]) -> str:
    items = []
    for case_id in case_ids:
        expected_slices = CASE_SLICES.get(case_id, [])
        expected_slices_text = ", ".join(f"Y = {value:g} m" for value in expected_slices) if expected_slices else "not specified"
        items.append((case_id, link_from_root(case_page_path(case_id)), f"Expected slice location(s): {expected_slices_text}."))

    return f"""
    <section class="participant-section">
      <h2>Cases</h2>
      {build_link_list(items)}
    </section>
    """


def build_case_landing_content(case_id: str) -> str:
    expected_slices = CASE_SLICES.get(case_id, [])
    expected_slices_text = ", ".join(f"Y = {value:g} m" for value in expected_slices) if expected_slices else "not specified"
    items = [("Grid convergence", link_from_pages(convergence_page_path(case_id)), "Case-level convergence plots from gridConvergence data.")]

    for grid_level in sorted(VALID_GRID_LEVELS):
        items.append((grid_level, link_from_pages(grid_page_path(case_id, grid_level)), f"CutData and ice-shape plots for {grid_level}."))

    return f"""
    <section class="participant-section">
      <h2>{escape(case_id)}</h2>
      <p class="case-meta standalone-meta">Expected slice location(s): {escape(expected_slices_text)}</p>
      {build_link_list(items)}
    </section>
    """


def build_grid_page_content(participants, case_id: str, grid_level: str) -> str:
    return f"""
    <section class="plot-subsection">
      <h3>{escape(case_id)} | {escape(grid_level)}</h3>
      {build_grid_level_plots(participants, case_id, grid_level)}
    </section>
    """


def build_convergence_page_content(participants, case_id: str) -> str:
    return f"""
    <section class="plot-subsection">
      <h3>{escape(case_id)} | Grid convergence</h3>
      {build_grid_convergence_section(participants, case_id)}
    </section>
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


def build_page_html(title: str, body_html: str, stylesheet_href: str = "style.css", back_href: str | None = None, back_label: str = "Index", nav_html: str = "") -> str:
    back_link = ""
    if back_href is not None:
        back_link = f'<a class="back-link" href="{escape(back_href)}">{escape(back_label)}</a>'

    return f"""
    <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>{escape(title)}</title>
        <link rel="stylesheet" href="{escape(stylesheet_href)}" />
        </head>
        <body>
        <header class="page-header">
            <div class="header-title-row">
            <img class="site-logo" src="{escape('../assets/ipw3_logo_small.png' if stylesheet_href.startswith('../') else 'assets/ipw3_logo_small.png')}" alt="IPW logo" />
            <div class="header-text">
                <h1>{escape(title)}</h1>
                {back_link}
            </div>
            </div>
        </header>

        <main class="page-content">
            {nav_html}
            {body_html}
        </main>
        </body>
    </html>
    """


def build_index_html(participants_table_html: str, case_index_html: str) -> str:
    return build_page_html("IPW3 Post-Processing", participants_table_html + case_index_html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the IPW3 post-processing comparison website.")
    parser.add_argument("--clean", action="store_true", help="Recompute cutData s mapping files instead of reusing existing *_sMap.dat files.")
    return parser.parse_args()


def write_case_pages(participants, case_ids: list[str]) -> None:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    for case_id in case_ids:
        case_html = build_page_html(
            title=f"{case_id}",
            body_html=build_case_landing_content(case_id),
            stylesheet_href="../style.css",
            back_href="../index.html",
            nav_html=build_page_navigation(case_ids, current_case_id=case_id, current_view="case"),
        )
        case_page_path(case_id).write_text(case_html, encoding="utf-8")

        convergence_html = build_page_html(
            title=f"{case_id} | Grid convergence",
            body_html=build_convergence_page_content(participants, case_id),
            stylesheet_href="../style.css",
            back_href=link_from_pages(case_page_path(case_id)),
            back_label=case_id,
            nav_html=build_page_navigation(case_ids, current_case_id=case_id, current_view="grid_convergence"),
        )
        convergence_page_path(case_id).write_text(convergence_html, encoding="utf-8")

        for grid_level in sorted(VALID_GRID_LEVELS):
            grid_html = build_page_html(
                title=f"{case_id} | {grid_level}",
                body_html=build_grid_page_content(participants, case_id, grid_level),
                stylesheet_href="../style.css",
                back_href=link_from_pages(case_page_path(case_id)),
                back_label=case_id,
                nav_html=build_page_navigation(case_ids, current_case_id=case_id, current_view=grid_level),
            )
            grid_page_path(case_id, grid_level).write_text(grid_html, encoding="utf-8")


def main() -> None:
    args = parse_args()

    # Per-case highlight point coordinates are (X, Y, Z). Use None for a
    # coordinate that should be taken from each cutData slice, usually Y.
    highlight_points_by_case: HighlightPointsByCase = {
        "TC_NACA0012_AE3932": (0.0, None, 0.0),
        "TC_NACA0012_AE3933": (0.0, None, 0.0),
        "TC_ONERAM6": (0.0, None, 0.0),
    }

    participants = load_participants(ROOT_DIR, highlight_points_by_case=highlight_points_by_case, clean_s_cache=args.clean)
    case_ids = get_case_ids(participants)

    participants_table_html = build_participants_table()
    case_index_html = build_case_index_section(case_ids)

    write_case_pages(participants, case_ids)
    index_html = build_page_html(
        "IPW3 Post-Processing",
        participants_table_html + case_index_html,
        nav_html=build_page_navigation(case_ids),
    )
    OUTPUT_HTML.write_text(index_html, encoding="utf-8")

    print(f"Wrote {OUTPUT_HTML}")
    print(f"Wrote pages in {PAGES_DIR}")
    print(f"Participants found: {len(participants)}")
    print(f"Cases included: {', '.join(case_ids)}")


if __name__ == "__main__":
    main()
