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
import shutil

from tools.gatherParticipantData import CASE_SLICES, VALID_CASES, VALID_GRID_LEVELS, HighlightPointsByCase, collect_case_ids, scan_all_participants
from tools.cutdata_builder import build_grid_level_cutdata_plots
from tools.iceshape_builder import build_ice_shape_section
from tools.convergence_data_builder import build_grid_convergence_section
from tools import convergence_data_builder, cutdata_builder, iceshape_builder
from tools.participant_style import PARTICIPANTS, PREVIEW_PARTICIPANT_NAME, normalize_participant_id, participant_color, participant_info, preview_participant_name

ROOT_DIR = Path(".")
OUTPUT_DIR = ROOT_DIR / "PREVIEW"
OUTPUT_HTML = OUTPUT_DIR / "index.html"
PAGES_DIR = OUTPUT_DIR / "PAGES"

VARIABLE_FILTER_SCRIPT = """
<script>
document.addEventListener("DOMContentLoaded", () => {
  const checkedValues = (root, selector) => new Set(
    Array.from(root.querySelectorAll(selector)).filter((input) => input.checked).map((input) => input.value)
  );

  document.querySelectorAll(".plot-filter-scope").forEach((scope, scopeIndex) => {
    const controls = scope.querySelector(":scope > .variable-filter-controls");
    const sections = Array.from(scope.querySelectorAll(":scope > .plot-subsection[data-variable-key]"));

    if (!controls || sections.length === 0) {
      if (controls) {
        controls.hidden = true;
      }
    } else {
      const title = controls.dataset.filterTitle || "Variables";
      const seen = new Map();

      sections.forEach((section) => {
        const key = section.dataset.variableKey || "";
        const label = section.dataset.variableLabel || key;
        if (key && !seen.has(key)) {
          seen.set(key, label);
        }
      });

      const checkboxHtml = Array.from(seen.entries()).map(([key, label], index) => {
        const id = `variable-filter-${scopeIndex}-${index}`;
        return `
          <label class="variable-filter-option" for="${id}">
            <input id="${id}" type="checkbox" value="${key}" />
            <span>${label}</span>
          </label>
        `;
      }).join("");

      controls.innerHTML = `
        <div class="variable-filter-header">
          <span>${title}</span>
          <button type="button" data-filter-action="all">All</button>
          <button type="button" data-filter-action="none">None</button>
        </div>
        <div class="variable-filter-options">${checkboxHtml}</div>
      `;

      const update = () => {
        const activeKeys = checkedValues(controls, "input[type='checkbox']");
        sections.forEach((section) => {
          section.hidden = !activeKeys.has(section.dataset.variableKey || "");
        });
        scope.dispatchEvent(new CustomEvent("plot-filter:update"));
      };

      controls.addEventListener("change", update);
      controls.addEventListener("click", (event) => {
        const button = event.target.closest("button[data-filter-action]");
        if (!button) {
          return;
        }
        const checked = button.dataset.filterAction === "all";
        controls.querySelectorAll("input[type='checkbox']").forEach((input) => {
          input.checked = checked;
        });
        update();
      });
      update();
    }

    const iceControls = scope.querySelector(":scope > .ice-shape-filter-controls");
    if (!iceControls) {
      return;
    }

    const groups = Array.from(scope.querySelectorAll(".ice-shape-subsection .slice-plot-group[data-slice-key][data-roughness-key]"));
    if (groups.length === 0) {
      iceControls.hidden = true;
      return;
    }

    const slices = new Map();
    const roughnessBySlice = new Map();

    groups.forEach((group) => {
      const sliceKey = group.dataset.sliceKey || "";
      const sliceLabel = group.dataset.sliceLabel || sliceKey;
      const roughnessKey = group.dataset.roughnessKey || "";
      const roughnessLabel = group.dataset.roughnessLabel || roughnessKey;

      if (!slices.has(sliceKey)) {
        slices.set(sliceKey, sliceLabel);
      }
      if (!roughnessBySlice.has(sliceKey)) {
        roughnessBySlice.set(sliceKey, new Map());
      }
      roughnessBySlice.get(sliceKey).set(roughnessKey, roughnessLabel);
    });

    const sliceCheckboxHtml = Array.from(slices.entries()).map(([key, label], index) => {
      const id = `ice-slice-filter-${scopeIndex}-${index}`;
      return `
        <label class="variable-filter-option" for="${id}">
          <input id="${id}" type="checkbox" value="${key}" data-ice-slice-toggle />
          <span>${label}</span>
        </label>
      `;
    }).join("");

    const roughnessHtml = Array.from(slices.entries()).map(([sliceKey, sliceLabel], sliceIndex) => {
      const options = Array.from(roughnessBySlice.get(sliceKey).entries()).map(([roughnessKey, roughnessLabel], roughnessIndex) => {
        const id = `ice-roughness-filter-${scopeIndex}-${sliceIndex}-${roughnessIndex}`;
        return `
          <label class="variable-filter-option" for="${id}">
            <input id="${id}" type="checkbox" value="${roughnessKey}" data-ice-roughness-toggle data-slice-key="${sliceKey}" />
            <span>${roughnessLabel}</span>
          </label>
        `;
      }).join("");

      return `
        <div class="ice-roughness-filter-group" data-slice-key="${sliceKey}">
          <div class="variable-filter-header">
            <span>${sliceLabel}</span>
            <button type="button" data-slice-key="${sliceKey}" data-roughness-action="all">All</button>
            <button type="button" data-slice-key="${sliceKey}" data-roughness-action="none">None</button>
          </div>
          <div class="variable-filter-options">${options}</div>
        </div>
      `;
    }).join("");

    iceControls.innerHTML = `
      <div class="variable-filter-header">
        <span>Ice-shape slices</span>
        <button type="button" data-slice-action="all">All</button>
        <button type="button" data-slice-action="none">None</button>
      </div>
      <div class="variable-filter-options">${sliceCheckboxHtml}</div>
      <div class="ice-roughness-filter-list">${roughnessHtml}</div>
    `;

    const updateIceFilters = () => {
      const activeSlices = checkedValues(iceControls, "input[data-ice-slice-toggle]");
      groups.forEach((group) => {
        const sliceKey = group.dataset.sliceKey || "";
        const roughnessKey = group.dataset.roughnessKey || "";
        const activeRoughness = checkedValues(iceControls, `input[data-ice-roughness-toggle][data-slice-key="${sliceKey}"]`);
        group.hidden = !activeSlices.has(sliceKey) || !activeRoughness.has(roughnessKey);
      });

      iceControls.querySelectorAll(".ice-roughness-filter-group").forEach((group) => {
        group.hidden = !activeSlices.has(group.dataset.sliceKey || "");
      });
    };

    iceControls.addEventListener("change", updateIceFilters);
    iceControls.addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (!button) {
        return;
      }

      if (button.dataset.sliceAction) {
        const checked = button.dataset.sliceAction === "all";
        iceControls.querySelectorAll("input[data-ice-slice-toggle]").forEach((input) => {
          input.checked = checked;
        });
        updateIceFilters();
      }

      if (button.dataset.roughnessAction) {
        const checked = button.dataset.roughnessAction === "all";
        const sliceKey = button.dataset.sliceKey || "";
        iceControls.querySelectorAll(`input[data-ice-roughness-toggle][data-slice-key="${sliceKey}"]`).forEach((input) => {
          input.checked = checked;
        });
        updateIceFilters();
      }
    });

    scope.addEventListener("plot-filter:update", updateIceFilters);
    updateIceFilters();
  });

  document.querySelectorAll("button[data-page-filter-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const checked = button.dataset.pageFilterAction === "show";
      document.querySelectorAll(".plot-filter-scope input[type='checkbox']").forEach((input) => {
        input.checked = checked;
      });
      document.querySelectorAll(".plot-filter-scope .variable-filter-controls").forEach((controls) => {
        controls.dispatchEvent(new Event("change", { bubbles: true }));
      });
    });
  });
});
</script>
"""

SLIDESHOW_SCRIPT = """
<script>
document.addEventListener("DOMContentLoaded", () => {
  const deck = document.querySelector("[data-slideshow-deck]");
  const source = document.querySelector("[data-slideshow-source]");
  const sidebar = document.querySelector("[data-slide-sidebar]");
  const layout = document.querySelector("[data-slideshow-layout]");
  const sidebarToggle = document.querySelector("[data-sidebar-toggle]");
  const counter = document.querySelector("[data-slide-counter]");
  const previousButton = document.querySelector("[data-slide-previous]");
  const nextButton = document.querySelector("[data-slide-next]");

  if (!deck || !source || !sidebar || !layout || !sidebarToggle || !counter || !previousButton || !nextButton) {
    return;
  }

  const slides = [];
  const groups = new Map();
  const plotContainers = Array.from(source.querySelectorAll(".plot-container")).filter(
    (container) => !container.closest(".plotly-graph-div")
  );

  plotContainers.forEach((plotContainer, index) => {
    const sourceGroup = plotContainer.closest("[data-slide-case][data-slide-view]");
    if (!sourceGroup) {
      return;
    }
    const plotSection = plotContainer.closest(".plot-subsection[data-variable-label]");
    const localGroup = plotContainer.closest(".slice-plot-group, .beta-mini-plot, .combined-beta-card");
    const caseLabel = sourceGroup.dataset.slideCase;
    const viewLabel = sourceGroup.dataset.slideView;
    const plotLabel = plotSection?.dataset.variableLabel
      || plotSection?.querySelector("h4")?.textContent?.trim()
      || "Plot";
    const detailLabel = localGroup?.querySelector("h5, h4")?.textContent?.trim() || "";
    const title = detailLabel && detailLabel !== plotLabel ? `${plotLabel} — ${detailLabel}` : plotLabel;
    const slideId = `slide-${index + 1}`;

    const slide = document.createElement("article");
    slide.className = "slideshow-slide";
    slide.id = slideId;
    slide.hidden = true;
    const slideHeader = document.createElement("header");
    slideHeader.className = "slideshow-slide-header";
    const slideContext = document.createElement("p");
    slideContext.textContent = `${caseLabel} · ${viewLabel}`;
    const slideTitle = document.createElement("h2");
    slideTitle.textContent = title;
    slideHeader.append(slideContext, slideTitle);
    slide.appendChild(slideHeader);

    const description = plotSection?.querySelector(":scope > .plot-description");
    if (description) {
      slide.appendChild(description.cloneNode(true));
    }
    slide.appendChild(plotContainer);
    deck.appendChild(slide);
    slides.push(slide);

    const groupKey = JSON.stringify([caseLabel, viewLabel]);
    if (!groups.has(groupKey)) {
      groups.set(groupKey, { caseLabel, viewLabel, items: [] });
    }
    groups.get(groupKey).items.push({ title, slideId, index });
  });

  source.remove();

  sidebarToggle.addEventListener("click", () => {
    const collapsed = layout.classList.toggle("sidebar-collapsed");
    sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
    sidebarToggle.textContent = collapsed ? "Show sidebar" : "Hide sidebar";
  });

  let currentCase = "";
  let currentCaseGroup = null;
  groups.forEach((group) => {
    if (group.caseLabel !== currentCase) {
      currentCaseGroup = document.createElement("details");
      currentCaseGroup.className = "slide-sidebar-case";
      currentCaseGroup.open = true;
      const caseSummary = document.createElement("summary");
      caseSummary.textContent = group.caseLabel;
      currentCaseGroup.appendChild(caseSummary);
      sidebar.appendChild(currentCaseGroup);
      currentCase = group.caseLabel;
    }

    const viewGroup = document.createElement("details");
    viewGroup.className = "slide-sidebar-view";
    viewGroup.open = true;
    const viewSummary = document.createElement("summary");
    viewSummary.textContent = group.viewLabel;
    viewGroup.appendChild(viewSummary);

    group.items.forEach((item) => {
      const link = document.createElement("a");
      link.href = `#${item.slideId}`;
      link.dataset.slideIndex = String(item.index);
      link.textContent = item.title;
      viewGroup.appendChild(link);
    });
    currentCaseGroup.appendChild(viewGroup);
  });

  let activeIndex = 0;
  const materializationPromises = new Map();

  const materializeSlide = (index) => {
    if (index < 0 || index >= slides.length) {
      return Promise.resolve();
    }
    if (materializationPromises.has(index)) {
      return materializationPromises.get(index);
    }

    const promise = (async () => {
      const frames = Array.from(slides[index].querySelectorAll("iframe.plotly-lazy-frame[data-plot-src]"));
      for (const frame of frames) {
        if (frame.src) {
          continue;
        }
        const loaded = new Promise((resolve) => {
          frame.addEventListener("load", resolve, { once: true });
          frame.addEventListener("error", resolve, { once: true });
        });
        frame.src = frame.dataset.plotSrc;
        await loaded;
        frame.parentNode.querySelectorAll(":scope > .plot-loading").forEach((loading) => loading.remove());
      }
    })();
    materializationPromises.set(index, promise);
    return promise;
  };

  const loadCurrentAndNext = async () => {
    await materializeSlide(activeIndex);
    window.dispatchEvent(new Event("resize"));
    void materializeSlide(activeIndex + 1);
  };

  const showSlide = (requestedIndex, updateHash = true) => {
    if (slides.length === 0) {
      counter.textContent = "No plots found";
      previousButton.disabled = true;
      nextButton.disabled = true;
      return;
    }

    activeIndex = Math.max(0, Math.min(requestedIndex, slides.length - 1));
    slides.forEach((slide, index) => {
      slide.hidden = index !== activeIndex;
    });
    sidebar.querySelectorAll("a[data-slide-index]").forEach((link) => {
      const active = Number(link.dataset.slideIndex) === activeIndex;
      link.classList.toggle("active", active);
      link.setAttribute("aria-current", active ? "page" : "false");
      if (active) {
        link.closest(".slide-sidebar-view").open = true;
        link.closest(".slide-sidebar-case").open = true;
        link.scrollIntoView({ block: "nearest" });
      }
    });
    counter.textContent = `Plot ${activeIndex + 1} of ${slides.length}`;
    previousButton.disabled = activeIndex === 0;
    nextButton.disabled = activeIndex === slides.length - 1;
    if (updateHash) {
      history.replaceState(null, "", `#${slides[activeIndex].id}`);
    }
    void loadCurrentAndNext();
  };

  sidebar.addEventListener("click", (event) => {
    const link = event.target.closest("a[data-slide-index]");
    if (!link) {
      return;
    }
    event.preventDefault();
    showSlide(Number(link.dataset.slideIndex));
  });
  previousButton.addEventListener("click", () => showSlide(activeIndex - 1));
  nextButton.addEventListener("click", () => showSlide(activeIndex + 1));
  document.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") {
      showSlide(activeIndex - 1);
    } else if (event.key === "ArrowRight") {
      showSlide(activeIndex + 1);
    }
  });

  const hashIndex = slides.findIndex((slide) => `#${slide.id}` === window.location.hash);
  showSlide(hashIndex >= 0 ? hashIndex : 0, hashIndex < 0);
});
</script>
"""

def load_participants(root_dir: Path, highlight_points_by_case: HighlightPointsByCase | None = None, clean_s_cache: bool = False, participant_id: str | None = None):
    participants = scan_all_participants(root_dir)
    if participant_id is not None:
        normalized_id = normalize_participant_id(participant_id)
        participants = [participant for participant in participants if normalize_participant_id(participant.participant_id) == normalized_id]

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


def order_slideshow_cases(case_ids: list[str]) -> list[str]:
    """Use the presentation order requested for the IPW3 cases."""
    preferred = ["TC_NACA0012_AE3932", "TC_NACA0012_AE3933", "TC_ONERAM6"]
    return [case_id for case_id in preferred if case_id in case_ids] + [
        case_id for case_id in case_ids if case_id not in preferred
    ]


def build_participants_table(participants_metadata: list[dict[str, str]], participant_id: str | None = None) -> str:
    header_cells = ["Participant ID", "Organization", "Solver(s)", "Name(s)"]
    header_cells_with_color = header_cells + ["Color"]
    normalized_filter = normalize_participant_id(participant_id) if participant_id is not None else None
    rows = []

    for participant in participants_metadata:
        metadata_id = normalize_participant_id(participant.get("Participant ID", ""))
        if normalized_filter is not None and metadata_id != normalized_filter:
            continue

        color = participant_color(metadata_id)

        row_html = "<tr>"

        for header in header_cells:
            row_html += f"<td>{html.escape(participant.get(header, ''))}</td>"

        row_html += f"""
            <td>
                <span class="participant-color-line" style="background-color: {html.escape(color)};"></span>
                <span>{html.escape(color)}</span>
            </td>
        """

        row_html += "</tr>"
        rows.append(row_html)

    header_html = "<tr>"
    for header in header_cells_with_color:
        header_html += f"<th>{html.escape(header)}</th>"
    header_html += "</tr>"

    if not rows:
        rows_html = f"""
        <tr>
            <td colspan="{len(header_cells_with_color)}">No participants found in participant_style.py.</td>
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


def preview_folder_slug(text: str) -> str:
    text = text.strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    return text.strip("_") or "PARTICIPANT"


def output_dir_for_participant(participant_id: str | None) -> Path:
    if participant_id is None:
        return ROOT_DIR / "PREVIEW"

    normalized_id = normalize_participant_id(participant_id)
    info = participant_info(normalized_id)
    organization = info.get("Organization", "") if info is not None else ""
    suffix = preview_folder_slug(organization or normalized_id)
    return ROOT_DIR / f"PREVIEW_{suffix}"


def configure_output_paths(participant_id: str | None) -> None:
    global OUTPUT_DIR, OUTPUT_HTML, PAGES_DIR

    OUTPUT_DIR = output_dir_for_participant(participant_id)
    OUTPUT_HTML = OUTPUT_DIR / "index.html"
    PAGES_DIR = OUTPUT_DIR / "PAGES"


def prepare_output_directory() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stylesheet_path = ROOT_DIR / "style.css"
    if stylesheet_path.exists():
        shutil.copy2(stylesheet_path, OUTPUT_DIR / "style.css")

    assets_path = ROOT_DIR / "assets"
    if assets_path.exists():
        shutil.copytree(assets_path, OUTPUT_DIR / "assets", dirs_exist_ok=True)


def case_page_path(case_id: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}.html"


def convergence_page_path(case_id: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_grid_convergence.html"


def grid_page_path(case_id: str, grid_level: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_{grid_level.lower()}.html"


def link_from_root(path: Path) -> str:
    return path.relative_to(OUTPUT_DIR).as_posix()


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
    <section class="page-filter-toolbar" aria-label="Page plot controls">
      <button type="button" data-page-filter-action="show">Show all plots</button>
      <button type="button" data-page-filter-action="hide">Hide all plots</button>
    </section>

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


def build_slideshow_content(participants, case_ids: list[str]) -> str:
    source_sections = ""
    plot_modules = (convergence_data_builder, cutdata_builder, iceshape_builder)
    plots_dir = OUTPUT_DIR / "PLOTS"
    for module in plot_modules:
        module.set_defer_plotly_html(plots_dir)

    try:
        for case_id in order_slideshow_cases(case_ids):
            ordered_views = [("Grid convergence", build_grid_convergence_section(participants, case_id))]
            ordered_views.extend(
                (grid_level, build_grid_level_plots(participants, case_id, grid_level))
                for grid_level in ("L4", "L3", "L2", "L1")
            )

            for view_label, plots_html in ordered_views:
                source_sections += f"""
                <section data-slide-case="{escape(case_id)}" data-slide-view="{escape(view_label)}">
                  {plots_html}
                </section>
                """
    finally:
        for module in plot_modules:
            module.set_defer_plotly_html(None)

    return f"""
    <div class="slideshow-layout" data-slideshow-layout>
      <aside class="slide-sidebar" aria-label="Plot navigation" data-slide-sidebar>
        <button class="slide-sidebar-toggle" type="button" data-sidebar-toggle aria-expanded="true">Hide sidebar</button>
      </aside>
      <section class="slideshow-stage">
        <div class="slideshow-deck" data-slideshow-deck></div>
        <nav class="slideshow-controls" aria-label="Slideshow controls">
          <button type="button" data-slide-previous>Previous</button>
          <span data-slide-counter>Loading plots…</span>
          <button type="button" data-slide-next>Next</button>
        </nav>
      </section>
    </div>
    <div data-slideshow-source>
      {source_sections}
    </div>
    {SLIDESHOW_SCRIPT}
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
        {VARIABLE_FILTER_SCRIPT}
        </body>
    </html>
    """


def build_index_html(participants_table_html: str, case_index_html: str) -> str:
    return build_page_html("IPW3 Post-Processing", participants_table_html + case_index_html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the IPW3 post-processing comparison website.")
    parser.add_argument("--clean", action="store_true", help="Recompute cutData s mapping files instead of reusing existing *_sMap.dat files.")
    parser.add_argument("--p", "--participant", dest="participant_id", help="Build a preview containing only one participant ID, for example --p 004.")
    parser.add_argument("--slides", action="store_true", help="Build index.html as a one-plot-per-slide presentation with sidebar and previous/next controls.")
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
    configure_output_paths(args.participant_id)

    # Per-case highlight point coordinates are (X, Y, Z). Use None for a
    # coordinate that should be taken from each cutData slice, usually Y.
    highlight_points_by_case: HighlightPointsByCase = {
        "TC_NACA0012_AE3932": (0.0, None, 0.0),
        "TC_NACA0012_AE3933": (0.0, None, 0.0),
        "TC_ONERAM6": (0.0, None, 0.0),
    }
    
    participants = load_participants(ROOT_DIR, highlight_points_by_case=highlight_points_by_case, clean_s_cache=args.clean, participant_id=args.participant_id)
    if args.participant_id is not None and not participants:
        requested_id = normalize_participant_id(args.participant_id)
        raise SystemExit(f"No participant folder found for ID {requested_id}.")

    case_ids = get_case_ids(participants)
    preview_name = preview_participant_name(args.participant_id) if args.participant_id is not None else PREVIEW_PARTICIPANT_NAME

    participants_table_html = build_participants_table(PARTICIPANTS, participant_id=args.participant_id)
    case_index_html = build_case_index_section(case_ids)

    prepare_output_directory()
    page_title = "IPW3 Post-Processing" if args.participant_id is None else f"IPW3 Post-Processing | {preview_name}"
    if args.slides:
        index_html = build_page_html(page_title, build_slideshow_content(participants, case_ids))
    else:
        write_case_pages(participants, case_ids)
        index_html = build_page_html(
            page_title,
            participants_table_html + case_index_html,
            nav_html=build_page_navigation(case_ids),
        )
    OUTPUT_HTML.write_text(index_html, encoding="utf-8")

    print(f"Wrote {OUTPUT_HTML}")
    if not args.slides:
        print(f"Wrote pages in {PAGES_DIR}")
    print(f"Mode: {'slideshow' if args.slides else 'standard'}")
    print(f"Participants found: {len(participants)}")
    print(f"Preview participant: {preview_name}")
    print(f"Cases included: {', '.join(case_ids)}")


if __name__ == "__main__":
    main()
