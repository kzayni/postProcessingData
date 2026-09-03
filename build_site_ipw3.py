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
import tempfile

from tools.gatherParticipantData import CASE_SLICES, VALID_CASES, VALID_GRID_LEVELS, HighlightPointsByCase, cleanup_generated_sidecars, collect_case_ids, scan_all_participants
from tools.cutdata_builder import build_combined_levels_cutdata_section, build_grid_level_cutdata_plots
from tools.iceshape_builder import build_combined_levels_ice_shape_section, build_ice_shape_section
from tools.convergence_data_builder import build_ae3933_ice_mass_comparison_section, build_beta_max_analysis_section, build_grid_convergence_section, build_water_mass_analysis_section
from tools import convergence_data_builder, cutdata_builder, iceshape_builder
from tools.latex_builder import build_latex_preview
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
            <input id="${id}" type="checkbox" value="${key}" checked />
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
          <input id="${id}" type="checkbox" value="${key}" data-ice-slice-toggle checked />
          <span>${label}</span>
        </label>
      `;
    }).join("");

    const onlyCombinedRoughness = Array.from(roughnessBySlice.values()).every((options) =>
      options.size === 1 && options.has("all_roughness")
    );

    const roughnessHtml = Array.from(slices.entries()).map(([sliceKey, sliceLabel], sliceIndex) => {
      const options = Array.from(roughnessBySlice.get(sliceKey).entries()).map(([roughnessKey, roughnessLabel], roughnessIndex) => {
        const id = `ice-roughness-filter-${scopeIndex}-${sliceIndex}-${roughnessIndex}`;
        return `
          <label class="variable-filter-option" for="${id}">
            <input id="${id}" type="checkbox" value="${roughnessKey}" data-ice-roughness-toggle data-slice-key="${sliceKey}" checked />
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
      <div class="ice-roughness-filter-list" ${onlyCombinedRoughness ? "hidden" : ""}>${roughnessHtml}</div>
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

# HTML plot categories selected by --convergence, --cutdata, and --iceshape.
# With no category flags, all three categories are enabled.
HTML_BUILD_SECTIONS: set[str] = {"convergence", "cutdata", "iceshape"}


def html_section_enabled(section: str) -> bool:
    return section in HTML_BUILD_SECTIONS

PARTICIPANT_DETAILS_SCRIPT = """
<script>
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-participant-dialog-open]").forEach((button) => {
    button.addEventListener("click", () => {
      const dialog = document.getElementById(button.dataset.participantDialogOpen || "");
      if (dialog) {
        dialog.showModal();
      }
    });
  });

  document.querySelectorAll("[data-participant-dialog-close]").forEach((button) => {
    button.addEventListener("click", () => button.closest("dialog")?.close());
  });

  document.querySelectorAll(".participant-dialog").forEach((dialog) => {
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) {
        dialog.close();
      }
    });
  });
});
</script>
"""

SITE_SIDEBAR_SCRIPT = """
<script>
document.addEventListener("DOMContentLoaded", () => {
  const sidebar = document.querySelector("[data-site-sidebar]");
  const hideButton = document.querySelector("[data-site-sidebar-hide]");
  const showButton = document.querySelector("[data-site-sidebar-show]");
  if (!sidebar || !hideButton || !showButton) return;

  const setHidden = (hidden) => {
    document.body.classList.toggle("site-sidebar-hidden", hidden);
    showButton.hidden = !hidden;
    try { localStorage.setItem("ipw3-sidebar-hidden", hidden ? "1" : "0"); } catch (_) {}
  };
  let initiallyHidden = false;
  try { initiallyHidden = localStorage.getItem("ipw3-sidebar-hidden") === "1"; } catch (_) {}
  setHidden(initiallyHidden);
  hideButton.addEventListener("click", () => setHidden(true));
  showButton.addEventListener("click", () => setHidden(false));
});
</script>
"""

PLOT_DOWNLOAD_SCRIPT = """
<script>
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-plot-download]").forEach((button) => {
    button.addEventListener("click", async () => {
      const shell = button.closest(".plot-download-shell");
      const graph = shell?.querySelector(".plotly-graph-div");
      if (!shell || !graph || !window.Plotly) return;

      const withLegend = button.dataset.plotDownload === "with-legend";
      const original = {
        showlegend: graph.layout.showlegend,
        titleText: graph.layout.title?.text || "",
        titleX: graph.layout.title?.x,
        marginRight: graph.layout.margin?.r,
        marginTop: graph.layout.margin?.t,
      };
      const filename = shell.dataset.plotFilename || "plot";
      const title = shell.dataset.plotTitle || "";
      const downloadWidth = Number(shell.dataset.downloadWidth) || 1350;
      const downloadHeight = Number(shell.dataset.downloadHeight) || 900;
      button.disabled = true;

      try {
        await Plotly.relayout(graph, {
          showlegend: withLegend,
          "title.text": title,
          "title.x": 0.5,
          "title.xanchor": "center",
          "margin.t": 100,
          "margin.r": withLegend ? 260 : 60,
        });
        await Plotly.downloadImage(graph, {
          format: "png",
          filename: `${filename}_${withLegend ? "with_legend" : "without_legend"}`,
          width: downloadWidth,
          height: downloadHeight,
          scale: 3,
        });
      } finally {
        await Plotly.relayout(graph, {
          showlegend: original.showlegend,
          "title.text": original.titleText,
          "title.x": original.titleX,
          "margin.r": original.marginRight,
          "margin.t": original.marginTop,
        });
        button.disabled = false;
      }
    });
  });
});
</script>
"""

LINKED_CONVERGENCE_LEGEND_SCRIPT = r"""
<script>
document.addEventListener("DOMContentLoaded", () => {
  const participantId = (trace) => {
    const text = `${trace?.name || ""} ${trace?.legendgroup || ""}`;
    return text.match(/(?:^|\D)(\d{3})(?!\d)/)?.[1] || "";
  };
  const finite = (values) => values.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  const percentile = (values, fraction) => {
    if (!values.length) return null;
    const index = (values.length - 1) * fraction;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    return values[lower] + (values[upper] - values[lower]) * (index - lower);
  };
  const format = (value, percent = false) => {
    if (value === null || !Number.isFinite(value)) return "n/a";
    return `${Number(value.toPrecision(6))}${percent ? "%" : ""}`;
  };

  document.querySelectorAll('[data-linked-participant-legend="grid-convergence"]').forEach((group) => {
    const graphs = Array.from(group.querySelectorAll(".plotly-graph-div"));
    if (graphs.length < 3 || !window.Plotly) return;
    const [rawGraph, relativeGraph] = graphs;
    const boxGraph = graphs.find((graph, index) => index >= 2 &&
      (graph.data || []).some((trace) => trace.type === "box" && trace.customdata));
    const table = group.querySelector(".statistical-table");
    const participantSummary = group.querySelector(".statistical-participants");
    if (!boxGraph || !table) return;

    const pointRecords = [];
    (boxGraph.data || []).forEach((trace, traceIndex) => {
      const id = participantId(trace) || String(trace.customdata?.[0]?.[0] || "");
      const level = String(trace.x?.[0] || "").toUpperCase();
      const value = Number(trace.y?.[0]);
      if (trace.type === "box" && id && level && Number.isFinite(value)) {
        pointRecords.push({id, level, value, traceIndex});
      }
    });
    const summaryTraceByLevel = new Map();
    (boxGraph.data || []).forEach((trace, traceIndex) => {
      if (trace.type === "box" && !trace.customdata && /^L[1-4]$/i.test(String(trace.name || ""))) {
        summaryTraceByLevel.set(String(trace.name).toUpperCase(), traceIndex);
      }
    });

    const activeParticipants = () => new Set(
      (rawGraph.data || [])
        .filter((trace) => trace.visible !== false && trace.visible !== "legendonly")
        .map(participantId)
        .filter(Boolean)
    );

    const updateDependents = async () => {
      const active = activeParticipants();
      const relativeVisibility = (relativeGraph.data || []).map((trace) => {
        const id = participantId(trace);
        return id ? (active.has(id) ? true : "legendonly") : trace.visible;
      });
      await Plotly.restyle(relativeGraph, {visible: relativeVisibility});

      const boxVisibility = (boxGraph.data || []).map((trace) => {
        const id = participantId(trace) || String(trace.customdata?.[0]?.[0] || "");
        return id ? (active.has(id) ? true : false) : trace.visible;
      });
      for (const level of ["L1", "L2", "L3", "L4"]) {
        const values = finite(pointRecords.filter((point) => point.level === level && active.has(point.id)).map((point) => point.value));
        const traceIndex = summaryTraceByLevel.get(level);
        if (traceIndex !== undefined) {
          await Plotly.restyle(boxGraph, {y: [values]}, [traceIndex]);
        }
        const row = Array.from(table.tBodies[0]?.rows || []).find((candidate) => candidate.cells[0]?.textContent.trim() === level);
        if (!row) continue;
        const mean = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
        const median = percentile(values, 0.5);
        const q1 = percentile(values, 0.25);
        const q3 = percentile(values, 0.75);
        const standardDeviation = values.length > 1
          ? Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1))
          : null;
        const coefficientOfVariation = standardDeviation !== null && Math.abs(mean) > 1e-15
          ? standardDeviation / Math.abs(mean) * 100 : null;
        [mean, median, standardDeviation, coefficientOfVariation, q1, q3, q1 === null ? null : q3 - q1]
          .forEach((value, index) => { row.cells[index + 1].textContent = format(value, index === 3); });
      }
      await Plotly.restyle(boxGraph, {visible: boxVisibility});
      const includedIds = [...new Set(pointRecords.filter((point) => active.has(point.id)).map((point) => point.id))].sort();
      if (participantSummary) {
        participantSummary.textContent = `Number of participants considered = ${includedIds.length} | IDs: ${includedIds.join(", ")}`;
      }
    };

    let updateQueued = false;
    const queueDependentUpdate = () => {
      if (updateQueued) return;
      updateQueued = true;
      requestAnimationFrame(() => {
        updateQueued = false;
        updateDependents().catch((error) => console.error("Linked convergence plot update failed", error));
      });
    };
    // Keep Plotly's native single-click and double-click legend behavior.
    // Synchronize only after Plotly has committed the raw plot visibility.
    rawGraph.on("plotly_restyle", queueDependentUpdate);
  });
});
</script>
"""

PLOT_IDENTITY_SCRIPT = r"""
<script>
document.addEventListener("DOMContentLoaded", () => {
  const identityToggle = document.querySelector("[data-participant-legend-toggle]");
  if (!identityToggle) return;

  let identitiesHidden = false;
  try { identitiesHidden = localStorage.getItem("ipw3-participant-identities-hidden") === "1"; } catch (_) {}
  const participantIdPattern = /\b(?:Participant\s+)?\d{3}(?:\.[A-Za-z0-9_-]+)?\b/g;
  const anonymizeText = (value) => String(value).replace(participantIdPattern, "Participant");

  const updateTextIdentity = (element) => {
    if (!element.dataset.identityVisibleText) element.dataset.identityVisibleText = element.textContent || "";
    element.textContent = identitiesHidden ? anonymizeText(element.dataset.identityVisibleText) : element.dataset.identityVisibleText;
  };

  const updatePlotIdentity = async (plotDocument) => {
    const plotly = plotDocument?.defaultView?.Plotly;
    if (!plotly) return;
    for (const graph of plotDocument.querySelectorAll(".plotly-graph-div")) {
      if (!graph._ipw3VisibleTraceText) {
        graph._ipw3VisibleTraceText = (graph.data || []).map((trace) => ({ name: trace.name, hovertemplate: trace.hovertemplate }));
      }
      const visible = graph._ipw3VisibleTraceText;
      const names = visible.map((trace) => identitiesHidden ? anonymizeText(trace.name || "") : trace.name);
      const hovertemplates = visible.map((trace) =>
        identitiesHidden && typeof trace.hovertemplate === "string" ? anonymizeText(trace.hovertemplate) : trace.hovertemplate
      );
      await plotly.relayout(graph, { showlegend: !identitiesHidden });
      await plotly.restyle(graph, { name: names, hovertemplate: hovertemplates });
    }
  };

  const updateAllIdentities = async () => {
    identityToggle.textContent = identitiesHidden ? "Show participant IDs and legends" : "Hide participant IDs and legends";
    identityToggle.setAttribute("aria-pressed", String(identitiesHidden));
    document.querySelectorAll(".slideshow-slide-header h2, .slide-sidebar-view a, .combined-grid-card h5, .combined-beta-card h4").forEach(updateTextIdentity);
    await updatePlotIdentity(document);
    for (const frame of document.querySelectorAll("iframe.plotly-lazy-frame")) {
      try { if (frame.contentDocument) await updatePlotIdentity(frame.contentDocument); } catch (_) {}
    }
  };

  document.addEventListener("load", (event) => {
    if (event.target instanceof HTMLIFrameElement && event.target.matches("iframe.plotly-lazy-frame")) {
      try { void updatePlotIdentity(event.target.contentDocument); } catch (_) {}
    }
  }, true);
  identityToggle.addEventListener("click", () => {
    identitiesHidden = !identitiesHidden;
    try { localStorage.setItem("ipw3-participant-identities-hidden", identitiesHidden ? "1" : "0"); } catch (_) {}
    void updateAllIdentities();
  });
  void updateAllIdentities();
});
</script>
"""

SLIDESHOW_SCRIPT = r"""
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


def participant_readme_path(participant_id: str) -> Path | None:
    """Return the README belonging to a participant ID, if one is present."""
    prefix = f"{normalize_participant_id(participant_id)}_"
    for participant_dir in sorted(ROOT_DIR.iterdir()):
        readme_path = participant_dir / "README.md"
        if participant_dir.is_dir() and participant_dir.name.startswith(prefix) and readme_path.is_file():
            return readme_path
    return None


def render_inline_markdown(text: str) -> str:
    """Render the small inline Markdown subset used by participant READMEs."""
    rendered = html.escape(text)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', rendered)
    return rendered


def render_readme_markdown(markdown_text: str) -> str:
    """Convert participant README Markdown to safe, readable modal HTML."""
    lines = markdown_text.splitlines()
    output: list[str] = []
    paragraph: list[str] = []
    list_type: str | None = None
    in_code_block = False
    code_lines: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            output.append(f"<p>{render_inline_markdown(' '.join(part.strip() for part in paragraph))}</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            output.append(f"</{list_type}>")
            list_type = None

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            close_list()
            if in_code_block:
                output.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines.clear()
                in_code_block = False
            else:
                in_code_block = True
            index += 1
            continue

        if in_code_block:
            code_lines.append(line)
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[index + 1]):
            flush_paragraph()
            close_list()
            table_lines = [line]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            rows = [[cell.strip().rstrip("\\") for cell in table_line.strip().strip("|").split("|")] for table_line in table_lines]
            header_row, *body_rows = rows
            output.append("<div class=\"readme-table-wrapper\"><table><thead><tr>")
            output.extend(f"<th>{render_inline_markdown(cell)}</th>" for cell in header_row)
            output.append("</tr></thead><tbody>")
            for row in body_rows:
                output.append("<tr>")
                output.extend(f"<td>{render_inline_markdown(cell)}</td>" for cell in row)
                output.append("</tr>")
            output.append("</tbody></table></div>")
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            close_list()
            level = min(len(heading.group(1)) + 1, 6)
            output.append(f"<h{level}>{render_inline_markdown(heading.group(2))}</h{level}>")
        elif re.match(r"^[-*]\s+", stripped):
            flush_paragraph()
            if list_type != "ul":
                close_list()
                output.append("<ul>")
                list_type = "ul"
            output.append(f"<li>{render_inline_markdown(re.sub(r'^[-*]\\s+', '', stripped))}</li>")
        elif re.match(r"^\d+[.)]\s+", stripped):
            flush_paragraph()
            if list_type != "ol":
                close_list()
                output.append("<ol>")
                list_type = "ol"
            output.append(f"<li>{render_inline_markdown(re.sub(r'^\\d+[.)]\\s+', '', stripped))}</li>")
        elif not stripped:
            flush_paragraph()
            close_list()
        else:
            close_list()
            paragraph.append(stripped)
        index += 1

    if in_code_block:
        output.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
    flush_paragraph()
    close_list()
    return "\n".join(output)


def build_participants_table(participants_metadata: list[dict[str, str]], participant_id: str | None = None) -> str:
    header_cells = ["Participant ID", "Organization", "Solver(s)", "Name(s)"]
    header_cells_with_color = header_cells + ["Color", "Information"]
    normalized_filter = normalize_participant_id(participant_id) if participant_id is not None else None
    rows = []

    for participant in participants_metadata:
        metadata_id = normalize_participant_id(participant.get("Participant ID", ""))
        if normalized_filter is not None and metadata_id != normalized_filter:
            continue

        color = participant_color(metadata_id)
        readme_path = participant_readme_path(metadata_id)

        row_html = "<tr>"

        for header in header_cells:
            row_html += f"<td>{html.escape(participant.get(header, ''))}</td>"

        row_html += f"""
            <td>
                <span class="participant-color-line" style="background-color: {html.escape(color)};"></span>
                <span>{html.escape(color)}</span>
            </td>
        """

        show_information = participant.get("Show Information on Index", True)
        if show_information and readme_path:
            dialog_id = f"participant-details-{metadata_id}"
            details_html = render_readme_markdown(readme_path.read_text(encoding="utf-8"))
            participant_label = f"{metadata_id} — {participant.get('Organization', 'Participant')}"
            row_html += f"""
            <td>
                <button class="participant-details-button" type="button"
                        data-participant-dialog-open="{dialog_id}"
                        aria-haspopup="dialog">
                    View details
                </button>
                <dialog class="participant-dialog" id="{dialog_id}" aria-labelledby="{dialog_id}-title">
                    <div class="participant-dialog-shell">
                        <header class="participant-dialog-header">
                            <div>
                                <span class="participant-dialog-eyebrow">Participant information</span>
                                <h2 id="{dialog_id}-title">{html.escape(participant_label)}</h2>
                            </div>
                            <button class="participant-dialog-close" type="button"
                                    data-participant-dialog-close aria-label="Close participant details">×</button>
                        </header>
                        <div class="participant-readme">
                            {details_html}
                        </div>
                    </div>
                </dialog>
            </td>
            """
        else:
            row_html += """
            <td>
                <button class="participant-details-button" type="button" disabled title="No participant details available">
                    No details
                </button>
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


def normal_grid_roughness_filter(case_id: str):
    if case_id == "TC_ONERAM6":
        return lambda key: key in {"1mm", "smooth"}
    return None


def build_grid_level_plots(participants, case_id: str, grid_level: str) -> str:
    roughness_filter = normal_grid_roughness_filter(case_id)
    html_output = ""
    if html_section_enabled("cutdata"):
        html_output += build_grid_level_cutdata_plots(participants, case_id, grid_level, roughness_filter_predicate=roughness_filter)
    if html_section_enabled("iceshape"):
        html_output += build_ice_shape_section(participants, case_id, grid_level, roughness_filter_predicate=roughness_filter)
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


def png_output_dir_for_participant(participant_id: str | None) -> Path:
    if participant_id is None:
        return ROOT_DIR / "PREVIEW_PNG"

    normalized_id = normalize_participant_id(participant_id)
    info = participant_info(normalized_id)
    organization = info.get("Organization", "") if info is not None else ""
    suffix = preview_folder_slug(organization or normalized_id)
    return ROOT_DIR / f"PREVIEW_PNG_{suffix}"


def latex_output_dir_for_participant(participant_id: str | None) -> Path:
    if participant_id is None:
        return ROOT_DIR / "PREVIEW_LATEX"

    normalized_id = normalize_participant_id(participant_id)
    info = participant_info(normalized_id)
    organization = info.get("Organization", "") if info is not None else ""
    suffix = preview_folder_slug(organization or normalized_id)
    return ROOT_DIR / f"PREVIEW_LATEX_{suffix}"


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


def geometry_page_path(geometry_id: str) -> Path:
    return PAGES_DIR / f"{slugify(geometry_id)}.html"


def cfd_convergence_page_path(geometry_id: str) -> Path:
    return PAGES_DIR / f"{slugify(geometry_id)}_cfd_grid_convergence.html"


def icing_convergence_page_path(case_id: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_icing_grid_convergence.html"


def category_page_path(category: str) -> Path:
    return PAGES_DIR / f"{slugify(category)}.html"


def optional_cfd_convergence_page_path(geometry_id: str) -> Path:
    return PAGES_DIR / f"{slugify(geometry_id)}_optional_cfd_grid_convergence.html"


def optional_icing_convergence_page_path(case_id: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_optional_icing_grid_convergence.html"


def water_mass_analysis_page_path(case_id: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_water_mass_analysis.html"


def per_bin_analysis_page_path(case_id: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_per_bin_analysis.html"


def upper_horn_angle_page_path(case_id: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_upper_horn_angle_analysis.html"


def comparison_with_3932_page_path() -> Path:
    return PAGES_DIR / "tc_naca0012_ae3933_comparison_with_3932.html"


def optional_grid_plots_page_path(grid_level: str) -> Path:
    return PAGES_DIR / f"onera_m6_{grid_level.lower()}_optional_roughness_plots.html"


def icing_metric_page_path(case_id: str, metric: str, optional: bool = False) -> Path:
    optional_slug = "optional_" if optional else ""
    return PAGES_DIR / f"{slugify(case_id)}_{optional_slug}icing_{slugify(metric)}_grid_convergence.html"


def geometry_for_case(case_id: str) -> str:
    return "NACA0012" if "NACA0012" in case_id.upper() else "ONERA M6"


def cases_by_geometry(case_ids: list[str]) -> dict[str, list[str]]:
    grouped = {"NACA0012": [], "ONERA M6": []}
    for case_id in case_ids:
        grouped.setdefault(geometry_for_case(case_id), []).append(case_id)
    return {geometry: cases for geometry, cases in grouped.items() if cases}


def shared_cfd_case_id(geometry_id: str, case_ids: list[str]) -> str:
    matching = cases_by_geometry(case_ids).get(geometry_id, [])
    if not matching:
        raise ValueError(f"No cases available for {geometry_id}")
    return matching[0]


def grid_page_path(case_id: str, grid_level: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_{grid_level.lower()}.html"


def cutdata_page_path(case_id: str, grid_level: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_{grid_level.lower()}_cutdata.html"


def ice_shape_page_path(case_id: str, grid_level: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_{grid_level.lower()}_ice_shape.html"


def single_ice_shape_page_path(case_id: str, grid_level: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_{grid_level.lower()}_single_layer_ice_shape.html"


def multi_ice_shape_page_path(case_id: str, grid_level: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_{grid_level.lower()}_multi_layer_ice_shape.html"


def combined_levels_cutdata_page_path(case_id: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_levels_combined_cutdata.html"


def combined_levels_ice_shape_page_path(case_id: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_levels_combined_ice_shape.html"


def combined_levels_single_ice_shape_page_path(case_id: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_levels_combined_single_layer_ice_shape.html"


def combined_levels_multi_ice_shape_page_path(case_id: str) -> Path:
    return PAGES_DIR / f"{slugify(case_id)}_levels_combined_multi_layer_ice_shape.html"


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


def build_page_navigation(case_ids: list[str], current_case_id: str | None = None, current_view: str | None = None, current_geometry: str | None = None, combined: bool = False) -> str:
    on_subpage = current_case_id is not None or current_geometry is not None or current_view is not None
    def page_href(path: Path) -> str:
        return link_from_pages(path) if on_subpage else link_from_root(path)

    def sidebar_link(label: str, path: Path, active: bool = False, fragment: str | None = None) -> str:
        href = page_href(path)
        if fragment:
            href += f"#{fragment}"
        return f'<a class="sidebar-link{" active" if active else ""}" href="{escape(href)}">{escape(label)}</a>'

    geometry_links = lambda path_builder, view: "".join(
        sidebar_link(geometry_id, path_builder(geometry_id), current_view == view and current_geometry == geometry_id)
        for geometry_id in cases_by_geometry(case_ids)
    )
    case_links = lambda path_builder, view: "".join(
        sidebar_link(display_case_name(case_id), path_builder(case_id), current_view == view and current_case_id == case_id)
        for case_id in case_ids
    )
    def icing_sidebar_links(optional: bool) -> str:
        links = ""
        view_prefix = "optional_icing" if optional else "icing"
        for case_id in case_ids:
            case_open = current_case_id == case_id and isinstance(current_view, str) and (
                current_view.startswith(view_prefix + "_")
                or current_view in {"water_mass_analysis", "per_bin_analysis", "upper_horn_angle_analysis", "comparison_with_3932"}
            )
            metric_links = "".join(
                sidebar_link(
                    label,
                    icing_metric_page_path(case_id, metric, optional=optional),
                    current_case_id == case_id and current_view == f"{view_prefix}_{metric}",
                )
                for metric, label in (("water", "Water mass"), ("ice", "Ice mass"), ("evaporation", "Water evaporation"))
            )
            if not optional and case_id == "TC_NACA0012_AE3933":
                metric_links += sidebar_link(
                    "Comparison with 3932",
                    comparison_with_3932_page_path(),
                    current_case_id == case_id and current_view == "comparison_with_3932",
                )
            if not optional:
                metric_links += sidebar_link(
                    "Per Bin Analysis",
                    per_bin_analysis_page_path(case_id),
                    current_case_id == case_id and current_view == "per_bin_analysis",
                )
                if "NACA0012" in case_id:
                    metric_links += sidebar_link(
                        "Upper Ice-Horn Angle",
                        upper_horn_angle_page_path(case_id),
                        current_case_id == case_id and current_view == "upper_horn_angle_analysis",
                    )
                metric_links += sidebar_link(
                    "Water Mass Analysis",
                    water_mass_analysis_page_path(case_id),
                    current_case_id == case_id and current_view == "water_mass_analysis",
                )
            links += f"""
            <details class="sidebar-case-group" {'open' if case_open else ''}>
              <summary>{escape(display_case_name(case_id))}</summary>
              <div class="sidebar-case-levels">{metric_links}</div>
            </details>
            """
        return links
    grid_links = ""
    for case_id in case_ids:
        levels = f"""
        <details class="sidebar-grid-group" {'open' if current_case_id == case_id and current_view in {'levels_combined_cutdata', 'levels_combined_ice_shape'} else ''}>
          <summary>Levels Combined</summary>
          {sidebar_link('Cut-Data', combined_levels_cutdata_page_path(case_id), current_case_id == case_id and current_view == 'levels_combined_cutdata') if html_section_enabled('cutdata') else ''}
          {sidebar_link('Single-layer Ice-Shape', combined_levels_single_ice_shape_page_path(case_id)) if html_section_enabled('iceshape') else ''}
          {sidebar_link('Multi-layer Ice-Shape', combined_levels_multi_ice_shape_page_path(case_id)) if html_section_enabled('iceshape') else ''}
        </details>
        """ if combined else ""
        for grid_level in sorted(VALID_GRID_LEVELS):
            level_is_open = current_case_id == case_id and current_view in {f"{grid_level}_cutdata", f"{grid_level}_ice_shape"}
            levels += f"""
            <details class="sidebar-grid-group" {'open' if level_is_open else ''}>
              <summary>{escape(grid_level)}</summary>
              {sidebar_link('Cut-Data', cutdata_page_path(case_id, grid_level), current_case_id == case_id and current_view == f'{grid_level}_cutdata') if html_section_enabled('cutdata') else ''}
              {sidebar_link('Single-layer Ice-Shape', single_ice_shape_page_path(case_id, grid_level)) if html_section_enabled('iceshape') else ''}
              {sidebar_link('Multi-layer Ice-Shape', multi_ice_shape_page_path(case_id, grid_level)) if html_section_enabled('iceshape') else ''}
            </details>
            """
        case_is_open = current_case_id == case_id and (
            current_view in {"levels_combined_cutdata", "levels_combined_ice_shape"}
            or (isinstance(current_view, str) and any(current_view.startswith(f"{level}_") for level in VALID_GRID_LEVELS))
        )
        grid_links += f"""
        <details class="sidebar-case-group" {'open' if case_is_open else ''}>
          <summary>{escape(display_case_name(case_id))}</summary>
          <div class="sidebar-case-levels">{levels}</div>
        </details>
        """

    groups = [
        ("CFD grid convergence", "cfd_grid_convergence", geometry_links(cfd_convergence_page_path, "cfd_grid_convergence")),
        ("Icing grid convergence", "icing_grid_convergence", icing_sidebar_links(False)),
        ("Grid-level plots", "grid_level_plots", grid_links),
    ]
    convergence_categories = {
        "cfd_grid_convergence", "icing_grid_convergence",
    }
    groups = [
        group for group in groups
        if (group[1] in convergence_categories and html_section_enabled("convergence"))
        or (group[1] == "grid_level_plots" and (html_section_enabled("cutdata") or html_section_enabled("iceshape")))
    ]
    group_html = ""
    for label, category, links in groups:
        is_grid_view = current_view in {"levels_combined_cutdata", "levels_combined_ice_shape"} or (
            isinstance(current_view, str) and any(current_view.startswith(f"{level}_") for level in VALID_GRID_LEVELS)
        )
        icing_child = isinstance(current_view, str) and (
            (category == "icing_grid_convergence" and current_view.startswith("icing_"))
        )
        is_open = current_view == category or icing_child or (category == "grid_level_plots" and is_grid_view)
        group_html += f"""
        <details class="sidebar-section" {'open' if is_open else ''}>
          <summary>{escape(label)}</summary>
          <a class="sidebar-overview" href="{escape(page_href(category_page_path(category)))}">Overview</a>
          <div class="sidebar-links">{links}</div>
        </details>
        """

    return f"""
    <button class="sidebar-show-button" type="button" data-site-sidebar-show hidden>Show navigation</button>
    <aside class="site-sidebar" aria-label="Page navigation" data-site-sidebar>
      <div class="site-sidebar-header">
        <strong>Results Navigation</strong>
        <button type="button" data-site-sidebar-hide aria-label="Hide navigation">Hide</button>
      </div>
      <a class="sidebar-link sidebar-index{" active" if not on_subpage else ""}" href="{'../index.html' if on_subpage else 'index.html'}">Index</a>
      {group_html}
    </aside>
    """


def build_submission_matrix_section(participants, case_ids: list[str]) -> str:
    """Summarize which participant/case/grid-level combinations were submitted."""
    grid_levels = sorted(VALID_GRID_LEVELS)
    case_tables = ""
    for case_id in case_ids:
        rows = ""
        for participant in participants:
            case_data = participant.cases.get(case_id)
            cells = ""
            for grid_level in grid_levels:
                grid_data = case_data.grid_levels.get(grid_level) if case_data is not None else None
                submitted = bool(
                    grid_data is not None
                    and (grid_data.datasets or grid_data.other_files)
                )
                cells += f'<td class="submission-status">{"x" if submitted else ""}</td>'
            rows += f'<tr><th scope="row">{escape(participant.participant_id)}</th>{cells}</tr>'
        case_tables += f"""
        <article class="submission-matrix-card">
          <h3>{escape(display_case_name(case_id))}</h3>
          <div class="readme-table-wrapper">
            <table class="participant-table submission-matrix">
              <thead><tr><th>Participant</th>{''.join(f'<th>{escape(level)}</th>' for level in grid_levels)}</tr></thead>
              <tbody>{rows}</tbody>
            </table>
          </div>
        </article>
        """
    return f"""
    <section class="participant-section submission-matrices-section">
      <h2>Data Submission by Case and Grid Level</h2>
      <p>An x indicates that at least one dataset or submitted file was found for that participant, case, and grid level.</p>
      <div class="submission-matrices-grid">{case_tables}</div>
    </section>
    """


def build_case_index_section(case_ids: list[str], participants=None) -> str:
    convergence_items = [
        ("CFD grid convergence", link_from_root(category_page_path("cfd_grid_convergence")), "Required (R) CFD data for NACA0012 or ONERA M6."),
        ("Icing grid convergence", link_from_root(category_page_path("icing_grid_convergence")), "Required (R) icing data for ONERA M6, AE3932, or AE3933."),
    ]
    grid_items = [
        ("Grid-level plots", link_from_root(category_page_path("grid_level_plots")), "L1–L4 CutData and ice-shape plots for each case."),
    ]
    items = (convergence_items if html_section_enabled("convergence") else []) + (
        grid_items if html_section_enabled("cutdata") or html_section_enabled("iceshape") else []
    )

    results_html = f"""
    <section class="participant-section">
      <h2>Results</h2>
      {build_link_list(items)}
    </section>
    """
    return results_html + (build_submission_matrix_section(participants, case_ids) if participants is not None else "")


def build_geometry_landing_content(geometry_id: str, case_ids: list[str]) -> str:
    geometry_cases = cases_by_geometry(case_ids)[geometry_id]
    items = [(
        "CFD grid convergence",
        link_from_pages(cfd_convergence_page_path(geometry_id)),
        "Shared aerodynamic grid-convergence plots for this model.",
    )]
    for case_id in geometry_cases:
        condition_label = case_id.replace("TC_NACA0012_", "") if geometry_id == "NACA0012" else "Icing condition"
        items.append((
            f"{condition_label} — Icing grid convergence",
            link_from_pages(icing_convergence_page_path(case_id)),
            "Required icing-mass convergence plots for this condition.",
        ))

    return f"""
    <section class="participant-section">
      <h2>{escape(geometry_id)}</h2>
      <p class="plot-description">Choose the shared CFD study or an icing condition.</p>
      {build_link_list(items)}
    </section>
    """


def display_case_name(case_id: str) -> str:
    if case_id == "TC_ONERAM6":
        return "ONERA M6"
    return case_id.replace("TC_NACA0012_", "NACA0012 ")


def build_category_landing_content(category: str, case_ids: list[str], combined: bool = False) -> str:
    if category in {"cfd_grid_convergence", "optional_cfd_grid_convergence"}:
        optional = category.startswith("optional_")
        items = []
        for geometry_id in cases_by_geometry(case_ids):
            path = optional_cfd_convergence_page_path(geometry_id) if optional else cfd_convergence_page_path(geometry_id)
            items.append((geometry_id, link_from_pages(path), f"{'Optional (O)' if optional else 'Required (R)'} CFD convergence plots."))
    elif category in {"icing_grid_convergence", "optional_icing_grid_convergence"}:
        optional = category.startswith("optional_")
        items = []
        for case_id in case_ids:
            requirement_label = "Optional (O)" if optional else "Required (R)"
            for metric, label in (("water", "Water mass"), ("ice", "Ice mass"), ("evaporation", "Water evaporation")):
                items.append((
                    f"{display_case_name(case_id)} — {label}",
                    link_from_pages(icing_metric_page_path(case_id, metric, optional=optional)),
                    f"{requirement_label} {label.lower()} convergence plots.",
                ))
            if not optional and case_id == "TC_NACA0012_AE3933":
                items.append((
                    "NACA0012 AE3933 — Comparison with 3932",
                    link_from_pages(comparison_with_3932_page_path()),
                    "AE3933 − AE3932 ice mass, water mass, and βmax differences.",
                ))
            # Keep one analysis link last in the required icing category. The
            # dedicated page itself contains required and optional ratios.
            if not optional:
                items.append((
                    f"{display_case_name(case_id)} — Per Bin Analysis",
                    link_from_pages(per_bin_analysis_page_path(case_id)),
                    "Per-bin L1–L4 convergence summaries from the optional 15-bin diameter-resolved submission.",
                ))
                if "NACA0012" in case_id:
                    items.append((
                        f"{display_case_name(case_id)} — Upper Ice-Horn Angle",
                        link_from_pages(upper_horn_angle_page_path(case_id)),
                        "Upper-horn angle convergence with MaxCCS, MeanCCS, and MinCCS references.",
                    ))
                items.append((
                    f"{display_case_name(case_id)} — Water Mass Analysis",
                    link_from_pages(water_mass_analysis_page_path(case_id)),
                    "Ice/water and (ice + evaporation)/water mass ratios only.",
                ))
    elif category == "grid_level_plots":
        items = []
        for case_id in case_ids:
            if combined:
                if html_section_enabled("cutdata"):
                    items.append((f"{display_case_name(case_id)} — Levels Combined Cut-Data", link_from_pages(combined_levels_cutdata_page_path(case_id)), "Participant matrices with L1–L4 CutData curves overlaid."))
                if html_section_enabled("iceshape"):
                    items.append((f"{display_case_name(case_id)} — Levels Combined Single-layer Ice-Shape", link_from_pages(combined_levels_single_ice_shape_page_path(case_id)), "Participant matrices with L1–L4 single-layer ice shapes overlaid."))
                    items.append((f"{display_case_name(case_id)} — Levels Combined Multi-layer Ice-Shape", link_from_pages(combined_levels_multi_ice_shape_page_path(case_id)), "Participant matrices with L1–L4 multi-layer ice shapes overlaid."))
            for grid_level in sorted(VALID_GRID_LEVELS):
                if html_section_enabled("cutdata"):
                    items.append((f"{display_case_name(case_id)} — {grid_level} Cut-Data", link_from_pages(cutdata_page_path(case_id, grid_level)), "Cut-Data comparison plots."))
                if html_section_enabled("iceshape"):
                    items.append((f"{display_case_name(case_id)} — {grid_level} Single-layer Ice-Shape", link_from_pages(single_ice_shape_page_path(case_id, grid_level)), "Single-layer ice-shape comparison plots."))
                    items.append((f"{display_case_name(case_id)} — {grid_level} Multi-layer Ice-Shape", link_from_pages(multi_ice_shape_page_path(case_id, grid_level)), "Multi-layer ice-shape comparison plots."))
    elif category == "optional_grid_plots":
        items = [
            (
                f"ONERA M6 — {grid_level}",
                link_from_pages(optional_grid_plots_page_path(grid_level)),
                "Optional roughness plots excluding the 1 mm baseline.",
            )
            for grid_level in sorted(VALID_GRID_LEVELS)
        ]
    else:
        raise ValueError(f"Unknown site category: {category}")

    title = category.replace("_", " ").title().replace("Cfd", "CFD")
    return f"""
    <section class="participant-section">
      <h2>{escape(title)}</h2>
      {build_link_list(items)}
    </section>
    """


def build_case_landing_content(case_id: str, combined: bool = False) -> str:
    expected_slices = CASE_SLICES.get(case_id, [])
    expected_slices_text = ", ".join(f"Y = {value:g} m" for value in expected_slices) if expected_slices else "not specified"
    items = []
    if html_section_enabled("convergence"):
        items.append(("Grid convergence", link_from_pages(convergence_page_path(case_id)), "Case-level convergence plots from gridConvergence data."))
    if combined and html_section_enabled("cutdata"):
        items.append(("Levels Combined — Cut-Data", link_from_pages(combined_levels_cutdata_page_path(case_id)), "Participant matrices with L1–L4 CutData curves overlaid."))
    if combined and html_section_enabled("iceshape"):
        items.append(("Levels Combined — Ice-Shape", link_from_pages(combined_levels_ice_shape_page_path(case_id)), "Participant matrices with L1–L4 ice shapes overlaid."))

    for grid_level in sorted(VALID_GRID_LEVELS):
        if html_section_enabled("cutdata"):
            items.append((f"{grid_level} Cut-Data", link_from_pages(cutdata_page_path(case_id, grid_level)), f"CutData plots for {grid_level}."))
        if html_section_enabled("iceshape"):
            items.append((f"{grid_level} Ice-Shape", link_from_pages(ice_shape_page_path(case_id, grid_level)), f"Ice-shape plots for {grid_level}."))

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


def build_grid_view_switch(case_id: str, grid_level: str, target: str) -> str:
    if target == "ice_shape" and not html_section_enabled("iceshape"):
        return ""
    if target == "cutdata" and not html_section_enabled("cutdata"):
        return ""
    if target == "ice_shape":
        href = link_from_pages(ice_shape_page_path(case_id, grid_level))
        label = f"View {grid_level} Ice-Shape plots"
        description = "Continue to the ice-shape results for this test case and grid level."
    else:
        href = link_from_pages(cutdata_page_path(case_id, grid_level))
        label = f"View {grid_level} Cut-Data plots"
        description = "Return to the Cut-Data results for this test case and grid level."
    return f"""
    <nav class="grid-view-switch" aria-label="Switch grid-level plot type">
      <span>{escape(description)}</span>
      <a href="{escape(href)}">{escape(label)} <span aria-hidden="true">→</span></a>
    </nav>
    """


def build_cutdata_page_content(participants, case_id: str, grid_level: str) -> str:
    return f"""
    <section class="page-filter-toolbar" aria-label="Page plot controls">
      <button type="button" data-page-filter-action="show">Show all plots</button>
      <button type="button" data-page-filter-action="hide">Hide all plots</button>
    </section>
    <section class="plot-subsection">
      <h3>{escape(display_case_name(case_id))} | {escape(grid_level)} | Cut-Data</h3>
      {build_grid_level_cutdata_plots(participants, case_id, grid_level, roughness_filter_predicate=normal_grid_roughness_filter(case_id))}
    </section>
    {build_grid_view_switch(case_id, grid_level, 'ice_shape')}
    """


def build_optional_grid_plots_page_content(participants, grid_level: str) -> str:
    optional_roughness_filter = lambda key: key not in {"1mm", "smooth"}
    return f"""
    <section class="page-filter-toolbar" aria-label="Page plot controls">
      <button type="button" data-page-filter-action="show">Show all plots</button>
      <button type="button" data-page-filter-action="hide">Hide all plots</button>
    </section>
    <section class="plot-subsection">
      <h2>ONERA M6 | {escape(grid_level)} | Optional roughness plots</h2>
      <p class="plot-description">Includes 0.5 mm, 1.5 mm, and variable-roughness submissions. The required 1 mm baseline and smooth curves are excluded.</p>
      {build_grid_level_cutdata_plots(
          participants,
          "TC_ONERAM6",
          grid_level,
          roughness_filter_predicate=optional_roughness_filter,
      ) if html_section_enabled("cutdata") else ""}
      {build_ice_shape_section(
          participants,
          "TC_ONERAM6",
          grid_level,
          roughness_filter_predicate=optional_roughness_filter,
      ) if html_section_enabled("iceshape") else ""}
    </section>
    """


def build_combined_levels_line_style_key() -> str:
    return """
    <aside class="combined-levels-key" aria-label="Grid-level line styles">
      <strong>Grid-level line styles</strong>
      <span><i class="level-line level-line-l1"></i>L1 — solid</span>
      <span><i class="level-line level-line-l2"></i>L2 — dash</span>
      <span><i class="level-line level-line-l3"></i>L3 — dot</span>
      <span><i class="level-line level-line-l4"></i>L4 — dash-dot</span>
    </aside>
    """


def build_combined_levels_cutdata_page_content(participants, case_id: str) -> str:
    return f"""
    {build_combined_levels_line_style_key()}
    <section class="page-filter-toolbar" aria-label="Page plot controls">
      <button type="button" data-page-filter-action="show">Show all plots</button>
      <button type="button" data-page-filter-action="hide">Hide all plots</button>
    </section>
    <section class="plot-subsection">
      <h2>{escape(display_case_name(case_id))} | Levels Combined | Cut-Data</h2>
      {build_combined_levels_cutdata_section(participants, case_id)}
    </section>
    """


def build_combined_levels_ice_shape_page_content(participants, case_id: str, shape_kind: str | None = None) -> str:
    shape_label = {"single": "Single-layer Ice-Shape", "multi": "Multi-layer Ice-Shape"}.get(shape_kind, "Ice-Shape")
    return f"""
    {build_combined_levels_line_style_key()}
    <section class="page-filter-toolbar" aria-label="Page plot controls">
      <button type="button" data-page-filter-action="show">Show all plots</button>
      <button type="button" data-page-filter-action="hide">Hide all plots</button>
    </section>
    <section class="plot-subsection">
      <h2>{escape(display_case_name(case_id))} | Levels Combined | {escape(shape_label)}</h2>
      {build_combined_levels_ice_shape_section(participants, case_id, shape_kind_filter=shape_kind)}
    </section>
    """


def build_ice_shape_page_content(participants, case_id: str, grid_level: str, shape_kind: str | None = None) -> str:
    shape_label = {"single": "Single-layer Ice-Shape", "multi": "Multi-layer Ice-Shape"}.get(shape_kind, "Ice-Shape")
    return f"""
    <section class="page-filter-toolbar" aria-label="Page plot controls">
      <button type="button" data-page-filter-action="show">Show all plots</button>
      <button type="button" data-page-filter-action="hide">Hide all plots</button>
    </section>
    <section class="plot-subsection">
      <h3>{escape(display_case_name(case_id))} | {escape(grid_level)} | {escape(shape_label)}</h3>
      {build_ice_shape_section(participants, case_id, grid_level, shape_kind_filter=shape_kind, roughness_filter_predicate=normal_grid_roughness_filter(case_id))}
    </section>
    {build_grid_view_switch(case_id, grid_level, 'cutdata')}
    """


def build_convergence_page_content(participants, case_id: str) -> str:
    return f"""
    <section class="plot-subsection">
      <h3>{escape(case_id)} | Grid convergence</h3>
      {build_grid_convergence_section(participants, case_id)}
    </section>
    """


def build_cfd_convergence_page_content(participants, case_id: str) -> str:
    return build_grid_convergence_section(participants, case_id, category="cfd")


def build_icing_convergence_page_content(participants, case_id: str) -> str:
    return build_grid_convergence_section(participants, case_id, category="icing", requirement="required")


def build_comparison_with_3932_page_content(participants) -> str:
    """Build the dedicated AE3933-versus-AE3932 comparison-only page."""
    return build_ae3933_ice_mass_comparison_section(participants, requirement="required")


def build_optional_cfd_convergence_page_content(participants, case_id: str) -> str:
    return build_grid_convergence_section(participants, case_id, category="cfd", requirement="optional")


def build_optional_icing_convergence_page_content(participants, case_id: str) -> str:
    return build_grid_convergence_section(participants, case_id, category="icing", requirement="optional")


def build_water_mass_analysis_page_content(participants, case_id: str) -> str:
    """Build the water-fate, beta-maximum, and impingement analysis page."""
    return (
        build_water_mass_analysis_section(participants, case_id, requirement="required")
        + build_beta_max_analysis_section(participants, case_id)
    )


def build_per_bin_analysis_page_content(participants, case_id: str) -> str:
    """Build optional diameter-resolved data into a dedicated per-bin summary."""
    return convergence_data_builder.build_per_bin_analysis_section(participants, case_id)


def build_upper_horn_angle_page_content(participants, case_id: str) -> str:
    """Build the dedicated upper-horn convergence page."""
    return convergence_data_builder.build_upper_horn_angle_convergence_section(participants, case_id)


def build_icing_metric_page_content(participants, case_id: str, metric: str, optional: bool = False) -> str:
    return build_grid_convergence_section(
        participants,
        case_id,
        category="icing",
        requirement="optional" if optional else "required",
        metric=metric,
    )


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
        {PARTICIPANT_DETAILS_SCRIPT}
        {SITE_SIDEBAR_SCRIPT}
        {PLOT_DOWNLOAD_SCRIPT}
        {LINKED_CONVERGENCE_LEGEND_SCRIPT}
        </body>
    </html>
    """


def build_index_html(participants_table_html: str, case_index_html: str) -> str:
    return build_page_html("IPW3 Post-Processing", participants_table_html + case_index_html)


VARIABLE_HELP = """\
--var values (repeat the option or use a comma-separated list):

  Grid convergence
    cl, cd, cmy
    water_mass, ice_mass, water_evap_mass
    ratio               both icing water-fate ratios
    ice_to_water_ratio_vs_n
    ice_evap_to_water_ratio_vs_n
    beta_max            beta maximum and impingement-width convergence
    upper_horn_angle    NACA0012/ONERA M6 upper ice-horn angle convergence
    impingement         alias for the same three water-analysis plots
    water_mass_by_diameter_vs_n
    ice_mass_by_diameter_vs_n
    water_evap_mass_by_diameter_vs_n
    qc_prime

  Cut data
    cp                  Cp vs X and Cp vs s
    cp_vs_x             Cp vs X only
    cp_vs_s             Cp vs s only
    htc                 heat-transfer coefficient
    beta                every collection-efficiency bin/card plot
    surface_temperature
    recovery_temperature
    freezing_fraction

  Ice shapes
    ice_shape

Useful aliases:
  water, ice, evaporation, water_evaporation, qc, q_c,
  collection_efficiency, temperature, trec, t_rec, iceshape, shape

Examples:
  --var cl
  --var cp --var beta
  --var cl,cd,cmy
  --var water_mass,ice_mass,water_evap_mass
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the IPW3 post-processing comparison website.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=VARIABLE_HELP,
    )
    parser.add_argument("--clean", action="store_true", help="Recompute cutData s mapping files instead of reusing existing *_sMap.dat files.")
    parser.add_argument("--cleanup", action="store_true", help="Remove generated *_sMap.dat and *_rotated.dat sidecars, then exit without rebuilding.")
    parser.add_argument("--clear", action="store_true", help="Remove the selected output folder before creating the new build.")
    parser.add_argument("--p", "--participant", dest="participant_id", help="Build a preview containing only one participant ID, for example --p 004.")
    parser.add_argument("--slides", action="store_true", help="Build index.html as a one-plot-per-slide presentation with sidebar and previous/next controls.")
    parser.add_argument("--png", action="store_true", help="Export every generated plot as a PNG instead of building HTML pages.")
    parser.add_argument("--latex", action="store_true", help="Export PNG plots and compile them into a LaTeX preview PDF.")
    parser.add_argument(
        "--naca0012-pres",
        action="store_true",
        help="Export the curated NACA0012 presentation figures into FIGURES category folders.",
    )
    parser.add_argument("--lower-res", action="store_true", help="Export PNGs at 1350x900 instead of the default 4050x2700. Used with --png or --latex.")
    parser.add_argument("--no-exp", action="store_true", help="Exclude experimental results from every generated plot.")
    parser.add_argument("--combined", action="store_true", help="Generate combined-level plots and pages.")
    parser.add_argument("--convergence", action="store_true", help="HTML only: build grid-convergence pages and plots.")
    parser.add_argument("--cutdata", action="store_true", help="HTML only: build Cut-Data pages and plots.")
    parser.add_argument("--iceshape", action="store_true", help="HTML only: build ice-shape pages and plots.")
    parser.add_argument("--include-horns", action="store_true", help="NACA0012 ice shapes: draw the clean-leading-edge to detected upper-horn construction line.")
    parser.add_argument(
        "--include-qc",
        action="store_true",
        help="Deprecated compatibility option; integrated convective heat-transfer plots are always included.",
    )
    parser.add_argument(
        "--var",
        dest="variables",
        action="append",
        help=(
            "Generate only the selected variable. May be repeated or comma-separated, "
            "for example --var cp or --var cp --var beta. See the list below."
        ),
    )
    return parser.parse_args()


def parse_variable_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    variables = {
        item.strip().lower().replace("-", "_")
        for value in values
        for item in value.split(",")
        if item.strip()
    }
    return variables or None


def write_png_plots(
    participants,
    case_ids: list[str],
    output_dir: Path,
    lower_res: bool = False,
    variable_filter: set[str] | None = None,
    include_qc: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cutdata_builder.clear_png_export_queue()
    iceshape_builder.clear_png_export_queue()
    convergence_data_builder.clear_png_export_queue()

    for case_id in case_ids:
        case_output_dir = output_dir / case_id
        cutdata_builder.set_png_export_dir(case_output_dir)
        iceshape_builder.set_png_export_dir(case_output_dir)
        convergence_data_builder.set_png_export_dir(case_output_dir)
        build_convergence_page_content(participants, case_id)
        if include_qc:
            build_icing_convergence_page_content(participants, case_id)
        for grid_level in sorted(VALID_GRID_LEVELS):
            build_grid_page_content(participants, case_id, grid_level)

    scale = 1 if lower_res else 3
    convergence_data_builder.flush_png_exports(scale=scale)
    cutdata_builder.flush_png_exports(scale=scale)
    iceshape_builder.flush_png_exports(scale=scale)


NACA0012_PRESENTATION_FIGURES: dict[str, tuple[str, str]] = {
    # AERODYNAMIC
    "AERODYNAMIC/tc_naca0012_ae3932_L1_cp_vs_s_slice_0p9144_all_roughness_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_cp_vs_s_slice_0p9144_all_roughness.png"),
    "AERODYNAMIC/tc_naca0012_ae3932_L1_cp_vs_x_slice_0p9144_all_roughness_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_cp_vs_x_slice_0p9144_all_roughness.png"),
    "AERODYNAMIC/tc_naca0012_ae3932_cd_vs_n_all_roughness_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_cd_vs_n_all_roughness.png"),
    "AERODYNAMIC/tc_naca0012_ae3932_cl_vs_n_all_roughness_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_cl_vs_n_all_roughness.png"),
    "AERODYNAMIC/tc_naca0012_ae3932_cmy_vs_n_all_roughness_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_cmy_vs_n_all_roughness.png"),
    # HTC
    "HTC/tc_naca0012_ae3932_L1_htc_vs_s_slice_0p9144_all_roughness_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_htc_vs_s_slice_0p9144_all_roughness.png"),
    "HTC/tc_naca0012_ae3932_L1_recovery_temperature_vs_s_slice_0p9144_all_roughness_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_recovery_temperature_vs_s_slice_0p9144_all_roughness.png"),
    "HTC/tc_naca0012_ae3932_qc_prime_vs_n_y_0.9144_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_qc_prime_vs_n_y_0.9144.png"),
    # ICE ACCRETION — AE3932
    "ICE_ACCRETION/tc_naca0012_ae3932_L1_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png"),
    "ICE_ACCRETION/tc_naca0012_ae3932_ice_evap_to_water_ratio_vs_n_required_bins15_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_ice_evap_to_water_ratio_vs_n_required_bins15.png"),
    "ICE_ACCRETION/tc_naca0012_ae3932_ice_mass_vs_n_unspecified_bins15_all_grid_levels_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_ice_mass_vs_n_unspecified_bins15_all_grid_levels.png"),
    "ICE_ACCRETION/tc_naca0012_ae3932_ice_mass_vs_n_unspecified_l1_vs_inverse_bins_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_ice_mass_vs_n_unspecified_l1_vs_inverse_bins.png"),
    "ICE_ACCRETION/tc_naca0012_ae3932_ice_to_water_ratio_vs_n_required_bins15_with_legend(1).png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_ice_to_water_ratio_vs_n_required_bins15.png"),
    "ICE_ACCRETION/tc_naca0012_ae3932_upper_horn_angle_bins15_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_upper_horn_angle_bins15.png"),
    "ICE_ACCRETION/tc_naca0012_ae3932_upper_horn_angle_distribution_convergence_l1_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_upper_horn_angle_distribution_convergence_l1.png"),
    # ICE ACCRETION — AE3933
    "ICE_ACCRETION/tc_naca0012_ae3933_L1_multilayer_ice_shape_slice_0p9144_bins01_roughness_unspecified_with_legend.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_multilayer_ice_shape_slice_0p9144_bins01_roughness_unspecified.png"),
    "ICE_ACCRETION/tc_naca0012_ae3933_L1_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified_with_legend.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_single_layer_ice_shape_slice_0p9144_bins15_roughness_unspecified.png"),
    "ICE_ACCRETION/tc_naca0012_ae3933_comparison_with_3932_ice_mass_bins15_difference_with_legend.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_comparison_with_3932_ice_mass_bins15_difference.png"),
    "ICE_ACCRETION/tc_naca0012_ae3933_ice_evap_to_water_ratio_vs_n_required_bins15_with_legend.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_ice_evap_to_water_ratio_vs_n_required_bins15.png"),
    "ICE_ACCRETION/tc_naca0012_ae3933_ice_mass_vs_n_unspecified_bins15_all_grid_levels_with_legend.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_ice_mass_vs_n_unspecified_bins15_all_grid_levels.png"),
    "ICE_ACCRETION/tc_naca0012_ae3933_ice_mass_vs_n_unspecified_l1_vs_inverse_bins_with_legend.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_ice_mass_vs_n_unspecified_l1_vs_inverse_bins.png"),
    "ICE_ACCRETION/tc_naca0012_ae3933_upper_horn_angle_bins15_with_legend.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_upper_horn_angle_bins15.png"),
    "ICE_ACCRETION/tc_naca0012_ae3933_upper_horn_angle_distribution_convergence_l1_with_legend.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_upper_horn_angle_distribution_convergence_l1.png"),
    # IMPINGEMENT
    "IMPINGEMENT/tc_naca0012_ae3932_L1_beta_bins15_vs_s_slice_0p9144_all_roughness_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_beta_bins15_vs_s_slice_0p9144_all_roughness.png"),
    "IMPINGEMENT/tc_naca0012_ae3932_beta_max_bins15_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_beta_max_bins15.png"),
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_vs_n_unspecified_bins15_all_grid_levels_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_vs_n_unspecified_bins15_all_grid_levels.png"),
    "IMPINGEMENT/tc_naca0012_ae3932_water_mass_vs_n_unspecified_l1_vs_inverse_bins_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_water_mass_vs_n_unspecified_l1_vs_inverse_bins.png"),
    "IMPINGEMENT/tc_naca0012_ae3932_width_bins15_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_width_bins15.png"),
    # SURFACE TEMPERATURE / FREEZING FRACTION
    "SURF_TEMP_FF/tc_naca0012_ae3932_L1_freezing_fraction_vs_s_slice_0p9144_all_roughness_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_freezing_fraction_vs_s_slice_0p9144_all_roughness.png"),
    "SURF_TEMP_FF/tc_naca0012_ae3932_L1_surface_temperature_vs_s_slice_0p9144_all_roughness_with_legend.png": ("TC_NACA0012_AE3932", "tc_naca0012_ae3932_L1_surface_temperature_vs_s_slice_0p9144_all_roughness.png"),
    "SURF_TEMP_FF/tc_naca0012_ae3933_L1_freezing_fraction_vs_s_slice_0p9144_all_roughness_with_legend.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_freezing_fraction_vs_s_slice_0p9144_all_roughness.png"),
    "SURF_TEMP_FF/tc_naca0012_ae3933_L1_surface_temperature_vs_s_slice_0p9144_all_roughness_with_legend.png": ("TC_NACA0012_AE3933", "tc_naca0012_ae3933_L1_surface_temperature_vs_s_slice_0p9144_all_roughness.png"),
}


def write_naca0012_presentation_figures(participants, output_dir: Path) -> None:
    """Generate and route the curated NACA0012 plot set for the presentation."""
    case_ids = ("TC_NACA0012_AE3932", "TC_NACA0012_AE3933")
    with tempfile.TemporaryDirectory(prefix="ipw3_naca0012_pres_") as staging_text:
        staging_dir = Path(staging_text)
        cutdata_builder.clear_png_export_queue()
        iceshape_builder.clear_png_export_queue()
        convergence_data_builder.clear_png_export_queue()

        for case_id in case_ids:
            case_dir = staging_dir / case_id
            cutdata_builder.set_png_export_dir(case_dir)
            iceshape_builder.set_png_export_dir(case_dir)
            convergence_data_builder.set_png_export_dir(case_dir)
            build_grid_convergence_section(participants, case_id, category="cfd")
            build_grid_convergence_section(participants, case_id, category="icing", requirement="required")
            build_water_mass_analysis_section(participants, case_id)
            build_beta_max_analysis_section(participants, case_id)
            convergence_data_builder.build_upper_horn_angle_convergence_section(participants, case_id)
            build_grid_page_content(participants, case_id, "L1")

        convergence_data_builder.set_png_export_dir(staging_dir / "TC_NACA0012_AE3933")
        build_ae3933_ice_mass_comparison_section(participants, requirement="required")

        # Apply one presentation-ready canvas and legend treatment across all
        # plot families before rasterization.
        roughness_legend_part = re.compile(
            r"(?:roughness|^no roughness$|^default roughness$|^variable roughness$|^unspecified roughness height$)",
            re.IGNORECASE,
        )
        for module in (convergence_data_builder, cutdata_builder, iceshape_builder):
            for figure, _ in module.PNG_EXPORT_QUEUE:
                for trace in figure.data:
                    if trace.name:
                        name_parts = [part.strip() for part in str(trace.name).split(" | ")]
                        filtered_parts = [part for part in name_parts if not roughness_legend_part.search(part)]
                        trace.name = " | ".join(filtered_parts) or name_parts[0]
                figure.update_layout(
                    width=2000,
                    height=700,
                    showlegend=True,
                    font={"family": "Arial, Helvetica, sans-serif", "size": 16},
                    legend={
                        "orientation": "h",
                        "x": 0.0,
                        "xanchor": "left",
                        "y": 1.02,
                        "yanchor": "bottom",
                        "font": {"size": 12},
                    },
                    margin={"l": 100, "r": 50, "t": 125, "b": 85},
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                )

        convergence_data_builder.flush_png_exports(scale=1, width=2000, height=700)
        cutdata_builder.flush_png_exports(scale=1, width=2000, height=700)
        iceshape_builder.flush_png_exports(scale=1, width=2000, height=700)

        missing: list[str] = []
        for destination_text, (case_id, source_name) in NACA0012_PRESENTATION_FIGURES.items():
            source = staging_dir / case_id / source_name
            destination = output_dir / destination_text
            if not source.exists():
                missing.append(f"{destination_text} <- {case_id}/{source_name}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        if missing:
            raise RuntimeError("Missing presentation plot exports:\n  " + "\n  ".join(missing))


def write_case_pages(participants, case_ids: list[str], combined: bool = False) -> None:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    category_titles = {
        "cfd_grid_convergence": "CFD grid convergence",
        "icing_grid_convergence": "Icing grid convergence",
        "grid_level_plots": "Grid-level plots",
    }
    if not html_section_enabled("convergence"):
        for category in (
            "cfd_grid_convergence", "icing_grid_convergence",
        ):
            category_titles.pop(category)
    if not (html_section_enabled("cutdata") or html_section_enabled("iceshape")):
        category_titles.pop("grid_level_plots")
    for category, title in category_titles.items():
        category_html = build_page_html(
            title=title,
            body_html=build_category_landing_content(category, case_ids, combined=combined),
            stylesheet_href="../style.css",
            back_href="../index.html",
            nav_html=build_page_navigation(case_ids, current_view=category, combined=combined),
        )
        category_page_path(category).write_text(category_html, encoding="utf-8")

    for geometry_id in cases_by_geometry(case_ids):
        if not html_section_enabled("convergence"):
            continue
        geometry_html = build_page_html(
            title=geometry_id,
            body_html=build_geometry_landing_content(geometry_id, case_ids),
            stylesheet_href="../style.css",
            back_href="../index.html",
            nav_html=build_page_navigation(case_ids, current_view="geometry", current_geometry=geometry_id, combined=combined),
        )
        geometry_page_path(geometry_id).write_text(geometry_html, encoding="utf-8")

        cfd_case_id = shared_cfd_case_id(geometry_id, case_ids)
        cfd_html = build_page_html(
            title=f"{geometry_id} | CFD grid convergence",
            body_html=build_cfd_convergence_page_content(participants, cfd_case_id),
            stylesheet_href="../style.css",
            back_href=link_from_pages(category_page_path("cfd_grid_convergence")),
            back_label="CFD grid convergence",
            nav_html=build_page_navigation(case_ids, current_view="cfd_grid_convergence", current_geometry=geometry_id, combined=combined),
        )
        cfd_convergence_page_path(geometry_id).write_text(cfd_html, encoding="utf-8")

    for case_id in case_ids:
        case_html = build_page_html(
            title=f"{case_id}",
            body_html=build_case_landing_content(case_id, combined=combined),
            stylesheet_href="../style.css",
            back_href="../index.html",
            nav_html=build_page_navigation(case_ids, current_case_id=case_id, current_view="case", combined=combined),
        )
        case_page_path(case_id).write_text(case_html, encoding="utf-8")

        if html_section_enabled("convergence"):
            convergence_html = build_page_html(
                title=f"{case_id} | Grid convergence",
                body_html=build_convergence_page_content(participants, case_id),
                stylesheet_href="../style.css",
                back_href=link_from_pages(case_page_path(case_id)),
                back_label=case_id,
                nav_html=build_page_navigation(case_ids, current_case_id=case_id, current_view="grid_convergence", combined=combined),
            )
            convergence_page_path(case_id).write_text(convergence_html, encoding="utf-8")

            geometry_id = geometry_for_case(case_id)
            icing_html = build_page_html(
                title=f"{case_id} | Icing grid convergence",
                body_html=build_icing_convergence_page_content(participants, case_id),
                stylesheet_href="../style.css",
                back_href=link_from_pages(category_page_path("icing_grid_convergence")),
                back_label="Icing grid convergence",
                nav_html=build_page_navigation(case_ids, current_case_id=case_id, current_view="icing_grid_convergence", current_geometry=geometry_id, combined=combined),
            )
            icing_convergence_page_path(case_id).write_text(icing_html, encoding="utf-8")

            for metric, metric_label in (("water", "Water mass"), ("ice", "Ice mass"), ("evaporation", "Water evaporation")):
                metric_html = build_page_html(
                    title=f"{display_case_name(case_id)} | {metric_label} grid convergence",
                    body_html=build_icing_metric_page_content(participants, case_id, metric, optional=False),
                    stylesheet_href="../style.css",
                    back_href=link_from_pages(category_page_path("icing_grid_convergence")),
                    back_label="Icing grid convergence",
                    nav_html=build_page_navigation(case_ids, current_case_id=case_id, current_view=f"icing_{metric}", current_geometry=geometry_id, combined=combined),
                )
                icing_metric_page_path(case_id, metric, optional=False).write_text(metric_html, encoding="utf-8")

            if case_id == "TC_NACA0012_AE3933":
                comparison_html = build_page_html(
                    title="NACA0012 AE3933 | Comparison with 3932",
                    body_html=build_comparison_with_3932_page_content(participants),
                    stylesheet_href="../style.css",
                    back_href=link_from_pages(category_page_path("icing_grid_convergence")),
                    back_label="Icing grid convergence",
                    nav_html=build_page_navigation(
                        case_ids,
                        current_case_id=case_id,
                        current_view="comparison_with_3932",
                        current_geometry=geometry_id,
                        combined=combined,
                    ),
                )
                comparison_with_3932_page_path().write_text(comparison_html, encoding="utf-8")

            # Write this last so it remains a separate final icing section and
            # contains only the two derived mass-ratio analyses.
            water_analysis_html = build_page_html(
                title=f"{display_case_name(case_id)} | Water Mass Analysis",
                body_html=build_water_mass_analysis_page_content(participants, case_id),
                stylesheet_href="../style.css",
                back_href=link_from_pages(category_page_path("icing_grid_convergence")),
                back_label="Icing grid convergence",
                nav_html=build_page_navigation(
                    case_ids,
                    current_case_id=case_id,
                    current_view="water_mass_analysis",
                    current_geometry=geometry_id,
                    combined=combined,
                ),
            )
            water_mass_analysis_page_path(case_id).write_text(water_analysis_html, encoding="utf-8")

            per_bin_html = build_page_html(
                title=f"{display_case_name(case_id)} | Per Bin Analysis",
                body_html=build_per_bin_analysis_page_content(participants, case_id),
                stylesheet_href="../style.css",
                back_href=link_from_pages(category_page_path("icing_grid_convergence")),
                back_label="Icing grid convergence",
                nav_html=build_page_navigation(
                    case_ids,
                    current_case_id=case_id,
                    current_view="per_bin_analysis",
                    current_geometry=geometry_id,
                    combined=combined,
                ),
            )
            per_bin_analysis_page_path(case_id).write_text(per_bin_html, encoding="utf-8")

            if "NACA0012" in case_id:
                horn_analysis_html = build_page_html(
                    title=f"{display_case_name(case_id)} | Upper Ice-Horn Angle",
                    body_html=build_upper_horn_angle_page_content(participants, case_id),
                    stylesheet_href="../style.css",
                    back_href=link_from_pages(category_page_path("icing_grid_convergence")),
                    back_label="Icing grid convergence",
                    nav_html=build_page_navigation(
                        case_ids,
                        current_case_id=case_id,
                        current_view="upper_horn_angle_analysis",
                        current_geometry=geometry_id,
                        combined=combined,
                    ),
                )
                upper_horn_angle_page_path(case_id).write_text(horn_analysis_html, encoding="utf-8")

        if combined and html_section_enabled("cutdata"):
            combined_cutdata_html = build_page_html(
                title=f"{display_case_name(case_id)} | Levels Combined | Cut-Data",
                body_html=build_combined_levels_cutdata_page_content(participants, case_id),
                stylesheet_href="../style.css",
                back_href=link_from_pages(category_page_path("grid_level_plots")),
                back_label="Grid-level plots",
                nav_html=build_page_navigation(case_ids, current_case_id=case_id, current_view="levels_combined_cutdata", combined=combined),
            )
            combined_levels_cutdata_page_path(case_id).write_text(combined_cutdata_html, encoding="utf-8")

        if combined and html_section_enabled("iceshape"):
            combined_ice_html = build_page_html(
                title=f"{display_case_name(case_id)} | Levels Combined | Ice-Shape",
                body_html=build_combined_levels_ice_shape_page_content(participants, case_id),
                stylesheet_href="../style.css",
                back_href=link_from_pages(category_page_path("grid_level_plots")),
                back_label="Grid-level plots",
                nav_html=build_page_navigation(case_ids, current_case_id=case_id, current_view="levels_combined_ice_shape", combined=combined),
            )
            combined_levels_ice_shape_page_path(case_id).write_text(combined_ice_html, encoding="utf-8")
            for shape_kind, shape_label, shape_path in (
                ("single", "Single-layer Ice-Shape", combined_levels_single_ice_shape_page_path(case_id)),
                ("multi", "Multi-layer Ice-Shape", combined_levels_multi_ice_shape_page_path(case_id)),
            ):
                shape_html = build_page_html(
                    title=f"{display_case_name(case_id)} | Levels Combined | {shape_label}",
                    body_html=build_combined_levels_ice_shape_page_content(participants, case_id, shape_kind=shape_kind),
                    stylesheet_href="../style.css",
                    back_href=link_from_pages(category_page_path("grid_level_plots")),
                    back_label="Grid-level plots",
                    nav_html=build_page_navigation(case_ids, current_case_id=case_id, current_view="levels_combined_ice_shape", combined=combined),
                )
                shape_path.write_text(shape_html, encoding="utf-8")

        for grid_level in sorted(VALID_GRID_LEVELS):
            if html_section_enabled("cutdata"):
                cutdata_html = build_page_html(
                    title=f"{display_case_name(case_id)} | {grid_level} | Cut-Data",
                    body_html=build_cutdata_page_content(participants, case_id, grid_level),
                    stylesheet_href="../style.css",
                    back_href=link_from_pages(category_page_path("grid_level_plots")),
                    back_label="Grid-level plots",
                    nav_html=build_page_navigation(case_ids, current_case_id=case_id, current_view=f"{grid_level}_cutdata", combined=combined),
                )
                cutdata_page_path(case_id, grid_level).write_text(cutdata_html, encoding="utf-8")

            if html_section_enabled("iceshape"):
                ice_shape_html = build_page_html(
                    title=f"{display_case_name(case_id)} | {grid_level} | Ice-Shape",
                    body_html=build_ice_shape_page_content(participants, case_id, grid_level),
                    stylesheet_href="../style.css",
                    back_href=link_from_pages(category_page_path("grid_level_plots")),
                    back_label="Grid-level plots",
                    nav_html=build_page_navigation(case_ids, current_case_id=case_id, current_view=f"{grid_level}_ice_shape", combined=combined),
                )
                ice_shape_page_path(case_id, grid_level).write_text(ice_shape_html, encoding="utf-8")
                for shape_kind, shape_label, shape_path in (
                    ("single", "Single-layer Ice-Shape", single_ice_shape_page_path(case_id, grid_level)),
                    ("multi", "Multi-layer Ice-Shape", multi_ice_shape_page_path(case_id, grid_level)),
                ):
                    shape_html = build_page_html(
                    title=f"{display_case_name(case_id)} | {grid_level} | {shape_label}",
                    body_html=build_ice_shape_page_content(participants, case_id, grid_level, shape_kind=shape_kind),
                    stylesheet_href="../style.css",
                    back_href=link_from_pages(category_page_path("grid_level_plots")),
                    back_label="Grid-level plots",
                    nav_html=build_page_navigation(case_ids, current_case_id=case_id, current_view=f"{grid_level}_ice_shape", combined=combined),
                )
                    shape_path.write_text(shape_html, encoding="utf-8")


def main() -> None:
    global HTML_BUILD_SECTIONS
    args = parse_args()
    if args.cleanup:
        removed = cleanup_generated_sidecars(ROOT_DIR, participant_id=args.participant_id)
        print(f"Removed {len(removed)} generated sidecar file(s).")
        return
    if args.naca0012_pres and any((args.png, args.latex, args.slides, args.convergence, args.cutdata, args.iceshape)):
        raise SystemExit("--naca0012-pres is a standalone export mode and cannot be combined with other output modes.")
    selected_html_sections = {
        section
        for section, selected in (
            ("convergence", args.convergence),
            ("cutdata", args.cutdata),
            ("iceshape", args.iceshape),
        )
        if selected
    }
    if selected_html_sections:
        if args.png or args.latex or args.slides:
            raise SystemExit("--convergence, --cutdata, and --iceshape are supported only for standard HTML builds.")
        HTML_BUILD_SECTIONS = selected_html_sections
    configure_output_paths(args.participant_id)
    if args.clear:
        if args.latex:
            clear_output_dir = latex_output_dir_for_participant(args.participant_id)
        elif args.png:
            clear_output_dir = png_output_dir_for_participant(args.participant_id)
        else:
            clear_output_dir = OUTPUT_DIR
        if clear_output_dir.exists():
            shutil.rmtree(clear_output_dir)
            print(f"Removed existing output folder: {clear_output_dir}")
    variable_filter = parse_variable_filter(args.variables)
    include_qc = True
    cutdata_builder.set_variable_filter(variable_filter)
    convergence_data_builder.set_variable_filter(variable_filter)
    convergence_data_builder.set_include_qc(include_qc)
    iceshape_builder.set_variable_filter(variable_filter)
    cutdata_builder.set_include_experimental_data(not args.no_exp)
    iceshape_builder.set_include_experimental_data(not args.no_exp)
    iceshape_builder.set_include_horn_overlays(args.include_horns)

    # Per-case highlight point coordinates are (X, Y, Z). Use None for a
    # coordinate that should be taken from each cutData slice, usually Y.
    highlight_points_by_case: HighlightPointsByCase = {
        "TC_NACA0012_AE3932": (0.0, None, 0.0),
        "TC_NACA0012_AE3933": (0.0, None, 0.0),
        "TC_ONERAM6": (0.0, None, 0.0),
    }
    
    participants = load_participants(ROOT_DIR, highlight_points_by_case=highlight_points_by_case, clean_s_cache=args.clean, participant_id=args.participant_id)
    convergence_data_builder.apply_participant_mass_conventions(participants)
    if args.participant_id is not None and not participants:
        requested_id = normalize_participant_id(args.participant_id)
        raise SystemExit(f"No participant folder found for ID {requested_id}.")

    case_ids = get_case_ids(participants)
    preview_name = preview_participant_name(args.participant_id) if args.participant_id is not None else PREVIEW_PARTICIPANT_NAME

    if args.naca0012_pres:
        if args.participant_id is not None:
            raise SystemExit("--naca0012-pres requires the all-participant dataset; do not combine it with --participant.")
        presentation_dir = ROOT_DIR / "FIGURES"
        write_naca0012_presentation_figures(participants, presentation_dir)
        print(f"Wrote {len(NACA0012_PRESENTATION_FIGURES)} NACA0012 presentation figures in {presentation_dir}")
        return

    if args.png or args.latex:
        latex_output_dir = latex_output_dir_for_participant(args.participant_id) if args.latex else None
        png_output_dir = (
            latex_output_dir / "PNG"
            if latex_output_dir is not None
            else png_output_dir_for_participant(args.participant_id)
        )
        write_png_plots(
            participants,
            case_ids,
            png_output_dir,
            lower_res=args.lower_res,
            variable_filter=variable_filter,
            include_qc=include_qc,
        )
        png_count = sum(1 for _ in png_output_dir.rglob("*.png"))
        print(f"Wrote {png_count} PNG plots in {png_output_dir}")
        if args.latex:
            assert latex_output_dir is not None
            pdf_path = build_latex_preview(
                template_path=ROOT_DIR / "TEMPLATE_LATEX" / "presentation.tex",
                png_dir=png_output_dir,
                output_dir=latex_output_dir,
                case_ids=case_ids,
            )
            print(f"Wrote LaTeX preview PDF to {pdf_path}")
        print(f"Participants found: {len(participants)}")
        print(f"Preview participant: {preview_name}")
        print(f"Cases included: {', '.join(case_ids)}")
        return

    participants_table_html = build_participants_table(PARTICIPANTS, participant_id=args.participant_id)
    case_index_html = build_case_index_section(case_ids, participants)

    # A filtered build replaces generated page files so excluded categories do
    # not remain as stale HTML from an earlier full build.
    if selected_html_sections and PAGES_DIR.exists():
        shutil.rmtree(PAGES_DIR)
    prepare_output_directory()
    page_title = "IPW3 Post-Processing" if args.participant_id is None else f"IPW3 Post-Processing | {preview_name}"
    if args.slides:
        index_html = build_page_html(page_title, build_slideshow_content(participants, case_ids))
    else:
        write_case_pages(participants, case_ids, combined=args.combined)
        index_html = build_page_html(
            page_title,
            participants_table_html + case_index_html,
            nav_html=build_page_navigation(case_ids, combined=args.combined),
        )
    OUTPUT_HTML.write_text(index_html, encoding="utf-8")

    print(f"Wrote {OUTPUT_HTML}")
    if not args.slides:
        print(f"Wrote pages in {PAGES_DIR}")
    print(f"Mode: {'slideshow' if args.slides else 'standard'}")
    print(f"HTML sections: {', '.join(sorted(HTML_BUILD_SECTIONS))}")
    print(f"Participants found: {len(participants)}")
    print(f"Preview participant: {preview_name}")
    print(f"Cases included: {', '.join(case_ids)}")


if __name__ == "__main__":
    main()
