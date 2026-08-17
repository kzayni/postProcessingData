"""Build the IPW3 PDF presentation from PNG plot exports."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


SLIDE_MARKER = "% IPW3_LATEX_SLIDES"
NACA_CASE_ORDER = ("TC_NACA0012_AE3932", "TC_NACA0012_AE3933")
ONERA_CASE_ORDER = ("TC_ONERAM6",)
GRID_LEVELS = ("L1", "L2", "L3", "L4")
CFD_PLOT_MARKERS = ("_cl_vs_n", "_cd_vs_n", "_cmy_vs_n")
GRID_LEVEL_PATTERN = re.compile(r"_L([1-4])_", re.IGNORECASE)
LATEX_AUXILIARY_SUFFIXES = (
    ".aux",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".nav",
    ".out",
    ".snm",
    ".toc",
    ".vrb",
)


def latex_escape(text: str) -> str:
    """Escape text used in LaTeX headings."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def plot_title(path: Path, case_id: str) -> str:
    """Turn a PNG filename into a compact human-readable frame title."""
    title = path.stem
    prefix = case_id.lower() + "_"
    if title.lower().startswith(prefix):
        title = title[len(prefix):]
    title = title.replace("0p", "0.").replace("_", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title.upper() if title in GRID_LEVELS else title.title()


def figure_frame(path: Path, case_id: str) -> str:
    image_path = path.resolve().as_posix()
    return "\n".join(
        (
            rf"\begin{{frame}}{{{latex_escape(plot_title(path, case_id))}}}",
            r"  \centering",
            rf"  \includegraphics[width=\textwidth,height=0.80\textheight,keepaspectratio]{{\detokenize{{{image_path}}}}}",
            r"\end{frame}",
        )
    )


def section_slide(title: str) -> str:
    escaped = latex_escape(title)
    return "\n".join(
        (
            rf"\section{{{escaped}}}",
            rf"\begin{{frame}}[plain]{{{escaped}}}",
            r"  \centering",
            rf"  \Huge {escaped}",
            r"\end{frame}",
        )
    )


def case_pngs(png_dir: Path, case_id: str) -> list[Path]:
    case_dir = png_dir / case_id
    return sorted(case_dir.glob("*.png"), key=lambda path: path.name.lower()) if case_dir.is_dir() else []


def is_cfd_plot(path: Path) -> bool:
    name = path.stem.lower()
    return GRID_LEVEL_PATTERN.search(path.stem) is None and any(marker in name for marker in CFD_PLOT_MARKERS)


def is_icing_convergence_plot(path: Path) -> bool:
    return GRID_LEVEL_PATTERN.search(path.stem) is None and not is_cfd_plot(path)


def append_frames(parts: list[str], paths: list[Path], case_id: str) -> None:
    parts.extend(figure_frame(path, case_id) for path in paths)


def build_slide_body(png_dir: Path, case_ids: list[str]) -> str:
    """Create slides in NACA CFD/icing/L1-L4 order, followed by ONERA M6."""
    available = set(case_ids)
    parts: list[str] = []

    naca_cases = [case_id for case_id in NACA_CASE_ORDER if case_id in available]
    if naca_cases:
        parts.append(section_slide("NACA0012 — Common CFD"))
        shared_case = next(
            (
                case_id for case_id in naca_cases
                if any(is_cfd_plot(path) for path in case_pngs(png_dir, case_id))
            ),
            naca_cases[0],
        )
        append_frames(
            parts,
            [path for path in case_pngs(png_dir, shared_case) if is_cfd_plot(path)],
            shared_case,
        )

        for case_id in naca_cases:
            condition = case_id.removeprefix("TC_NACA0012_")
            paths = case_pngs(png_dir, case_id)
            parts.append(section_slide(f"NACA0012 — {condition} Icing"))
            append_frames(parts, [path for path in paths if is_icing_convergence_plot(path)], case_id)
            for grid_level in GRID_LEVELS:
                level_paths = [
                    path for path in paths
                    if (match := GRID_LEVEL_PATTERN.search(path.stem)) and match.group(1) == grid_level[1:]
                ]
                if level_paths:
                    parts.append(rf"\subsection{{{grid_level}}}")
                    append_frames(parts, level_paths, case_id)

    for case_id in ONERA_CASE_ORDER:
        if case_id not in available:
            continue
        paths = case_pngs(png_dir, case_id)
        parts.append(section_slide("ONERA M6 — CFD"))
        append_frames(parts, [path for path in paths if is_cfd_plot(path)], case_id)
        parts.append(section_slide("ONERA M6 — Icing"))
        append_frames(parts, [path for path in paths if is_icing_convergence_plot(path)], case_id)
        for grid_level in GRID_LEVELS:
            level_paths = [
                path for path in paths
                if (match := GRID_LEVEL_PATTERN.search(path.stem)) and match.group(1) == grid_level[1:]
            ]
            if level_paths:
                parts.append(rf"\subsection{{{grid_level}}}")
                append_frames(parts, level_paths, case_id)

    return "\n\n".join(parts)


def build_latex_preview(
    template_path: Path,
    png_dir: Path,
    output_dir: Path,
    case_ids: list[str],
) -> Path:
    """Generate and compile a presentation, returning the output PDF path."""
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError("latexmk is required for --latex but was not found on PATH.")

    template = template_path.read_text(encoding="utf-8")
    if template.count(SLIDE_MARKER) != 1:
        raise ValueError(f"Expected exactly one {SLIDE_MARKER!r} marker in {template_path}.")

    slide_body = build_slide_body(png_dir, case_ids)
    if "\\includegraphics" not in slide_body:
        raise ValueError(f"No PNG plots were found in {png_dir} for: {', '.join(case_ids)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_tex = output_dir / "presentation.tex"
    generated_tex.write_text(template.replace(SLIDE_MARKER, slide_body), encoding="utf-8")

    # An interrupted pdflatex run can leave a truncated .aux/.nav file that
    # makes the next build fail before it reads the newly generated slides.
    for suffix in LATEX_AUXILIARY_SUFFIXES:
        auxiliary_path = output_dir / f"presentation{suffix}"
        if auxiliary_path.exists():
            auxiliary_path.unlink()

    command = [
        latexmk,
        "-pdf",
        "-silent",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-outdir={output_dir.resolve()}",
        str(generated_tex.resolve()),
    ]
    subprocess.run(command, cwd=template_path.parent, check=True)
    return output_dir / "presentation.pdf"
