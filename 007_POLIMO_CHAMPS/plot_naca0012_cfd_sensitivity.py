#!/usr/bin/env python3
"""Plot NACA0012 CFD coefficients from RESULTS*.dat sensitivity files.

Place this script beside the result files and run it with no arguments. The
baseline RESULTS_Initial.dat is always drawn with a dashed black line; every
other RESULTS*.dat case is drawn with a solid colored line.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COEFFICIENTS = (
    ("CL", r"$C_L$"),
    ("CD", r"$C_D$"),
    ("CMZ", r"$C_{M_z}$"),
)


def read_results(path: Path) -> dict[str, list[float]]:
    """Read the repeated LEVEL/header/value blocks in an IPW result file."""
    rows: list[tuple[int, float, float, float]] = []
    level: int | None = None

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        match = re.match(r"^LEVEL\s+(\d+)\b", line, flags=re.IGNORECASE)
        if match:
            level = int(match.group(1))
            continue
        if level is None or not line or "roughness height" in line.lower():
            continue

        fields = re.split(r"[\s,;]+", line)
        if len(fields) < 4:
            continue
        try:
            _, cl, cd, cmz = map(float, fields[:4])
        except ValueError:
            continue
        rows.append((level, cl, cd, cmz))
        level = None

    if not rows:
        raise ValueError(f"No LEVEL data blocks found in {path.name}")

    rows.sort(key=lambda row: row[0])
    return {
        "LEVEL": [row[0] for row in rows],
        "CL": [row[1] for row in rows],
        "CD": [row[2] for row in rows],
        "CMZ": [row[3] for row in rows],
    }


def case_label(path: Path) -> str:
    name = re.sub(r"^RESULTS[_\s-]*", "", path.stem, flags=re.IGNORECASE)
    return re.sub(r"[_-]+", " ", name).strip() or path.stem


def find_result_files(input_dir: Path) -> list[Path]:
    files = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".dat"
        and path.name.upper().startswith("RESULTS")
    ]
    return sorted(files, key=lambda path: (path.name.lower() != "results_initial.dat", path.name.lower()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "NACA0012_CFD_parameter_plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_result_files(input_dir)
    initial = next((path for path in files if path.name.lower() == "results_initial.dat"), None)
    if initial is None:
        raise FileNotFoundError(f"RESULTS_Initial.dat was not found in {input_dir}")

    datasets: list[tuple[Path, dict[str, list[float]]]] = []
    for path in files:
        try:
            datasets.append((path, read_results(path)))
        except ValueError as error:
            print(f"Skipping {path.name}: {error}")

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.get_cmap("tab10")

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    for axis, (key, ylabel) in zip(axes, COEFFICIENTS):
        color_index = 0
        for path, data in datasets:
            is_initial = path.name.lower() == "results_initial.dat"
            color = "black" if is_initial else colors(color_index)
            if not is_initial:
                color_index += 1
            axis.plot(
                data["LEVEL"],
                data[key],
                linestyle="--" if is_initial else "-",
                linewidth=2.2,
                marker="o",
                markersize=5,
                color=color,
                label=case_label(path),
            )
        axis.set_xlabel("Grid level (1 = finest, 4 = coarsest)")
        axis.set_ylabel(ylabel)
        axis.set_xticks([1, 2, 3, 4])
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.legend(frameon=True)

    figure.suptitle("NACA0012 CFD parameter sensitivity", fontsize=15, fontweight="bold")
    figure.savefig(output_dir / "NACA0012_CFD_coefficients.png", dpi=300)
    figure.savefig(output_dir / "NACA0012_CFD_coefficients.pdf")
    plt.close(figure)

    for key, ylabel in COEFFICIENTS:
        figure, axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
        color_index = 0
        for path, data in datasets:
            is_initial = path.name.lower() == "results_initial.dat"
            color = "black" if is_initial else colors(color_index)
            if not is_initial:
                color_index += 1
            axis.plot(
                data["LEVEL"], data[key],
                linestyle="--" if is_initial else "-", linewidth=2.3,
                marker="o", markersize=6, color=color, label=case_label(path),
            )
        axis.set_title(f"NACA0012 CFD: {ylabel} sensitivity")
        axis.set_xlabel("Grid level (1 = finest, 4 = coarsest)")
        axis.set_ylabel(ylabel)
        axis.set_xticks([1, 2, 3, 4])
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.legend(frameon=True)
        figure.savefig(output_dir / f"NACA0012_{key}.png", dpi=300)
        plt.close(figure)

    print(f"Plotted {len(datasets)} result file(s) in {output_dir}")


if __name__ == "__main__":
    main()
