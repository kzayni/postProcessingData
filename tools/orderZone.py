#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_dat(path: Path) -> tuple[list[str], np.ndarray]:
    header: list[str] = []
    rows: list[list[float]] = []

    for line in path.read_text().splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        parts = stripped.split()

        try:
            values = [float(x) for x in parts]
            rows.append(values)
        except ValueError:
            header.append(line.rstrip())

    if not rows:
        raise ValueError(f"No numeric rows found in {path}")

    return header, np.array(rows, dtype=float)


def remove_closing_duplicate(data: np.ndarray, tol: float = 1.0e-10) -> np.ndarray:
    if len(data) < 2:
        return data

    if np.linalg.norm(data[0, :3] - data[-1, :3]) < tol:
        return data[:-1]

    return data


def reorder_from_trailing_edge(data: np.ndarray) -> np.ndarray:
    x = data[:, 0]

    # If several points have almost the same max X, use the one with smallest |Z|
    # to avoid starting at the upper/lower TE corner when there is a blunt TE.
    x_max = np.max(x)
    candidates = np.where(np.abs(x - x_max) < 1.0e-8)[0]

    if len(candidates) > 1:
        z = data[candidates, 2]
        start = candidates[np.argmin(np.abs(z - np.mean(z)))]
    else:
        start = int(np.argmax(x))

    return np.vstack((data[start:], data[:start]))


def write_dat(path: Path, header: list[str], data: np.ndarray, close_loop: bool) -> None:
    if close_loop:
        data = np.vstack((data, data[0]))

    with path.open("w") as f:
        for line in header:
            f.write(line + "\n")

        for row in data:
            f.write(" ".join(f"{value:.5f}" for value in row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--no-close", action="store_true", help="Do not duplicate the first point at the end.")
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = args.input.with_name(args.input.stem + "_reordered.dat")

    header, data = read_dat(args.input)
    data = remove_closing_duplicate(data)
    data = reorder_from_trailing_edge(data)
    write_dat(output, header, data, close_loop=not args.no_close)

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()