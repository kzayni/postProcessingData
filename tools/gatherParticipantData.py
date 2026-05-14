from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import math
import re
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import pandas as pd


# =============================================================================
# IPW3 configuration
# =============================================================================

# Accepted test-case identifiers. These are searched in file names and dataset
# folder names to decide where each submitted file belongs.
VALID_CASES = {
    "TC_NACA0012_AE3932",
    "TC_NACA0012_AE3933",
    "TC_ONERAM6",
}

CASE_ALIASES = {
    "TC_ONERA_M6": "TC_ONERAM6",
}

# Expected cut locations for each case. These are not required to read the files,
# but they are stored in CaseData and can later be used for validation.
CASE_SLICES = {
    "TC_NACA0012_AE3932": [0.9144],
    "TC_NACA0012_AE3933": [0.9144],
    "TC_ONERAM6": [0.1, 0.75, 1.4],
}

# Expected grid levels. This is used to identify grid levels from file names and
# can later be used to check missing submissions.
VALID_GRID_LEVELS = {"L1", "L2", "L3", "L4"}

# Highlight point coordinates are (X, Y, Z). A None coordinate is replaced from
# the slice data; for Y this uses the slice mean.
HighlightPoint = tuple[Optional[float], Optional[float], Optional[float]]
HighlightPointsByCase = dict[str, HighlightPoint]
DEFAULT_CUTDATA_HIGHLIGHT_POINT: HighlightPoint = (0.0, None, 0.0)


# =============================================================================
# Small utilities
# =============================================================================

def log_step(message: str) -> None:
    """Print a simple, visible progress message."""
    print(f"\n[gatherParticipantData] {message}", flush=True)


# =============================================================================
# Tecplot data containers
# =============================================================================

@dataclass
class ZoneData:
    """
    One Tecplot zone.

    Example cutData zone name:
        SLICE_Y_0p9144m_BINS07_D01

    Example iceShape zone name:
        SLICE_Y_0p9144m_BINS07_D01

    Attributes
    ----------
    name:
        Tecplot zone name.
    data:
        Numerical zone data stored as a pandas DataFrame, with columns taken from
        the global VARIABLES line.
    auxdata:
        Zone-level AUXDATA fields, if present.
    """

    name: str
    data: pd.DataFrame
    auxdata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TecplotData:
    """
    Parsed content of one Tecplot ASCII .dat file.

    A file can contain one or more zones. Each zone is stored in the zones
    dictionary using the zone name as key.
    """

    path: Path
    title: Optional[str] = None
    variables: list[str] = field(default_factory=list)
    auxdata: dict[str, Any] = field(default_factory=dict)
    zones: dict[str, ZoneData] = field(default_factory=dict)

    def zone_names(self) -> list[str]:
        """Return all zone names in this Tecplot file."""
        return list(self.zones.keys())

    def get_zone(self, zone_name: str) -> ZoneData:
        """Return a zone by exact name."""
        return self.zones[zone_name]

    def find_zones(self, contains: str) -> list[ZoneData]:
        """Return all zones whose name contains a given substring."""
        contains_lower = contains.lower()
        return [zone for zone_name, zone in self.zones.items() if contains_lower in zone_name.lower()]

    def ipw3_zones(self) -> list[tuple[dict[str, str], ZoneData]]:
        """
        Return zones matching the IPW3 cutData/iceShape zone naming convention.

        Returns a list of (metadata, zone) pairs, where metadata contains:
            slice, bins, dataset
        """
        output: list[tuple[dict[str, str], ZoneData]] = []

        for zone_name, zone in self.zones.items():
            info = parse_ipw3_zone_name(zone_name)
            if info is not None:
                output.append((info, zone))

        return output

    def get_ipw3_zone(self, bins_id: str, dataset_id: str, slice_position: float | None = None, tolerance: float = 1.0e-6) -> ZoneData | None:
        """
        Find one IPW3 zone by bins, dataset, and optionally slice position.

        Example:
            get_ipw3_zone("BINS07", "D01", 0.9144)
        """
        bins_id = bins_id.upper()
        dataset_id = normalize_dataset_id(dataset_id)

        for info, zone in self.ipw3_zones():
            if info["bins"] != bins_id:
                continue

            if info["dataset"] != dataset_id:
                continue

            if slice_position is None:
                return zone

            zone_slice_position = decode_slice_position(info["slice"])
            if zone_slice_position is None:
                continue

            if abs(zone_slice_position - slice_position) <= tolerance:
                return zone

        return None


# =============================================================================
# Participant/case/grid/dataset hierarchy
# =============================================================================

@dataclass
class DatasetData:
    """
    One submitted dataset/attempt for one participant, one case, and one grid level.

    Desired hierarchy:
        Participant
          Test case
            Grid level
              Dataset / attempt

    Example:
        participant.cases["TC_NACA0012_AE3933"].grid_levels["L1"].datasets["D01"]
    """

    dataset_id: str
    path: Path
    cut_data_file: Optional[Path] = None
    ice_shape_file: Optional[Path] = None
    cut_data: Optional[TecplotData] = None
    ice_shape_data: Optional[TecplotData] = None
    other_files: list[Path] = field(default_factory=list)

    def read_files(self, case_id: str | None = None, highlight_points_by_case: HighlightPointsByCase | None = None, clean_s_cache: bool = False) -> None:
        """Read available cutData and iceShape files for this dataset/attempt."""
        if self.cut_data_file is not None:
            self.cut_data = read_tecplot_dat(self.cut_data_file, case_id=case_id, highlight_points_by_case=highlight_points_by_case, clean_s_cache=clean_s_cache)

        if self.ice_shape_file is not None:
            self.ice_shape_data = read_tecplot_dat(self.ice_shape_file)

    def get_cut_zone(self, bins_id: str, slice_position: float | None = None) -> ZoneData | None:
        """
        Convenience wrapper to access an IPW3 cutData zone for this dataset.

        The dataset_id is taken from this DatasetData object.
        """
        if self.cut_data is None:
            return None

        return self.cut_data.get_ipw3_zone(bins_id=bins_id, dataset_id=self.dataset_id, slice_position=slice_position)


@dataclass
class GridLevelData:
    """All datasets/attempts available for one grid level, for example L1."""

    grid_level: str
    datasets: dict[str, DatasetData] = field(default_factory=dict)
    other_files: list[Path] = field(default_factory=list)

    def get_or_create_dataset(self, dataset_id: str, path: Path) -> DatasetData:
        """Return an existing dataset/attempt or create it if missing."""
        dataset_id = normalize_dataset_id(dataset_id)

        if dataset_id not in self.datasets:
            self.datasets[dataset_id] = DatasetData(dataset_id=dataset_id, path=path)

        return self.datasets[dataset_id]

    def read_files(self, case_id: str | None = None, highlight_points_by_case: HighlightPointsByCase | None = None, clean_s_cache: bool = False) -> None:
        """Read all dataset/attempt files attached to this grid level."""
        for dataset_data in self.datasets.values():
            dataset_data.read_files(case_id=case_id, highlight_points_by_case=highlight_points_by_case, clean_s_cache=clean_s_cache)


@dataclass
class CaseData:
    """
    All files associated with one test case for one participant.

    Grid convergence is stored directly under the case because it compares grid
    levels and is therefore not attached to one specific grid level.
    """

    case_id: str
    expected_slices: list[float] = field(default_factory=list)
    grid_convergence_file: Optional[Path] = None
    grid_convergence_data: Optional[TecplotData] = None
    grid_levels: dict[str, GridLevelData] = field(default_factory=dict)
    other_files: list[Path] = field(default_factory=list)

    def get_or_create_grid_level(self, grid_level: str) -> GridLevelData:
        """Return an existing grid level or create it if missing."""
        grid_level = grid_level.upper()

        if grid_level not in self.grid_levels:
            self.grid_levels[grid_level] = GridLevelData(grid_level=grid_level)

        return self.grid_levels[grid_level]

    def read_files(self, highlight_points_by_case: HighlightPointsByCase | None = None, clean_s_cache: bool = False) -> None:
        """Read all files attached to this case."""
        if self.grid_convergence_file is not None:
            if self.grid_convergence_file.suffix.lower() == ".xlsx":
                self.grid_convergence_data = read_grid_convergence_xlsx(self.grid_convergence_file, self.case_id)
            else:
                self.grid_convergence_data = read_tecplot_dat(self.grid_convergence_file)

        for grid_level_data in self.grid_levels.values():
            grid_level_data.read_files(case_id=self.case_id, highlight_points_by_case=highlight_points_by_case, clean_s_cache=clean_s_cache)


@dataclass
class Participant:
    """
    One participant folder.

    Example folder:
        001_POLIMO_CHAMPS
    """

    participant_id: str
    organization: str
    solver: str
    path: Path
    name: str = ""
    cases: dict[str, CaseData] = field(default_factory=dict)
    other_files: list[Path] = field(default_factory=list)

    def get_or_create_case(self, case_id: str) -> CaseData:
        """Return an existing case or create it if missing."""
        case_id = canonical_case_id(case_id)

        if case_id not in self.cases:
            self.cases[case_id] = create_case_data(case_id)

        return self.cases[case_id]

    def read_files(self, highlight_points_by_case: HighlightPointsByCase | None = None, clean_s_cache: bool = False) -> None:
        """Read all submitted files for this participant."""
        for case_data in self.cases.values():
            case_data.read_files(highlight_points_by_case=highlight_points_by_case, clean_s_cache=clean_s_cache)

    def summary(self) -> None:
        """Print a human-readable summary of what was found and parsed."""
        print(f"Participant: {self.participant_id} | {self.organization} | {self.solver}")

        for case_id, case_data in sorted(self.cases.items()):
            print(f"  Case: {case_id}")
            print(f"    Expected slices: {case_data.expected_slices}")

            if case_data.grid_convergence_data is not None:
                print(f"    Grid convergence file: {case_data.grid_convergence_data.path.name}")
                print(f"    Grid convergence variables: {case_data.grid_convergence_data.variables}")
                print(f"    Grid convergence zones: {case_data.grid_convergence_data.zone_names()}")

            for grid_level, grid_data in sorted(case_data.grid_levels.items()):
                print(f"    Grid level: {grid_level}")

                for dataset_id, dataset_data in sorted(grid_data.datasets.items()):
                    print(f"      Dataset: {dataset_id}")

                    if dataset_data.cut_data is not None:
                        print(f"        cutData file: {dataset_data.cut_data.path.name}")
                        print(f"        cutData variables: {dataset_data.cut_data.variables}")
                        print("        cutData zones:")

                        for zone_name in dataset_data.cut_data.zone_names():
                            zone = dataset_data.cut_data.get_zone(zone_name)
                            info = parse_ipw3_zone_name(zone_name)

                            if info is not None:
                                slice_position = decode_slice_position(info["slice"])
                                print(f"          {zone_name} | slice={slice_position} | bins={info['bins']} | dataset={info['dataset']} | rows={len(zone.data)}")
                            else:
                                print(f"          {zone_name} | rows={len(zone.data)}")

                    if dataset_data.ice_shape_data is not None:
                        print(f"        iceShape file: {dataset_data.ice_shape_data.path.name}")
                        print(f"        iceShape variables: {dataset_data.ice_shape_data.variables}")
                        print(f"        iceShape zones: {dataset_data.ice_shape_data.zone_names()}")

                    if dataset_data.other_files:
                        print("        Other .dat files:")
                        for other_file in dataset_data.other_files:
                            print(f"          {other_file.name}")

            if case_data.other_files:
                print("    Other case files:")
                for other_file in case_data.other_files:
                    print(f"      {other_file.name}")

        if self.other_files:
            print("  Other participant files:")
            for other_file in self.other_files:
                print(f"    {other_file.name}")


def create_case_data(case_id: str) -> CaseData:
    """Create a CaseData object and attach its expected slice locations."""
    case_id = canonical_case_id(case_id)
    return CaseData(case_id=case_id, expected_slices=CASE_SLICES.get(case_id, []))


# =============================================================================
# Tecplot ASCII reader
# =============================================================================

def clean_line(line: str) -> str:
    """Strip whitespace from one line."""
    return line.strip()


def is_comment_or_empty(line: str) -> bool:
    """Return True for blank lines and comment lines starting with #."""
    stripped = line.strip()
    return stripped == "" or stripped.startswith("#")


def parse_quoted_value(line: str) -> Optional[str]:
    """Return the first value found inside double quotes."""
    match = re.search(r'"([^"]*)"', line)
    if match:
        return match.group(1)
    return None


def parse_variables_line(line: str) -> list[str]:
    """Extract Tecplot variable names from a VARIABLES line."""
    return re.findall(r'"([^"]+)"', line)


def parse_auxdata_line(line: str) -> tuple[str, str] | None:
    """Parse a Tecplot AUXDATA line into (key, value)."""
    match = re.match(r'\s*AUXDATA\s+([A-Za-z0-9_]+)\s*=\s*"([^"]*)"', line, re.IGNORECASE)
    if match is None:
        return None
    return match.group(1), match.group(2)


def parse_zone_name(line: str) -> str:
    """Extract the name from a Tecplot ZONE line."""
    name = parse_quoted_value(line)
    if name is not None:
        return name

    match = re.search(r'T\s*=\s*([^,\s]+)', line, re.IGNORECASE)
    if match:
        return match.group(1)

    return "UNKNOWN_ZONE"


def try_parse_numeric_row(line: str) -> Optional[list[float]]:
    """Try to parse a numerical data row. Return None if parsing fails."""
    line = line.replace(",", " ")
    parts = line.split()

    if not parts:
        return None

    values: list[float] = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            return None

    return values


def read_tecplot_dat(path: Path, case_id: str | None = None, highlight_points_by_case: HighlightPointsByCase | None = None, clean_s_cache: bool = False, process_cutdata: bool = True) -> TecplotData:
    """
    Read a Tecplot ASCII .dat file.

    Supported records:
        TITLE = "..."
        VARIABLES = "..." "..."
        ZONE T="..."
        AUXDATA KEY = "VALUE"
        numerical rows

    Each zone is stored as a pandas DataFrame.
    """
    result = TecplotData(path=path)

    current_zone_name: Optional[str] = None
    current_zone_rows: list[list[float]] = []
    current_zone_auxdata: dict[str, Any] = {}

    def flush_current_zone() -> None:
        """Convert the current zone rows into a DataFrame and store it."""
        nonlocal current_zone_name, current_zone_rows, current_zone_auxdata

        if current_zone_name is None:
            return

        n_variables = len(result.variables)
        valid_rows: list[list[float]] = []

        for row in current_zone_rows:
            if len(row) == n_variables:
                valid_rows.append(row)
            else:
                print(f"Warning: skipped row in {path.name}, zone {current_zone_name}: expected {n_variables} values, got {len(row)}")

        dataframe = pd.DataFrame(valid_rows, columns=result.variables)
        result.zones[current_zone_name] = ZoneData(name=current_zone_name, data=dataframe, auxdata=current_zone_auxdata.copy())

        current_zone_name = None
        current_zone_rows = []
        current_zone_auxdata = {}

    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = clean_line(raw_line)

            if is_comment_or_empty(line):
                continue

            if line.upper().startswith("TITLE"):
                result.title = parse_quoted_value(line)
                continue

            if line.upper().startswith("VARIABLES"):
                result.variables = parse_variables_line(line)
                continue

            if line.upper().startswith("AUXDATA"):
                parsed_auxdata = parse_auxdata_line(line)

                if parsed_auxdata is not None:
                    key, value = parsed_auxdata

                    if current_zone_name is None:
                        result.auxdata[key] = value
                    else:
                        current_zone_auxdata[key] = value

                continue

            if line.upper().startswith("ZONE"):
                flush_current_zone()
                current_zone_name = parse_zone_name(line)
                current_zone_rows = []
                current_zone_auxdata = {}
                continue

            numeric_row = try_parse_numeric_row(line)
            if numeric_row is not None and current_zone_name is not None:
                current_zone_rows.append(numeric_row)

    flush_current_zone()

    if process_cutdata and identify_file_type(path.name) == "cutData":
        add_curvilinear_distance_to_cutdata(result, case_id=case_id, highlight_points_by_case=highlight_points_by_case, clean_s_cache=clean_s_cache)

    return result


def find_column_case_insensitive(columns, candidates: list[str]) -> str | None:
    """Return the first column matching any candidate, case-insensitively."""
    lookup = {column.lower(): column for column in columns}

    for candidate in candidates:
        column = lookup.get(candidate.lower())
        if column is not None:
            return column

    return None


def point_distance_squared(point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> float:
    """Squared Euclidean distance between two 3D points."""
    return sum((point_a[index] - point_b[index]) ** 2 for index in range(3))


def order_surface_points(dataframe: pd.DataFrame, x_column: str, y_column: str, z_column: str) -> pd.DataFrame:
    """
    Order cutData rows around the local slice surface.

    Sorting by angle in the dominant slice plane gives a stable airfoil/wrap
    order while keeping every non-coordinate variable attached to its row.
    """
    if len(dataframe) <= 2:
        return dataframe.reset_index(drop=True)

    x_values = dataframe[x_column].astype(float).tolist()
    y_values = dataframe[y_column].astype(float).tolist()
    z_values = dataframe[z_column].astype(float).tolist()
    spans = {
        "x": max(x_values) - min(x_values),
        "y": max(y_values) - min(y_values),
        "z": max(z_values) - min(z_values),
    }
    plane_axes = sorted(spans, key=spans.get, reverse=True)[:2]
    axis_columns = {"x": x_column, "y": y_column, "z": z_column}
    first_axis = axis_columns[plane_axes[0]]
    second_axis = axis_columns[plane_axes[1]]
    first_values = dataframe[first_axis].astype(float)
    second_values = dataframe[second_axis].astype(float)
    first_center = first_values.mean()
    second_center = second_values.mean()

    ordered_indices = sorted(
        range(len(dataframe)),
        key=lambda index: math.atan2(second_values.iloc[index] - second_center, first_values.iloc[index] - first_center),
    )

    return dataframe.iloc[ordered_indices].reset_index(drop=True)


def rotate_surface_order_from_farthest_point(dataframe: pd.DataFrame, x_column: str, y_column: str, z_column: str, highlight: tuple[float, float, float]) -> pd.DataFrame:
    """Rotate an already ordered surface so the highlight lies inside the path."""
    if len(dataframe) <= 2:
        return dataframe.reset_index(drop=True)

    points = list(
        zip(
            dataframe[x_column].astype(float),
            dataframe[y_column].astype(float),
            dataframe[z_column].astype(float),
        )
    )
    start_index = max(range(len(points)), key=lambda index: point_distance_squared(points[index], highlight))

    if start_index == 0:
        return dataframe.reset_index(drop=True)

    rotated = pd.concat([dataframe.iloc[start_index:], dataframe.iloc[:start_index]], ignore_index=True)
    return rotated.reset_index(drop=True)


def curvilinear_mapping_path(cutdata_path: Path) -> Path:
    """Return the sidecar path used to cache cutData s mappings."""
    return cutdata_path.with_name(f"{cutdata_path.stem}_sMap.dat")


def resolve_highlight_point(case_id: str | None, path: Path, dataframe: pd.DataFrame, x_column: str, y_column: str, z_column: str, highlight_points_by_case: HighlightPointsByCase | None) -> tuple[float, float, float]:
    """Return the case highlight point as X, Y, Z, filling slice-dependent coordinates."""
    resolved_case_id = case_id or extract_case_id_from_name(path.name)
    highlight_points_by_case = highlight_points_by_case or {}
    configured_point = highlight_points_by_case.get(resolved_case_id or "", DEFAULT_CUTDATA_HIGHLIGHT_POINT)
    column_values = {
        0: dataframe[x_column].astype(float),
        1: dataframe[y_column].astype(float),
        2: dataframe[z_column].astype(float),
    }
    resolved = []

    for index, value in enumerate(configured_point):
        if value is None:
            resolved.append(float(column_values[index].mean()))
        else:
            resolved.append(float(value))

    return resolved[0], resolved[1], resolved[2]


def cumulative_distances(points: list[tuple[float, float, float]]) -> list[float]:
    """Return cumulative polyline distances for ordered points."""
    output = [0.0]

    for index in range(1, len(points)):
        segment_length = math.sqrt(point_distance_squared(points[index - 1], points[index]))
        output.append(output[-1] + segment_length)

    return output


def projected_distance_on_polyline(points: list[tuple[float, float, float]], distances: list[float], highlight: tuple[float, float, float]) -> float:
    """Project a highlight point onto a polyline and return its curvilinear coordinate."""
    if not points:
        return 0.0

    if len(points) == 1:
        return distances[0]

    best_distance_squared = float("inf")
    best_curvilinear_distance = distances[0]

    for index in range(len(points) - 1):
        start = points[index]
        end = points[index + 1]
        segment = tuple(end[axis] - start[axis] for axis in range(3))
        segment_length_squared = sum(value * value for value in segment)

        if segment_length_squared == 0.0:
            projection_fraction = 0.0
        else:
            projection_fraction = sum((highlight[axis] - start[axis]) * segment[axis] for axis in range(3)) / segment_length_squared
            projection_fraction = max(0.0, min(1.0, projection_fraction))

        projection = tuple(start[axis] + projection_fraction * segment[axis] for axis in range(3))
        distance_squared = point_distance_squared(highlight, projection)

        if distance_squared < best_distance_squared:
            segment_length = math.sqrt(segment_length_squared)
            best_distance_squared = distance_squared
            best_curvilinear_distance = distances[index] + projection_fraction * segment_length

    return best_curvilinear_distance


def add_curvilinear_distance_to_zone(path: Path, zone: ZoneData, case_id: str | None = None, highlight_points_by_case: HighlightPointsByCase | None = None) -> list[tuple[float, int]]:
    """Order one cutData zone and add/update its s coordinate."""
    x_column = find_column_case_insensitive(zone.data.columns, ["X", "CoordinateX"])
    y_column = find_column_case_insensitive(zone.data.columns, ["Y", "CoordinateY"])
    z_column = find_column_case_insensitive(zone.data.columns, ["Z", "CoordinateZ"])

    if x_column is None or y_column is None or z_column is None or zone.data.empty:
        return []

    working_data = zone.data.copy()
    working_data["__variable_index__"] = list(range(len(working_data)))
    ordered_data = order_surface_points(working_data, x_column, y_column, z_column)
    highlight = resolve_highlight_point(case_id, path, ordered_data, x_column, y_column, z_column, highlight_points_by_case)
    ordered_data = rotate_surface_order_from_farthest_point(ordered_data, x_column, y_column, z_column, highlight)
    points = list(
        zip(
            ordered_data[x_column].astype(float),
            ordered_data[y_column].astype(float),
            ordered_data[z_column].astype(float),
        )
    )
    distances = cumulative_distances(points)
    highlight_distance = projected_distance_on_polyline(points, distances, highlight)

    s_values = [distance - highlight_distance for distance in distances]
    ordered_data["s"] = s_values
    variable_indices = [int(value) for value in ordered_data["__variable_index__"]]
    zone.data = ordered_data.drop(columns=["__variable_index__"])
    zone.auxdata["HIGHLIGHT_X"] = f"{highlight[0]:.12g}"
    zone.auxdata["HIGHLIGHT_Y"] = f"{highlight[1]:.12g}"
    zone.auxdata["HIGHLIGHT_Z"] = f"{highlight[2]:.12g}"
    return list(zip(s_values, variable_indices))


def apply_curvilinear_mapping_to_cutdata(data: TecplotData, mapping_data: TecplotData) -> bool:
    """Apply a cached s-to-original-row mapping to cutData zones."""
    for zone_name, zone in data.zones.items():
        mapping_zone = mapping_data.zones.get(zone_name)
        if mapping_zone is None:
            return False

        s_column = find_column_case_insensitive(mapping_zone.data.columns, ["s", "S"])
        index_column = find_column_case_insensitive(mapping_zone.data.columns, ["variable Index", "Variable Index", "VARIABLE_INDEX"])
        if s_column is None or index_column is None:
            return False

        if len(mapping_zone.data) != len(zone.data):
            return False

        indices = [int(round(value)) for value in mapping_zone.data[index_column].astype(float)]
        if not indices or min(indices) < 0 or max(indices) >= len(zone.data):
            return False

        ordered_data = zone.data.iloc[indices].reset_index(drop=True).copy()
        ordered_data["s"] = mapping_zone.data[s_column].astype(float).tolist()
        zone.data = ordered_data

    if "s" not in data.variables:
        data.variables.append("s")

    return True


def write_curvilinear_mapping_file(path: Path, data: TecplotData, mappings: dict[str, list[tuple[float, int]]]) -> None:
    """Write the sidecar s mapping file in a Tecplot-like format."""
    lines = [
        'TITLE = "cutData curvilinear distance mapping"',
        'VARIABLES = "s" "variable Index"',
        "",
    ]

    for zone_name in data.zones:
        rows = mappings.get(zone_name, [])
        lines.append(f'ZONE T="{zone_name}"')
        for s_value, variable_index in rows:
            lines.append(f"{s_value:.12g} {variable_index:d}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def add_curvilinear_distance_to_cutdata(data: TecplotData, case_id: str | None = None, highlight_points_by_case: HighlightPointsByCase | None = None, clean_s_cache: bool = False) -> None:
    """Add computed curvilinear distance to every zone in a cutData file."""
    mapping_path = curvilinear_mapping_path(data.path)

    if not clean_s_cache and mapping_path.exists():
        mapping_data = read_tecplot_dat(mapping_path, process_cutdata=False)
        if apply_curvilinear_mapping_to_cutdata(data, mapping_data):
            return

        print(f"Warning: ignored stale s mapping file: {mapping_path}")

    mappings: dict[str, list[tuple[float, int]]] = {}
    for zone_name, zone in data.zones.items():
        mappings[zone_name] = add_curvilinear_distance_to_zone(data.path, zone, case_id=case_id, highlight_points_by_case=highlight_points_by_case)

    if "s" not in data.variables:
        data.variables.append("s")

    write_curvilinear_mapping_file(mapping_path, data, mappings)


# =============================================================================
# Grid-convergence XLSX reader
# =============================================================================

XLSX_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def excel_column_to_index(cell_ref: str) -> int:
    """Convert an Excel cell reference such as B12 to a zero-based column index."""
    letters = "".join(character for character in cell_ref if character.isalpha())
    index = 0

    for letter in letters:
        index = index * 26 + ord(letter.upper()) - ord("A") + 1

    return index - 1


def read_xlsx_sheets(path: Path) -> dict[str, list[list[str]]]:
    """Read basic cell values from an .xlsx workbook without requiring openpyxl."""
    sheets: dict[str, list[list[str]]] = {}

    with ZipFile(path) as archive:
        shared_strings: list[str] = []

        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for string_item in shared_root.findall("m:si", XLSX_NS):
                text_parts = [text_node.text or "" for text_node in string_item.findall(".//m:t", XLSX_NS)]
                shared_strings.append("".join(text_parts))

        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_root}

        for sheet in workbook_root.find("m:sheets", XLSX_NS):
            sheet_name = sheet.attrib["name"]
            rel_id = sheet.attrib[f"{{{XLSX_NS['r']}}}id"]
            target = rel_targets[rel_id]
            sheet_path = "xl/" + target.lstrip("/")
            sheet_root = ET.fromstring(archive.read(sheet_path))
            rows: list[list[str]] = []

            for row_node in sheet_root.findall(".//m:sheetData/m:row", XLSX_NS):
                row_values: list[str] = []

                for cell_node in row_node.findall("m:c", XLSX_NS):
                    column_index = excel_column_to_index(cell_node.attrib["r"])
                    while len(row_values) < column_index:
                        row_values.append("")

                    value_node = cell_node.find("m:v", XLSX_NS)
                    value = "" if value_node is None else value_node.text or ""

                    if cell_node.attrib.get("t") == "s" and value != "":
                        value = shared_strings[int(value)]
                    elif cell_node.attrib.get("t") == "inlineStr":
                        value = "".join(text.text or "" for text in cell_node.findall(".//m:t", XLSX_NS))

                    row_values.append(value)

                rows.append(row_values)

            sheets[sheet_name] = rows

    return sheets


def normalize_xlsx_case_ids(sheet_name: str) -> list[str]:
    """Map workbook sheet names to internal test-case IDs."""
    sheet_name_upper = sheet_name.upper().replace(" ", "")

    if "ONERAM6" in sheet_name_upper or "ONERA-M6" in sheet_name_upper:
        return ["TC_ONERAM6"]

    if "NACA0012" in sheet_name_upper:
        return ["TC_NACA0012_AE3932", "TC_NACA0012_AE3933"]

    return []


def detect_xlsx_grid_convergence_cases(path: Path) -> set[str]:
    """Return case IDs represented by grid-convergence workbook sheets."""
    case_ids: set[str] = set()

    for sheet_name in read_xlsx_sheets(path):
        case_ids.update(normalize_xlsx_case_ids(sheet_name))

    return case_ids


def parse_level_number(value: Any) -> int | None:
    """Parse labels such as Level 4 into integer level numbers."""
    match = re.search(r"\bLevel\s+(\d+)\b", str(value), re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def to_float_or_none(value: Any) -> float | None:
    """Convert spreadsheet cell content to float when possible."""
    text = str(value).strip()
    if text == "":
        return None

    try:
        return float(text)
    except ValueError:
        return None


def normalize_grid_convergence_column(header: str) -> str:
    """Normalize workbook headers to the columns expected by the plot builders."""
    header_upper = header.strip().upper()

    if header_upper == "CL":
        return "CL"
    if header_upper == "CD":
        return "CD"
    if header_upper.startswith("CMX"):
        return "CMX"
    if header_upper.startswith("CMY"):
        return "CMY"
    if header_upper.startswith("CMZ"):
        return "CMZ"
    if "WATER IMPINGED" in header_upper:
        return "WATER_MASS"
    if "ICE ACCRETED" in header_upper:
        return "ICE_MASS"
    if "WATER EVAPORATED" in header_upper:
        return "WATER_EVAP_MASS"
    if "DIAMETER" in header_upper:
        return "DIAMETER"
    if header_upper == "BIN":
        return "BIN"
    if "ROUGHNESS" in header_upper:
        return "ROUGHNESS_HEIGHT"

    return re.sub(r"[^A-Z0-9]+", "_", header_upper).strip("_")


def add_grid_convergence_zone(result: TecplotData, zone_name: str, rows: list[dict[str, Any]]) -> None:
    """Store one parsed workbook table as a Tecplot-like zone."""
    if not rows:
        return

    dataframe = pd.DataFrame(rows)
    result.zones[zone_name] = ZoneData(name=zone_name, data=dataframe)

    for column in dataframe.columns:
        if column not in result.variables:
            result.variables.append(column)


def parse_cfd_grid_convergence_sheet(result: TecplotData, sheet_name: str, rows: list[list[str]]) -> None:
    """Parse CFD coefficient tables grouped by grid level and roughness height."""
    rows_by_roughness: dict[str, list[dict[str, Any]]] = {}
    row_index = 0

    while row_index < len(rows):
        level_number = parse_level_number(rows[row_index][0] if rows[row_index] else "")
        if level_number is None:
            row_index += 1
            continue

        header_row = rows[row_index + 1] if row_index + 1 < len(rows) else []
        column_names = [normalize_grid_convergence_column(value) for value in header_row]
        data_index = row_index + 2

        while data_index < len(rows):
            first_cell = rows[data_index][0] if rows[data_index] else ""
            if parse_level_number(first_cell) is not None:
                break

            roughness = to_float_or_none(first_cell)
            if roughness is not None:
                parsed_row: dict[str, Any] = {
                    "N": float(level_number),
                    "GRID_LEVEL": f"L{level_number}",
                    "ROUGHNESS_HEIGHT": roughness,
                    "CL": -999.0,
                    "CD": -999.0,
                    "CMX": -999.0,
                    "CMY": -999.0,
                    "CMZ": -999.0,
                    "WATER_MASS": -999.0,
                    "ICE_MASS": -999.0,
                    "WATER_EVAP_MASS": -999.0,
                }

                for column_index, column_name in enumerate(column_names):
                    if column_name not in parsed_row:
                        continue
                    if column_index >= len(rows[data_index]):
                        continue
                    cell_value = to_float_or_none(rows[data_index][column_index])
                    if cell_value is not None:
                        parsed_row[column_name] = cell_value

                if any(parsed_row[column] != -999.0 for column in ["CL", "CD", "CMX", "CMY", "CMZ"]):
                    if roughness == 0.0:
                        roughness_key = "smooth"
                    elif roughness < 0.0:
                        roughness_key = "variable_roughness"
                    else:
                        roughness_key = f"{roughness:g}mm"
                    rows_by_roughness.setdefault(roughness_key, []).append(parsed_row)

            data_index += 1

        row_index = data_index

    for roughness_key, zone_rows in rows_by_roughness.items():
        zone_name = f"{slug_from_text(sheet_name)}_CFD_roughness_{roughness_key}"
        add_grid_convergence_zone(result, zone_name, zone_rows)


def parse_icing_grid_convergence_sheet(result: TecplotData, sheet_name: str, rows: list[list[str]]) -> None:
    """Parse icing mass tables grouped by grid level and bin set."""
    rows_by_bin_set: dict[str, list[dict[str, Any]]] = {}
    rows_by_diameter_bin_set: dict[str, list[dict[str, Any]]] = {}
    row_index = 0

    while row_index < len(rows):
        level_number = parse_level_number(rows[row_index][0] if rows[row_index] else "")
        if level_number is None:
            row_index += 1
            continue

        group_row = rows[row_index + 1] if row_index + 1 < len(rows) else []
        header_row = rows[row_index + 2] if row_index + 2 < len(rows) else []
        group_starts: list[tuple[int, str]] = []

        for column_index, value in enumerate(group_row):
            match = re.search(r"(\d+)\s*-\s*Bin", str(value), re.IGNORECASE)
            if match is not None:
                group_starts.append((column_index, f"BINS{int(match.group(1)):02d}"))

        group_starts.append((len(header_row), "END"))

        for group_index in range(len(group_starts) - 1):
            start_column, bin_set = group_starts[group_index]
            end_column = group_starts[group_index + 1][0]
            column_names = [normalize_grid_convergence_column(value) for value in header_row[start_column:end_column]]
            data_index = row_index + 3

            while data_index < len(rows):
                first_cell = rows[data_index][0] if rows[data_index] else ""
                if parse_level_number(first_cell) is not None:
                    break

                bin_label = rows[data_index][start_column] if start_column < len(rows[data_index]) else ""
                is_combined_row = re.search(r"Combined", str(bin_label), re.IGNORECASE) is not None
                bin_number = to_float_or_none(bin_label)

                if is_combined_row or bin_number is not None:
                    parsed_row: dict[str, Any] = {
                        "N": float(level_number),
                        "GRID_LEVEL": f"L{level_number}",
                        "BIN_SET": bin_set,
                        "BIN": bin_number if bin_number is not None else -999.0,
                        "DIAMETER": -999.0,
                        "WATER_MASS": -999.0,
                        "ICE_MASS": -999.0,
                        "WATER_EVAP_MASS": -999.0,
                        "CL": -999.0,
                        "CD": -999.0,
                        "CMX": -999.0,
                        "CMY": -999.0,
                        "CMZ": -999.0,
                    }

                    for offset, column_name in enumerate(column_names):
                        column_index = start_column + offset
                        if column_name not in parsed_row:
                            continue
                        if column_index >= len(rows[data_index]):
                            continue
                        cell_value = to_float_or_none(rows[data_index][column_index])
                        if cell_value is not None:
                            parsed_row[column_name] = cell_value

                    has_mass_data = any(parsed_row[column] != -999.0 for column in ["WATER_MASS", "ICE_MASS", "WATER_EVAP_MASS"])

                    if is_combined_row and has_mass_data:
                        rows_by_bin_set.setdefault(bin_set, []).append(parsed_row)
                    elif bin_number is not None and parsed_row["DIAMETER"] != -999.0 and has_mass_data:
                        rows_by_diameter_bin_set.setdefault(bin_set, []).append(parsed_row)

                data_index += 1

        row_index += 1

    for bin_set, zone_rows in rows_by_bin_set.items():
        zone_name = f"{slug_from_text(sheet_name)}_Icing_{bin_set}"
        add_grid_convergence_zone(result, zone_name, zone_rows)

    for bin_set, zone_rows in rows_by_diameter_bin_set.items():
        zone_name = f"{slug_from_text(sheet_name)}_Icing_{bin_set}_by_diameter"
        add_grid_convergence_zone(result, zone_name, zone_rows)


def slug_from_text(text: str) -> str:
    """Create a stable identifier from workbook sheet names."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return slug or "sheet"


def read_grid_convergence_xlsx(path: Path, case_id: str) -> TecplotData:
    """
    Read an IPW3 gridConvergence.xlsx workbook into the TecplotData shape.

    The workbook stores CFD and icing data on separate sheets. This reader keeps
    the existing plotting API intact by converting each usable table to a zone.
    """
    result = TecplotData(path=path, title=path.name)

    for sheet_name, rows in read_xlsx_sheets(path).items():
        if case_id not in normalize_xlsx_case_ids(sheet_name):
            continue

        if re.search(r"\bCFD\b", sheet_name, re.IGNORECASE):
            parse_cfd_grid_convergence_sheet(result, sheet_name, rows)
        elif re.search(r"\bIcing\b", sheet_name, re.IGNORECASE):
            parse_icing_grid_convergence_sheet(result, sheet_name, rows)

    return result


# =============================================================================
# IPW3 zone-name helpers
# =============================================================================

def parse_ipw3_zone_name(zone_name: str) -> dict[str, str] | None:
    """
    Parse an IPW3 cutData or iceShape zone name.

    Accepted examples:
        SLICE_Y_0p9144m_BINS07_D01
        SLICE_Y_0.9144_BINS07_D01
        SLICE_Y_0p1_BINS03_KS_1mm_CUTDATA
        MCCS_BINS07_D01

    Returns:
        {"type": "SLICE", "slice": "0p9144m", "bins": "BINS07", "dataset": "D01"}
        {"type": "MCCS", "slice": "", "bins": "BINS07", "dataset": "D01"}
    """
    zone_name = zone_name.strip()

    slice_pattern = re.compile(r"^SLICE_Y_(?P<slice>.+?)_(?P<bins>BINS\d+)(?:_(?P<tail>.*))?$", re.IGNORECASE)
    mccs_pattern = re.compile(r"^MCCS_(?P<bins>BINS\d+)(?:_(?P<tail>.*))?$", re.IGNORECASE)

    match = slice_pattern.match(zone_name)
    if match is not None:
        tail = match.group("tail") or ""
        dataset_match = re.search(r"(?:^|_)(D?\d+)(?:_|$)", tail, re.IGNORECASE)
        return {
            "type": "SLICE",
            "slice": match.group("slice"),
            "bins": match.group("bins").upper(),
            "dataset": normalize_dataset_id(dataset_match.group(1) if dataset_match is not None else None),
        }

    match = mccs_pattern.match(zone_name)
    if match is not None:
        tail = match.group("tail") or ""
        dataset_match = re.search(r"(?:^|_)(D?\d+)(?:_|$)", tail, re.IGNORECASE)
        return {
            "type": "MCCS",
            "slice": "",
            "bins": match.group("bins").upper(),
            "dataset": normalize_dataset_id(dataset_match.group(1) if dataset_match is not None else None),
        }

    return None


def decode_slice_position(slice_name: str) -> float | None:
    """
    Convert IPW3 slice labels to numerical positions.

    Accepted examples:
        0.9144   ->  0.9144
        0p9144   ->  0.9144
        0.9144m  ->  0.9144
        m0p1000  -> -0.1000
        -0.1000  -> -0.1000
    """
    try:
        value = slice_name.strip()

        if value.endswith("m"):
            value = value[:-1]

        if value.startswith("m") and not value.startswith("m."):
            value = "-" + value[1:]

        value = value.replace("p", ".")

        return float(value)

    except ValueError:
        return None


# =============================================================================
# Folder/file-name parsing helpers
# =============================================================================

def normalize_dataset_id(dataset_id: str | int | None) -> str:
    """
    Normalize a dataset/attempt identifier.

    Examples:
        1    -> D01
        01   -> D01
        D01  -> D01
        d02  -> D02
    """
    if dataset_id is None:
        return "DXX"

    text = str(dataset_id).strip().upper()

    if text.startswith("D"):
        number_text = text[1:]
    else:
        number_text = text

    if number_text.isdigit():
        return f"D{int(number_text):02d}"

    return text


def parse_participant_folder_name(folder_name: str) -> tuple[str, str, str]:
    """
    Parse a participant folder name.

    Example:
        001_POLIMO_CHAMPS

    Returns:
        participant_id, organization, solver
    """
    parts = folder_name.split("_")

    participant_id = parts[0] if len(parts) >= 1 else "UNKNOWN"
    organization = parts[1] if len(parts) >= 2 else "UNKNOWN"
    solver = "_".join(parts[2:]) if len(parts) >= 3 else "UNKNOWN"

    return participant_id, organization, solver


def parse_submission_folder_name(folder_name: str) -> tuple[str, Optional[str], Optional[str]]:
    """
    Parse a submitted dataset/attempt folder name.

    Current expected example:
        001_TC_NACA0012_AE3933_01

    Returns:
        participant_id, case_id, dataset_id

    Notes
    -----
    The folder does not necessarily contain the grid level. The grid level is
    normally read from the file names, for example:
        TC_NACA0012_AE3933_L1_cutData_V1.dat
    """
    pattern = re.compile(r"^(?P<pid>\d{3})_(?P<case>TC_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)_(?P<dataset>\d{2})$", re.IGNORECASE)
    match = pattern.match(folder_name)

    if match is None:
        return folder_name, None, None

    return match.group("pid"), canonical_case_id(match.group("case")), normalize_dataset_id(match.group("dataset"))


def extract_case_id_from_name(name: str) -> Optional[str]:
    """Return the first valid case identifier found in a file or folder name."""
    for alias, canonical in sorted(CASE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in name:
            return canonical

    for case_id in sorted(VALID_CASES, key=len, reverse=True):
        if case_id in name:
            return case_id
    return None


def canonical_case_id(case_id: str) -> str:
    """Normalize accepted case aliases to the IDs used by the website."""
    return CASE_ALIASES.get(case_id, case_id)


def extract_grid_level_from_name(name: str) -> Optional[str]:
    """Extract grid level such as L1, L2, L3, or L4 from a file name."""
    match = re.search(r"(?:^|_)(L[1-9][0-9]*)(?:_|\.|$)", name, re.IGNORECASE)

    if match is None:
        return None

    grid_level = match.group(1).upper()

    if grid_level in VALID_GRID_LEVELS:
        return grid_level

    # Return it anyway so that unexpected levels can still be inspected.
    return grid_level


def identify_file_type(name: str) -> Optional[str]:
    """Classify a submitted data file based on its name."""
    if re.search(r"sMap", name, re.IGNORECASE):
        return "sMap"

    if re.search(r"gridConvergence", name, re.IGNORECASE):
        return "gridConvergence"

    if re.search(r"cutData", name, re.IGNORECASE):
        return "cutData"

    if re.search(r"iceShape", name, re.IGNORECASE):
        return "iceShape"

    return None


def is_participant_folder(path: Path) -> bool:
    """Return True for participant folders such as 001_POLIMO_CHAMPS."""
    if not path.is_dir():
        return False

    return re.match(r"^\d{3}_", path.name) is not None


# =============================================================================
# Folder scanning
# =============================================================================

def attach_file_to_participant(participant: Participant, file_path: Path, default_case_id: str | None, default_dataset_id: str | None) -> None:
    """
    Attach one .dat file to the correct participant/case/grid/dataset location.

    The hierarchy is:
        Participant -> Test case -> Grid level -> Dataset

    Grid convergence files are stored directly under the case because they are
    not specific to one grid level.
    """
    file_type = identify_file_type(file_path.name)
    if file_type == "sMap":
        return

    case_id = extract_case_id_from_name(file_path.name) or default_case_id
    grid_level = extract_grid_level_from_name(file_path.name)
    dataset_id = default_dataset_id or "DXX"

    if file_type == "gridConvergence" and file_path.suffix.lower() == ".xlsx" and case_id is None:
        detected_case_ids = detect_xlsx_grid_convergence_cases(file_path)

        if not detected_case_ids:
            participant.other_files.append(file_path)
            return

        for detected_case_id in detected_case_ids:
            case_data = participant.get_or_create_case(detected_case_id)
            case_data.grid_convergence_file = file_path

        return

    if case_id is None:
        participant.other_files.append(file_path)
        return

    case_data = participant.get_or_create_case(case_id)

    if file_type == "gridConvergence":
        case_data.grid_convergence_file = file_path
        return

    if file_type in {"cutData", "iceShape"}:
        if grid_level is None:
            case_data.other_files.append(file_path)
            return

        grid_data = case_data.get_or_create_grid_level(grid_level)
        dataset_data = grid_data.get_or_create_dataset(dataset_id, path=file_path.parent)

        if file_type == "cutData":
            dataset_data.cut_data_file = file_path
        elif file_type == "iceShape":
            dataset_data.ice_shape_file = file_path

        return

    case_data.other_files.append(file_path)


def scan_submission_folder(participant: Participant, submission_dir: Path) -> None:
    """
    Scan one submitted dataset/attempt folder.

    Example folder:
        001_TC_NACA0012_AE3933_01

    The case and dataset ID are taken from the folder name when possible. The
    grid level is taken from each file name.
    """
    _, case_id_from_folder, dataset_id_from_folder = parse_submission_folder_name(submission_dir.name)

    for file_path in sorted(submission_dir.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in {".dat", ".xlsx"}:
            continue

        attach_file_to_participant(participant, file_path, default_case_id=case_id_from_folder, default_dataset_id=dataset_id_from_folder)


def scan_participant_folder(participant_dir: Path) -> Participant:
    """Scan one participant folder and all submission subfolders inside it."""
    participant_id, organization, solver = parse_participant_folder_name(participant_dir.name)

    participant = Participant(
        participant_id=participant_id,
        organization=organization,
        solver=solver,
        path=participant_dir,
    )

    for child in sorted(participant_dir.iterdir()):
        if child.is_dir():
            scan_submission_folder(participant, child)
            continue

        if child.is_file() and child.suffix.lower() in {".dat", ".xlsx"}:
            # This supports the case where someone puts data files directly in
            # the participant folder. In this case, the dataset ID must come from
            # the zone names or defaults to DXX.
            attach_file_to_participant(participant, child, default_case_id=None, default_dataset_id=None)
            continue

        if child.is_file():
            participant.other_files.append(child)

    return participant


def scan_all_participants(root_dir: Path) -> list[Participant]:
    """Scan the root directory and return all participant data structures."""
    participants: list[Participant] = []

    for participant_dir in sorted(root_dir.iterdir()):
        if not is_participant_folder(participant_dir):
            continue

        participant = scan_participant_folder(participant_dir)
        participants.append(participant)

    return participants


# =============================================================================
# Iteration helpers for site builders
# =============================================================================

def iter_case_data(participants: list[Participant], case_id: str):
    """
    Iterate over participants that contain a given case.

    Yields:
        participant, case_data
    """
    for participant in participants:
        case_data = participant.cases.get(case_id)
        if case_data is None:
            continue
        yield participant, case_data


def iter_grid_datasets(participants: list[Participant], case_id: str, grid_level: str):
    """
    Iterate over all datasets for a given case and grid level.

    Yields:
        participant, case_data, grid_data, dataset_data
    """
    grid_level = grid_level.upper()

    for participant, case_data in iter_case_data(participants, case_id):
        grid_data = case_data.grid_levels.get(grid_level)
        if grid_data is None:
            continue

        for dataset_data in grid_data.datasets.values():
            yield participant, case_data, grid_data, dataset_data


def collect_case_ids(participants: list[Participant]) -> set[str]:
    """Collect all case IDs detected in the participant data."""
    case_ids: set[str] = set()

    for participant in participants:
        case_ids.update(participant.cases.keys())

    return case_ids


# =============================================================================
# Example access
# =============================================================================

def print_example_zone_access(participants: list[Participant]) -> None:
    """
    Print one concrete example showing how to access zone data.

    This is only a demonstration. It finds the first available cutData zone and
    prints basic information and min/max values for common variables.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE: Accessing one cutData zone")
    print("=" * 80)

    for participant in participants:
        for case_data in participant.cases.values():
            for grid_data in case_data.grid_levels.values():
                for dataset_data in grid_data.datasets.values():
                    if dataset_data.cut_data is None:
                        continue

                    zone_names = dataset_data.cut_data.zone_names()
                    if not zone_names:
                        continue

                    first_zone_name = zone_names[0]
                    zone = dataset_data.cut_data.get_zone(first_zone_name)

                    print(f"Participant: {participant.participant_id}")
                    print(f"Case:        {case_data.case_id}")
                    print(f"Grid level:  {grid_data.grid_level}")
                    print(f"Dataset:     {dataset_data.dataset_id}")
                    print(f"Zone:        {zone.name}")
                    print(f"Columns:     {list(zone.data.columns)}")
                    print(f"Rows:        {len(zone.data)}")

                    info = parse_ipw3_zone_name(zone.name)
                    if info is not None:
                        print("Zone metadata:")
                        print(f"  Type:        {info['type']}")
                        print(f"  Slice token: {info['slice']}")
                        print(f"  Slice value: {decode_slice_position(info['slice']) if info['slice'] else None}")
                        print(f"  Bins:        {info['bins']}")
                        print(f"  Dataset:     {info['dataset']}")

                    print("\nFirst rows:")
                    print(zone.data.head())

                    for variable in ["s", "Beta", "Cp", "HTC", "Ts", "FF"]:
                        if variable in zone.data.columns and not zone.data.empty:
                            print(f"{variable} min/max: {zone.data[variable].min()} / {zone.data[variable].max()}")

                    return

    print("No cutData zone found.")


def print_specific_zone_example(participants: list[Participant], case_id: str = "TC_NACA0012_AE3933", grid_level: str = "L1", bins_id: str = "BINS07", dataset_id: str = "D01", slice_position: float = 0.9144) -> None:
    """Example showing how to access a specific IPW3 zone by metadata."""
    print("\n" + "=" * 80)
    print("EXAMPLE: Accessing a specific IPW3 zone")
    print("=" * 80)
    print(f"Looking for case={case_id}, grid={grid_level}, dataset={dataset_id}, bins={bins_id}, slice={slice_position}")

    dataset_id = normalize_dataset_id(dataset_id)

    for participant, case_data, grid_data, dataset_data in iter_grid_datasets(participants, case_id, grid_level):
        if dataset_data.dataset_id != dataset_id:
            continue

        zone = dataset_data.get_cut_zone(bins_id=bins_id, slice_position=slice_position)

        if zone is None:
            continue

        print(f"Found zone: {zone.name}")
        print(f"Participant: {participant.participant_id}")
        print(f"Case:        {case_data.case_id}")
        print(f"Grid level:  {grid_data.grid_level}")
        print(f"Dataset:     {dataset_data.dataset_id}")
        print(zone.data.head())
        return

    print("Specific zone was not found.")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Scan participants from the current directory and print a summary."""
    root_dir = Path(".")

    log_step(f"Scanning participants in: {root_dir.resolve()}")
    participants = scan_all_participants(root_dir)
    print(f"Found {len(participants)} participant folder(s).")

    for participant in participants:
        participant.read_files()
        participant.summary()

    # Uncomment these while debugging if needed.
    # print_example_zone_access(participants)
    # print_specific_zone_example(participants, bins_id="BINS03", dataset_id="D01", slice_position=0.9144)


if __name__ == "__main__":
    main()
