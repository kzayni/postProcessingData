from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import re

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

    def read_files(self) -> None:
        """Read available cutData and iceShape files for this dataset/attempt."""
        if self.cut_data_file is not None:
            self.cut_data = read_tecplot_dat(self.cut_data_file)

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

    def read_files(self) -> None:
        """Read all dataset/attempt files attached to this grid level."""
        for dataset_data in self.datasets.values():
            dataset_data.read_files()


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

    def read_files(self) -> None:
        """Read all files attached to this case."""
        if self.grid_convergence_file is not None:
            self.grid_convergence_data = read_tecplot_dat(self.grid_convergence_file)

        for grid_level_data in self.grid_levels.values():
            grid_level_data.read_files()


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
        if case_id not in self.cases:
            self.cases[case_id] = create_case_data(case_id)

        return self.cases[case_id]

    def read_files(self) -> None:
        """Read all submitted files for this participant."""
        for case_data in self.cases.values():
            case_data.read_files()

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


def read_tecplot_dat(path: Path) -> TecplotData:
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
        MCCS_BINS07_D01

    Returns:
        {"type": "SLICE", "slice": "0p9144m", "bins": "BINS07", "dataset": "D01"}
        {"type": "MCCS", "slice": "", "bins": "BINS07", "dataset": "D01"}
    """
    zone_name = zone_name.strip()

    slice_pattern = re.compile(r"^SLICE_Y_(?P<slice>.+?)_(?P<bins>BINS\d+)_(?P<dataset>D?\d+)$", re.IGNORECASE)
    mccs_pattern = re.compile(r"^MCCS_(?P<bins>BINS\d+)_(?P<dataset>D?\d+)$", re.IGNORECASE)

    match = slice_pattern.match(zone_name)
    if match is not None:
        return {
            "type": "SLICE",
            "slice": match.group("slice"),
            "bins": match.group("bins").upper(),
            "dataset": normalize_dataset_id(match.group("dataset")),
        }

    match = mccs_pattern.match(zone_name)
    if match is not None:
        return {
            "type": "MCCS",
            "slice": "",
            "bins": match.group("bins").upper(),
            "dataset": normalize_dataset_id(match.group("dataset")),
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

    return match.group("pid"), match.group("case"), normalize_dataset_id(match.group("dataset"))


def extract_case_id_from_name(name: str) -> Optional[str]:
    """Return the first valid case identifier found in a file or folder name."""
    for case_id in sorted(VALID_CASES, key=len, reverse=True):
        if case_id in name:
            return case_id
    return None


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
    """Classify a .dat file based on its name."""
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
    case_id = extract_case_id_from_name(file_path.name) or default_case_id
    grid_level = extract_grid_level_from_name(file_path.name)
    dataset_id = default_dataset_id or "DXX"

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

        if file_path.suffix.lower() != ".dat":
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

        if child.is_file() and child.suffix.lower() == ".dat":
            # This supports the case where someone puts .dat files directly in
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
