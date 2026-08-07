# tools/participant_style.py

PARTICIPANTS = [
    {
        "Participant ID": "001",
        "Organization": "CIRA",
        "Solver(s)": "SIMBA",
        "Name(s)": "Franscesco Capizzano",
        "Show Information on Index": True,
    },
    {
        "Participant ID": "002",
        "Organization": "AEROTEX",
        "Solver(s)": "IHB3D",
        "Name(s)": "Ariane Vieira",
        "Show Information on Index": True,
    },
    {
        "Participant ID": "003",
        "Organization": "NRC",
        "Solver(s)": "MORPHOGENETIC",
        "Name(s)": "Pete Forsyth",
        "Show Information on Index": False,
    },
    {
        "Participant ID": "004",
        "Organization": "ONERA",
        "Solver(s)": "CEDRE / IGLOO3D",
        "Name(s)": "Adèle Veilleux, Emmanuel Radenac",
        "Show Information on Index": True,
    },
    {
        "Participant ID": "006",
        "Organization": "DASSAULT AVIATION",
        "Solver(s)": "AETHER",
        "Name(s)": "François Caminade, Gianiel Zach",
        "Show Information on Index": True,
    },
    {
        "Participant ID": "007",
        "Organization": "Polytechnique Montreal",
        "Solver(s)": "CHAMPS",
        "Name(s)": "Karim Zayni",
        "Show Information on Index": True,
    },
    {
        "Participant ID": "009",
        "Organization": "Sikorsky",
        "Solver(s)": "STAR-CCM+",
        "Name(s)": "Jeewong Kim",
        "Show Information on Index": True,
    },
    {
        "Participant ID": "008",
        "Organization": "Collins Aerospace",
        "Solver(s)": "FENSAPICE",
        "Name(s)": "Mateusz Pawlucki",
        "Show Information on Index": True,
    },
    {
        "Participant ID": "010",
        "Organization": "NASA",
        "Solver(s)": "GlennICE",
        "Name(s)": "Thomas Ozoroski",
        "Show Information on Index": False,
    },
    {
        "Participant ID": "013",
        "Organization": "Boeing",
        "Solver(s)": "GlennICE 5.1 / 6.2",
        "Name(s)": "Adam Malone, Soroush Yazdani",
        "Show Information on Index": True,
    },
    {
        "Participant ID": "014",
        "Organization": "Airbus",
        "Solver(s)": "CODA IGLOO3D",
        "Name(s)": "Alberto Della Noce",
        "Show Information on Index": True,
    },
    {
        "Participant ID": "015",
        "Organization": "NIAR",
        "Solver(s)": "FENSAPICE",
        "Name(s)": "Harsh Shah",
        "Show Information on Index": True,
    },
    {
        "Participant ID": "019",
        "Organization": "Synopsys",
        "Solver(s)": "FLUENT ICING",
        "Name(s)": "Isik Ozcer",
        "Show Information on Index": False,
    },
    {
        "Participant ID": "020",
        "Organization": "Bombardier",
        "Solver(s)": "Dragon-Ice",
        "Name(s)": "Guy Fortin",
        "Show Information on Index": True,
    },
]

PARTICIPANT_COLORS = {
    "001": "#2ca02c",
    "002": "#17becf",
    "003": "#8c564b",
    "004": "#bcbd22",
    "006": "#9467bd",
    "007": "#1f77b4",
    "008": "#d62728",
    "009": "#ff7f0e",
    "010": "#7f7f7f",
    "013": "#e6550d",
    "014": "#e377c2",
    "015": "#393b79",
    "019": "#637939",
    "020": "#843c39",
}

# Set each participant to True or False to show or hide its markers.
PARTICIPANTS_MARKERS = {
    "001": False,
    "002": False,
    "003": False,
    "004": False,
    "006": False,
    "007": False,
    "008": False,
    "009": False,
    "010": False,
    "013": False,
    "014": False,
    "015": False,
    "019": False,
    "020": False,
}

# Shape options include: circle, square, diamond, cross, x, triangle-up,
# triangle-down, triangle-left, triangle-right, pentagon, hexagon, and star.
PARTICIPANT_MARKER_SHAPES = {
    "001": "circle",
    "002": "circle",
    "003": "circle",
    "004": "circle",
    "006": "circle",
    "007": "circle",
    "008": "circle",
    "009": "circle",
    "010": "circle",
    "013": "circle",
    "014": "circle",
    "015": "circle",
    "019": "circle",
    "020": "circle",
}

# Marker size is in pixels. Typical values are 6 (small), 9 (medium),
# 12 (large), and 16 (extra large).
PARTICIPANT_MARKER_SIZES = {
    "001": 9,
    "002": 9,
    "003": 9,
    "004": 9,
    "006": 9,
    "007": 9,
    "008": 9,
    "009": 9,
    "010": 9,
    "013": 9,
    "014": 9,
    "015": 9,
    "019": 9,
    "020": 9,
}

# Show a marker every N data points on non-convergence plots.
# For example: 1 = every point, 5 = every fifth point, 10 = every tenth point.
PARTICIPANT_MARKER_FREQUENCIES = {
    "001": 1,
    "002": 1,
    "003": 1,
    "004": 1,
    "006": 1,
    "007": 1,
    "008": 1,
    "009": 1,
    "010": 1,
    "013": 1,
    "014": 1,
    "015": 1,
    "019": 1,
    "020": 1,
}

PREVIEW_PARTICIPANT_NAME = "All participants"


def normalize_participant_id(participant_id: str | int) -> str:
    text = str(participant_id).strip()
    if text.isdigit():
        return f"{int(text):03d}"
    return text


def participant_color(participant_id: str) -> str:
    return PARTICIPANT_COLORS.get(normalize_participant_id(participant_id), "black")


def participant_trace_mode(participant_id: str | int) -> str:
    normalized_id = normalize_participant_id(participant_id)
    return "lines+markers" if PARTICIPANTS_MARKERS.get(normalized_id, False) else "lines"


def participant_marker(participant_id: str | int, point_count: int | None = None) -> dict:
    normalized_id = normalize_participant_id(participant_id)
    marker_size = PARTICIPANT_MARKER_SIZES.get(normalized_id, 9)
    if point_count is not None:
        frequency = max(1, PARTICIPANT_MARKER_FREQUENCIES.get(normalized_id, 1))
        marker_size = [
            marker_size if point_index % frequency == 0 else 0
            for point_index in range(point_count)
        ]

    return {
        "color": participant_color(normalized_id),
        "symbol": PARTICIPANT_MARKER_SHAPES.get(normalized_id, "circle"),
        "size": marker_size,
    }


def participant_legend_rank(participant_id: str | int) -> int:
    """Return the canonical participant-table order for Plotly legends."""
    normalized_id = normalize_participant_id(participant_id)
    ordered_ids = [normalize_participant_id(item["Participant ID"]) for item in PARTICIPANTS]
    try:
        return ordered_ids.index(normalized_id)
    except ValueError:
        return len(ordered_ids)


def participant_info(participant_id: str | int) -> dict[str, str] | None:
    normalized_id = normalize_participant_id(participant_id)
    for participant in PARTICIPANTS:
        if normalize_participant_id(participant["Participant ID"]) == normalized_id:
            return participant
    return None


def preview_participant_name(participant_id: str | int | None) -> str:
    if participant_id is None:
        return PREVIEW_PARTICIPANT_NAME

    normalized_id = normalize_participant_id(participant_id)
    info = participant_info(normalized_id)
    if info is None:
        return f"Participant {normalized_id}"

    organization = info.get("Organization", "").strip()
    solver = info.get("Solver(s)", "").strip()
    name = info.get("Name(s)", "").strip()
    display_parts = [part for part in [organization, solver] if part]
    display_name = " | ".join(display_parts) or name or f"Participant {normalized_id}"
    return f"{normalized_id} - {display_name}"
