# tools/participant_style.py

PARTICIPANTS = [
    {
        "Participant ID": "001",
        "Organization": "CIRA",
        "Solver(s)": "SIMBA",
        "Name(s)": "Franscesco Capizzano",
    },
    {
        "Participant ID": "002",
        "Organization": "AEROTEX",
        "Solver(s)": "IHB3D",
        "Name(s)": "Ariane Vieira",
    },
    {
        "Participant ID": "004",
        "Organization": "ONERA",
        "Solver(s)": "CEDRE / IGLOO3D",
        "Name(s)": "Adèle Veilleux",
    },
    {
        "Participant ID": "006",
        "Organization": "DASSAULT AVIATION",
        "Solver(s)": "AETHER",
        "Name(s)": "François Caminade, Gianiel Zach",
    },
    {
        "Participant ID": "007",
        "Organization": "Polytechnique Montreal",
        "Solver(s)": "CHAMPS",
        "Name(s)": "Karim Zayni",
    },
    {
        "Participant ID": "009",
        "Organization": "Sikorsky",
        "Solver(s)": "STAR-CCM+",
        "Name(s)": "Jeewong Kim",
    },
    {
        "Participant ID": "014",
        "Organization": "Airbus",
        "Solver(s)": "CODA IGLOO3D",
        "Name(s)": "Alberto Della Noce",
    },
    {
        "Participant ID": "019",
        "Organization": "Synopsys",
        "Solver(s)": "FLUENT ICING",
        "Name(s)": "Isik Ozcer",
    },
    {
        "Participant ID": "020",
        "Organization": "Bombardier",
        "Solver(s)": "Dragon-Ice",
        "Name(s)": "Guy Fortin",
    },
]

PARTICIPANT_COLORS = {
    "001": "#00FF40",
    "002": "#00BDDB",
    "004": "#4d7502",
    "006": "#4f0077",
    "007": "#1f77b4",
    "009": "#ad1d03",
    "014": "#aa00ff",
    "019": "#e39000",
    "020": "#573700",
}

PREVIEW_PARTICIPANT_NAME = "All participants"


def normalize_participant_id(participant_id: str | int) -> str:
    text = str(participant_id).strip()
    if text.isdigit():
        return f"{int(text):03d}"
    return text


def participant_color(participant_id: str) -> str:
    return PARTICIPANT_COLORS.get(normalize_participant_id(participant_id), "black")


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
