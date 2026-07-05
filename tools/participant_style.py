# tools/participant_style.py

PARTICIPANTS = [
    {
        "Participant ID": "004",
        "Organization": "ONERA",
        "Solver(s)": "CEDRE / IGLOO3D",
        "Name(s)": "Adèle Veilleux",
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
]

PARTICIPANT_COLORS = {
    "004": "#4d7502",
    "007": "#1f77b4",
    "009": "#ad1d03",
}

PREVIEW_PARTICIPANT_NAME = "All participants"


def normalize_participant_id(participant_id: str | int) -> str:
    text = str(participant_id).strip()
    if text.isdigit():
        return f"{int(text):03d}"
    return text


def participant_color(participant_id: str) -> str:
    return PARTICIPANT_COLORS.get(normalize_participant_id(participant_id), "black")


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
