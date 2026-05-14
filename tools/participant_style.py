# tools/participant_style.py

PARTICIPANT_COLORS = {
    "001": "#1f77b4",
}


def participant_color(participant_id: str) -> str:
    return PARTICIPANT_COLORS.get(participant_id, "black")