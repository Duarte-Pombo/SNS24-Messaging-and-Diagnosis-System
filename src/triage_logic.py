"""Manchester Triage mapping: diagnosis -> color, preferred specialty, care types.
"""

RED, ORANGE, YELLOW, GREEN, BLUE = "Red", "Orange", "Yellow", "Green", "Blue"

HOSPITAL      = "Hospital"
CLINIC        = "Clinic"
HEALTH_CENTER = "Health Center"

DIAGNOSIS_TRIAGE = {
    "Enfarte do Miocárdio": {"color": RED,    "specialty_keywords": ["cardio"], "care_types": [HOSPITAL]},
    "AVC":                  {"color": RED,    "specialty_keywords": ["neuro"],  "care_types": [HOSPITAL]},
    "Pneumonia":            {"color": ORANGE, "specialty_keywords": ["pneumo"], "care_types": [HOSPITAL, CLINIC]},
    "Apendicite":           {"color": ORANGE, "specialty_keywords": ["cirurg"], "care_types": [HOSPITAL]},
    "Asma (crise)":         {"color": YELLOW, "specialty_keywords": ["pneumo"], "care_types": [HOSPITAL, CLINIC]},
    "Fratura Óssea":        {"color": YELLOW, "specialty_keywords": ["ortop"],  "care_types": [HOSPITAL, CLINIC]},
    "Gripe":                {"color": GREEN,  "specialty_keywords": [],         "care_types": [CLINIC, HEALTH_CENTER, HOSPITAL]},
    "Gastroenterite":       {"color": GREEN,  "specialty_keywords": ["gastro"], "care_types": [CLINIC, HEALTH_CENTER, HOSPITAL]},
    "Enxaqueca":            {"color": GREEN,  "specialty_keywords": ["neuro"],  "care_types": [CLINIC, HEALTH_CENTER, HOSPITAL]},
    "Infeção Urinária":     {"color": BLUE,   "specialty_keywords": ["urolog"], "care_types": [HEALTH_CENTER, CLINIC, HOSPITAL]},
    "Constipação":          {"color": BLUE,   "specialty_keywords": [],         "care_types": [HEALTH_CENTER, CLINIC, HOSPITAL]},
}

COLOR_LABEL_PT = {
    RED:    ("Vermelho", "Emergência — ligue 112 imediatamente"),
    ORANGE: ("Laranja",  "Muito urgente — dirija-se a urgências"),
    YELLOW: ("Amarelo",  "Urgente — procure atendimento nas próximas 2 horas"),
    GREEN:  ("Verde",    "Pouco urgente — consulta médica nas próximas 24h"),
    BLUE:   ("Azul",     "Não urgente — consulte o médico de família"),
}

COLOR_HEX = {
    RED:    "#e74c3c",
    ORANGE: "#e67e22",
    YELLOW: "#f1c40f",
    GREEN:  "#27ae60",
    BLUE:   "#2980b9",
}


def get_triage(diagnosis: str) -> dict:
    info = DIAGNOSIS_TRIAGE.get(diagnosis)
    if info is None:
        return {"color": GREEN, "specialty_keywords": [], "care_types": [CLINIC, HEALTH_CENTER, HOSPITAL]}
    return info


def get_color_label(color: str) -> tuple[str, str]:
    return COLOR_LABEL_PT.get(color, ("Verde", "Consulte o seu médico"))


def get_color_hex(color: str) -> str:
    return COLOR_HEX.get(color, "#95a5a6")
