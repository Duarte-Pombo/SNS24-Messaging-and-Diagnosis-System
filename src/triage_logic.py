"""Clinical triage mapping utilities."""

TRIAGE_BY_DIAGNOSIS = {
    "Enfarte do Miocárdio": (
        "Red",
        "Ligue 112 e encaminhe para urgência hospitalar imediata.",
    ),
    "AVC": (
        "Red",
        "Ligue 112 e encaminhe para urgência hospitalar imediata.",
    ),
    "Pneumonia": (
        "Orange",
        "Encaminhar para observação médica urgente nas próximas horas.",
    ),
    "Apendicite": (
        "Orange",
        "Encaminhar para observação médica urgente nas próximas horas.",
    ),
    "Asma (crise)": (
        "Yellow",
        "Requer avaliação médica presencial no próprio dia.",
    ),
    "Fratura Óssea": (
        "Yellow",
        "Requer avaliação médica presencial no próprio dia.",
    ),
    "Gripe": (
        "Green",
        "Aconselhar cuidados em casa e vigilância com contacto ao médico se agravar.",
    ),
    "Gastroenterite": (
        "Green",
        "Aconselhar hidratação e vigilância com contacto ao médico se agravar.",
    ),
    "Enxaqueca": (
        "Green",
        "Aconselhar consulta programada e vigilância de sinais de alarme.",
    ),
    "Infeção Urinária": (
        "Blue",
        "Encaminhar para cuidados de saúde primários para avaliação não urgente.",
    ),
    "Constipação": (
        "Blue",
        "Encaminhar para autocuidados e/ou cuidados de saúde primários.",
    ),
}

_DIAGNOSIS_ALIASES = {
    "heart attack": "Enfarte do Miocárdio",
    "stroke": "AVC",
    "pneumonia": "Pneumonia",
    "appendicitis": "Apendicite",
    "asthma attack": "Asma (crise)",
    "fracture": "Fratura Óssea",
    "flu": "Gripe",
    "gastroenteritis": "Gastroenterite",
    "migraine": "Enxaqueca",
    "urinary tract infection": "Infeção Urinária",
    "common cold": "Constipação",
}


def triage_diagnosis(predicted_diagnosis):
    """Return Manchester color and recommendation for a predicted diagnosis."""
    if not isinstance(predicted_diagnosis, str) or not predicted_diagnosis.strip():
        raise ValueError("predicted_diagnosis must be a non-empty string")

    diagnosis = predicted_diagnosis.strip()
    normalized = _DIAGNOSIS_ALIASES.get(diagnosis.lower(), diagnosis)

    if normalized not in TRIAGE_BY_DIAGNOSIS:
        raise ValueError(f"Unsupported diagnosis: {predicted_diagnosis}")

    return TRIAGE_BY_DIAGNOSIS[normalized]
