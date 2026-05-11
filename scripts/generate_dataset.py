"""
Synthetic patient dataset generator for SNS24 triage system.

Generates 560+ rows covering 11 conditions across 5 specialties.
Row counts are set to approximate real-world Portuguese SNS prevalence:
  Gripe=100, Constipação=80, Gastroenterite=70, ITU=60, Enxaqueca=50,
  Asma=45, Fratura=40, Pneumonia=40, Apendicite=35, AVC=20, Enfarte=20

Demographic distributions per condition:
  - age_group (0=0-17, 1=18-40, 2=41-65, 3=65+)
  - gender (0=F, 1=M)
  - duration (0=<24h, 1=1-3 days, 2=>3 days)
  - pain_intensity (1-10 integer), skewed by Manchester triage level:
      Red   (Enfarte, AVC)        -> mean 8, std 1.2
      Orange (Pneumonia, Apendc.) -> mean 6, std 1.5
      Yellow (Asma, Fratura)      -> mean 5, std 1.5
      Green  (Gripe, Gastro, Enx) -> mean 4, std 1.5
      Blue   (ITU, Constipação)   -> mean 2, std 1.2

Symptom-to-condition probability matrix built from:
  Mayo Clinic, CDC, PubMed, and WHO clinical guidelines.
"""

import random
import csv
import os

random.seed(42)

SYMPTOMS = [
    "dor_no_peito", "irradiacao_braco", "palpitacoes", "suores_frios",
    "falta_de_ar", "pieira", "tosse", "expectoracao", "dor_ao_respirar",
    "febre", "calafrios", "fadiga", "cefaleia_subita", "confusao_mental",
    "visao_turva", "fraqueza_facial", "dificuldade_falar", "cefaleia_pulsatil",
    "nausea", "sensibilidade_luz", "vomito", "diarreia", "dor_abdominal_difusa",
    "dor_abdominal_qid", "rigidez_abdominal", "perda_apetite", "espirros",
    "dor_de_garganta", "dor_localizada", "inchaço", "ardor_ao_urinar",
    "frequencia_urinaria",
]

# Noise floor for symptoms not listed per condition
NOISE = 0.05

# fmt: off
# symptom_probs[condition] = {symptom: probability}
# Only non-noise probabilities are listed; everything else defaults to NOISE.
SYMPTOM_PROBS = {
    "Enfarte do Miocárdio": {
        "dor_no_peito": 0.95, "irradiacao_braco": 0.80, "suores_frios": 0.75,
        "falta_de_ar": 0.70, "palpitacoes": 0.60, "fadiga": 0.65, "nausea": 0.50,
    },
    "AVC": {
        "fraqueza_facial": 0.90, "dificuldade_falar": 0.85, "confusao_mental": 0.80,
        "cefaleia_subita": 0.75, "visao_turva": 0.70, "falta_de_ar": 0.30,
        "fadiga": 0.40,
    },
    "Pneumonia": {
        "febre": 0.90, "tosse": 0.90, "expectoracao": 0.85, "dor_ao_respirar": 0.75,
        "falta_de_ar": 0.70, "calafrios": 0.65, "fadiga": 0.60, "dor_no_peito": 0.30,
    },
    "Apendicite": {
        "dor_abdominal_qid": 0.95, "rigidez_abdominal": 0.80, "febre": 0.75,
        "nausea": 0.80, "vomito": 0.65, "perda_apetite": 0.85, "dor_abdominal_difusa": 0.30,
    },
    "Asma (crise)": {
        "pieira": 0.95, "falta_de_ar": 0.90, "tosse": 0.80, "dor_no_peito": 0.50,
        "fadiga": 0.40,
    },
    "Fratura Óssea": {
        "dor_localizada": 0.98, "inchaço": 0.90, "dor_ao_respirar": 0.20,
        "fadiga": 0.20,
    },
    "Gripe": {
        "febre": 0.85, "fadiga": 0.80, "tosse": 0.75, "calafrios": 0.70,
        "dor_de_garganta": 0.65, "suores_frios": 0.50, "espirros": 0.40,
        "cefaleia_pulsatil": 0.40, "nausea": 0.30, "dor_no_peito": 0.10,
    },
    "Gastroenterite": {
        "diarreia": 0.95, "nausea": 0.90, "vomito": 0.85, "dor_abdominal_difusa": 0.80,
        "febre": 0.40, "fadiga": 0.50, "perda_apetite": 0.50,
    },
    "Enxaqueca": {
        "cefaleia_pulsatil": 0.95, "sensibilidade_luz": 0.90, "nausea": 0.80,
        "visao_turva": 0.60, "vomito": 0.50, "fadiga": 0.45,
    },
    "Infeção Urinária": {
        "ardor_ao_urinar": 0.95, "frequencia_urinaria": 0.90, "febre": 0.45,
        "dor_abdominal_difusa": 0.30, "fadiga": 0.35,
    },
    "Constipação": {
        "espirros": 0.90, "dor_de_garganta": 0.85, "tosse": 0.60,
        "febre": 0.20, "fadiga": 0.30, "cefaleia_pulsatil": 0.25,
    },
}
# fmt: on

# age_group weights per condition: [0-17, 18-40, 41-65, 65+]
AGE_WEIGHTS = {
    "Enfarte do Miocárdio": [0.02, 0.08, 0.40, 0.50],
    "AVC":                  [0.03, 0.07, 0.35, 0.55],
    "Pneumonia":            [0.15, 0.15, 0.30, 0.40],
    "Apendicite":           [0.20, 0.45, 0.25, 0.10],
    "Asma (crise)":         [0.30, 0.30, 0.25, 0.15],
    "Fratura Óssea":        [0.20, 0.25, 0.25, 0.30],
    "Gripe":                [0.25, 0.30, 0.25, 0.20],
    "Gastroenterite":       [0.30, 0.30, 0.25, 0.15],
    "Enxaqueca":            [0.10, 0.45, 0.35, 0.10],
    "Infeção Urinária":     [0.10, 0.35, 0.30, 0.25],
    "Constipação":          [0.30, 0.30, 0.25, 0.15],
}

# duration weights per condition: [<24h, 1-3 days, >3 days]
DURATION_WEIGHTS = {
    "Enfarte do Miocárdio": [0.80, 0.15, 0.05],
    "AVC":                  [0.85, 0.10, 0.05],
    "Pneumonia":            [0.30, 0.50, 0.20],
    "Apendicite":           [0.60, 0.35, 0.05],
    "Asma (crise)":         [0.50, 0.35, 0.15],
    "Fratura Óssea":        [0.70, 0.25, 0.05],
    "Gripe":                [0.20, 0.50, 0.30],
    "Gastroenterite":       [0.25, 0.55, 0.20],
    "Enxaqueca":            [0.40, 0.45, 0.15],
    "Infeção Urinária":     [0.15, 0.35, 0.50],
    "Constipação":          [0.10, 0.30, 0.60],
}

# pain_intensity (mean, std) per condition, clamped to [1, 10]
PAIN_PARAMS = {
    "Enfarte do Miocárdio": (8.0, 1.2),
    "AVC":                  (7.5, 1.5),
    "Pneumonia":            (6.0, 1.5),
    "Apendicite":           (6.5, 1.5),
    "Asma (crise)":         (5.5, 1.5),
    "Fratura Óssea":        (6.0, 1.5),
    "Gripe":                (4.0, 1.5),
    "Gastroenterite":       (4.0, 1.5),
    "Enxaqueca":            (5.0, 1.5),
    "Infeção Urinária":     (3.0, 1.5),
    "Constipação":          (2.0, 1.2),
}

# Realistic prevalence-weighted row counts (total ~560)
ROW_COUNTS = {
    "Gripe":                100,
    "Constipação":           80,
    "Gastroenterite":        70,
    "Infeção Urinária":      60,
    "Enxaqueca":             50,
    "Asma (crise)":          45,
    "Fratura Óssea":         40,
    "Pneumonia":             40,
    "Apendicite":            35,
    "AVC":                   20,
    "Enfarte do Miocárdio":  20,
}


def _weighted_choice(options, weights):
    r = random.random()
    cumulative = 0.0
    for option, weight in zip(options, weights):
        cumulative += weight
        if r < cumulative:
            return option
    return options[-1]


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _gauss_int(mean, std, lo, hi):
    return _clamp(round(random.gauss(mean, std)), lo, hi)


def generate_row(condition):
    probs = SYMPTOM_PROBS[condition]
    symptoms = {s: 1 if random.random() < probs.get(s, NOISE) else 0 for s in SYMPTOMS}

    age_group = _weighted_choice([0, 1, 2, 3], AGE_WEIGHTS[condition])
    gender = random.randint(0, 1)
    duration = _weighted_choice([0, 1, 2], DURATION_WEIGHTS[condition])
    mean_pain, std_pain = PAIN_PARAMS[condition]
    pain_intensity = _gauss_int(mean_pain, std_pain, 1, 10)

    row = {
        "age_group": age_group,
        "gender": gender,
        "duration": duration,
        "pain_intensity": pain_intensity,
        **symptoms,
        "diagnosis": condition,
    }
    return row


def generate_dataset():
    rows = []
    for condition, count in ROW_COUNTS.items():
        for _ in range(count):
            rows.append(generate_row(condition))

    random.shuffle(rows)
    return rows


def check_age_importance(rows):
    """Quick train to verify age_group feature importance stays under 40%."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np

        columns = ["age_group", "gender", "duration", "pain_intensity"] + SYMPTOMS
        label_map = {c: i for i, c in enumerate(ROW_COUNTS.keys())}

        X = [[row[c] for c in columns] for row in rows]
        y = [label_map[row["diagnosis"]] for row in rows]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        importances = clf.feature_importances_
        age_importance = importances[0]  # age_group is first column
        print(f"  age_group importance: {age_importance:.3f} ({age_importance*100:.1f}%)")

        if age_importance > 0.40:
            print("  WARNING: age_group importance exceeds 40% — consider widening age distributions.")
            return False
        print("  age_group importance OK (< 40%)")
        return True
    except ImportError:
        print("  sklearn not available — skipping importance check.")
        return True


def save_csv(rows, path):
    fieldnames = ["age_group", "gender", "duration", "pain_intensity"] + SYMPTOMS + ["diagnosis"]
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows to {path}")


if __name__ == "__main__":
    print("Generating synthetic patient dataset...")
    rows = generate_dataset()
    print(f"  Generated {len(rows)} rows across {len(ROW_COUNTS)} conditions")

    print("Checking feature importances...")
    check_age_importance(rows)

    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "symptoms_data.csv")
    print("Saving dataset...")
    save_csv(rows, os.path.normpath(output_path))

    print("Done.")
