import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "symptoms_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")


EXPECTED_FEATURES = [
    "age_group", "gender", "duration", "pain_intensity",
    "dor_no_peito", "irradiacao_braco", "palpitacoes", "suores_frios",
    "falta_de_ar", "pieira", "tosse", "expectoracao", "dor_ao_respirar",
    "febre", "calafrios", "fadiga", "cefaleia_subita", "confusao_mental",
    "visao_turva", "fraqueza_facial", "dificuldade_falar", "cefaleia_pulsatil",
    "nausea", "sensibilidade_luz", "vomito", "diarreia", "dor_abdominal_difusa",
    "dor_abdominal_qid", "rigidez_abdominal", "perda_apetite", "espirros",
    "dor_de_garganta", "dor_localizada", "inchaço", "ardor_ao_urinar",
    "frequencia_urinaria",
]
TARGET = "diagnosis"


def load_data():
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in EXPECTED_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    return df[EXPECTED_FEATURES], df[TARGET]

def encode_labels(y):
    le = LabelEncoder()
    return le.fit_transform(y), le

