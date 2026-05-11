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

def split_data(X, y_encoded):
    return train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

def train_model(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=200, # decision trees
        min_samples_leaf=2,
        random_state=42, 
        n_jobs=-1 # use all CPU cores for training
        )
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, le, X_test, y_test):
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    importances = clf.feature_importances_
    age_imp = importances[EXPECTED_FEATURES.index("age_group")]
    print(f"Feature importance for 'age_group': {age_imp*100:.1f}%")
    if age_imp > 0.40:
        print("WARNING: 'age_group' importance exceeds 40% — consider widening age distributions and regenerating the dataset.")

def save_model(clf, le):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": clf, "label_encoder": le}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    data = joblib.load(MODEL_PATH)
    return data["model"], data["label_encoder"]

def predict_top3(feature_vector):
    clf, le = load_model()
    
    row = {f: 0 for f in EXPECTED_FEATURES}
    row.update(feature_vector)
    X = pd.DataFrame([row])[EXPECTED_FEATURES]

    proba = clf.predict_proba(X)[0]
    top3_idx = proba.argsort()[-3:][::-1]

    return [(le.classes_[i], proba[i] * 100, 1) for i in top3_idx]


FEATURE_LABELS_PT = {
    "age_group":            "grupo etário",
    "gender":               "género",
    "duration":             "duração dos sintomas",
    "pain_intensity":       "intensidade da dor",
    "dor_no_peito":         "dor no peito",
    "irradiacao_braco":     "irradiação para o braço",
    "palpitacoes":          "palpitações",
    "suores_frios":         "suores frios",
    "falta_de_ar":          "falta de ar",
    "pieira":               "pieira",
    "tosse":                "tosse",
    "expectoracao":         "expectoração",
    "dor_ao_respirar":      "dor ao respirar",
    "febre":                "febre",
    "calafrios":            "calafrios",
    "fadiga":               "fadiga",
    "cefaleia_subita":      "cefaleia súbita",
    "confusao_mental":      "confusão mental",
    "visao_turva":          "visão turva",
    "fraqueza_facial":      "fraqueza facial",
    "dificuldade_falar":    "dificuldade em falar",
    "cefaleia_pulsatil":    "cefaleia pulsátil",
    "nausea":               "náusea",
    "sensibilidade_luz":    "sensibilidade à luz",
    "vomito":               "vómito",
    "diarreia":             "diarreia",
    "dor_abdominal_difusa": "dor abdominal difusa",
    "dor_abdominal_qid":    "dor no quadrante inferior direito",
    "rigidez_abdominal":    "rigidez abdominal",
    "perda_apetite":        "perda de apetite",
    "espirros":             "espirros",
    "dor_de_garganta":      "dor de garganta",
    "dor_localizada":       "dor localizada",
    "inchaço":              "inchaço",
    "ardor_ao_urinar":      "ardor ao urinar",
    "frequencia_urinaria":  "frequência urinária",
}

def explain(feature_vector, top_n=3):
    clf, _  = load_model()

    row = {f: 0 for f in EXPECTED_FEATURES}
    row.update(feature_vector)

    active = [f for f in EXPECTED_FEATURES if row[f] != 0]
    importances = clf.feature_importances_

    ranked = sorted(active, key=lambda f: importances[EXPECTED_FEATURES.index(f)], reverse=True)
    top = ranked[:top_n]

    return [FEATURE_LABELS_PT[f] for f in top]

if __name__ == "__main__":
    print("Loading data...")
    X, y = load_data()

    print("Encoding labels...")
    y_encoded, le = encode_labels(y)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)

    print("Training model...")
    clf = train_model(X_train, y_train)

    print("Evaluating...")
    evaluate(clf, le, X_test, y_test)

    print("Saving model...")
    save_model(clf, le)
