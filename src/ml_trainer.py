import os
import joblib
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Directory configs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "symptoms_dataset")
SEVERITY_PATH = os.path.join(DATA_DIR, "Symptom-severity_pt.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DATA_FILES = {
    "clean": "dataset_binary_pt.csv",
    "low_noise": "dataset_binary_pt_low_noise.csv",
    "high_noise": "dataset_binary_pt_high_noise.csv",
    "synthetic": "dataset_synthetic_pt.csv"
}

TARGET = "diagnosis"

def load_data(dataset_type="clean"):
    if dataset_type == "combined":
        dfs = []
        for name, filename in DATA_FILES.items():
            path = os.path.join(DATA_DIR, filename)
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
            else:
                print(f"Warning: {path} not found. Skipping.")
        if not dfs:
            raise FileNotFoundError("No datasets found to combine.")
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded combined dataset. Total shape: {df.shape}")
    else:
        path = os.path.join(DATA_DIR, DATA_FILES[dataset_type])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path)
        print(f"Loaded {dataset_type} dataset. Shape: {df.shape}")
        
    # Exclude target and any other non-feature columns
    exclude_cols = {TARGET, 'prognóstico'}
    features = [c for c in df.columns if c not in exclude_cols]
    
    # Use severity dataset to engineer a 'total_severity' feature
    if os.path.exists(SEVERITY_PATH):
        severity_df = pd.read_csv(SEVERITY_PATH)
        severity_map = dict(zip(severity_df['Symptom'], severity_df['weight']))
        
        def calculate_severity(row):
            score = 0
            for sym in features:
                if row[sym] == 1:
                    score += severity_map.get(sym, 1)
            return score
            
        df['total_severity'] = df.apply(calculate_severity, axis=1)
        features.append('total_severity')
        
    return df[features], df[TARGET], features

def encode_labels(y):
    le = LabelEncoder()
    return le.fit_transform(y), le

def split_data(X, y_encoded):
    return train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

def train_gradient_boosting(X_train, y_train):
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def train_logistic_regression(X_train, y_train):
    clf = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

def save_model(clf, le, features, filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    # Save the feature list in the joblib dump for inference validation
    joblib.dump({"model": clf, "label_encoder": le, "features": features}, path)
    print(f"Model saved to {path}")

def load_model(model_name="random_forest.pkl"):
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    data = joblib.load(model_path)
    # Return extracted features with a fallback
    return data["model"], data["label_encoder"], data.get("features", [])

def predict_top3(feature_vector, model_name="random_forest.pkl"):
    clf, le, features = load_model(model_name)
    
    row = {f: 0 for f in features}
    row.update(feature_vector)
    
    # Recalculate total_severity safely for inference
    if 'total_severity' in features:
        try:
            severity_df = pd.read_csv(SEVERITY_PATH)
            severity_map = dict(zip(severity_df['Symptom'], severity_df['weight']))
            score = sum(severity_map.get(k, 1) for k, v in feature_vector.items() if v == 1 and k != 'total_severity')
            row['total_severity'] = score
        except Exception:
            row['total_severity'] = 0

    X = pd.DataFrame([row])[features]

    proba = clf.predict_proba(X)[0]
    top3_idx = proba.argsort()[-3:][::-1]

    return [(le.classes_[i], proba[i] * 100, 1) for i in top3_idx]

def explain(feature_vector, top_n=3, model_name="random_forest.pkl"):
    clf, _, features = load_model(model_name)

    row = {f: 0 for f in features}
    row.update(feature_vector)

    active = [f for f in features if row[f] != 0 and f != 'total_severity']
    
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        ranked = sorted(active, key=lambda f: importances[features.index(f)], reverse=True)
    elif hasattr(clf, 'coef_'):
        importances = np.abs(clf.coef_).mean(axis=0)
        ranked = sorted(active, key=lambda f: importances[features.index(f)], reverse=True)
    else:
        ranked = active
        
    top = ranked[:top_n]
    
    # Fallback to formatting the raw name dynamically
    return [f.replace('_', ' ').capitalize() for f in top]

def evaluate(clf, le, X_test, y_test, model_name="Model"):
    print(f"--- Evaluating {model_name} ---")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    print()

if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Train ML models on clean, noisy, or combined symptom datasets.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["clean", "low_noise", "high_noise", "combined"], 
        default="clean",
        help="Select which dataset variant to train on."
    )
    args = parser.parse_args()

    print(f"--- Starting training pipeline using '{args.dataset}' dataset ---")
    X, y, features = load_data(args.dataset)

    print("Encoding labels...")
    y_encoded, le = encode_labels(y)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)

    # Determine file suffix to avoid overwriting models unless intended
    suffix = "" if args.dataset == "clean" else f"_{args.dataset}"

    print("Training Random Forest...")
    rf_clf = train_random_forest(X_train, y_train)
    evaluate(rf_clf, le, X_test, y_test, "Random Forest")
    save_model(rf_clf, le, features, f"random_forest{suffix}.pkl")

    print("Training Gradient Boosting...")
    gb_clf = train_gradient_boosting(X_train, y_train)
    evaluate(gb_clf, le, X_test, y_test, "Gradient Boosting")
    save_model(gb_clf, le, features, f"gradient_boosting{suffix}.pkl")

    print("Training Logistic Regression...")
    lr_clf = train_logistic_regression(X_train, y_train)
    evaluate(lr_clf, le, X_test, y_test, "Logistic Regression")
    save_model(lr_clf, le, features, f"logistic_regression{suffix}.pkl")
