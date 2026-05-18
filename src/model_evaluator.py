import sys
import os
import pandas as pd

# allow relative imports if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.ml_trainer import load_data, encode_labels, split_data

def evaluate_models():
    print("loading data...")
    X, y = load_data()
    y_encoded, le = encode_labels(y)
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)

    # Define the models you want to test
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }

    results = []

    print("training and evaluating models - can take time...\n")
    for name, model in models.items():
        # train
        model.fit(X_train, y_train)
        
        # predict
        y_pred = model.predict(X_test)

        # calculate metrics 
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })

    # display results sorted by best F1-Score
    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    
    print("=== model performances ===")
    # format floats as percentages for readability
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        results_df[col] = (results_df[col] * 100).round(2).astype(str) + '%'
        
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    evaluate_models()
