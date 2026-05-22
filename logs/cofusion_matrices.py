import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.ml_trainer import load_data, encode_labels, split_data, MODEL_DIR


def export_cm_pngs():
    print("Loading data...")
    # Load the same data used for training
    X, y, features = load_data("combined")
    y_encoded, le = encode_labels(y)
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)

    classes = le.classes_

    # Models to evaluate
    models_to_plot = {
        "Random Forest": "random_forest.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
        "Logistic Regression": "logistic_regression.pkl"
    }

    for model_name, filename in models_to_plot.items():
        model_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(model_path):
            print(f"Skipping {model_name}: {filename} not found.")
            continue

        print(f"Generating Confusion Matrix for {model_name}...")

        # Load model and predict
        data = joblib.load(model_path)
        clf = data["model"]
        y_pred = clf.predict(X_test)

        # Generate raw confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Normalize the confusion matrix (percentages instead of raw counts)
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

        # --- Plotting ---
        # Make the figure large enough to fit all medical conditions
        plt.figure(figsize=(16, 12))

        # Create a beautiful heatmap
        sns.heatmap(
            cm_norm,
            annot=False,  # Set to True if you want numbers inside the boxes, but with 40 classes it gets messy
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            linewidths=.5,
            linecolor='lightgray'
        )

        plt.title(f'Normalized Confusion Matrix: {model_name}', fontsize=18, pad=20, weight='bold')
        plt.ylabel('True Medical Condition', fontsize=14, weight='bold')
        plt.xlabel('Predicted Medical Condition', fontsize=14, weight='bold')

        # Rotate x-labels so they don't overlap
        plt.xticks(rotation=90, fontsize=9)
        plt.yticks(rotation=0, fontsize=9)

        plt.tight_layout()

        output_file = os.path.join("", f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"--> Saved to: {output_file}")
        plt.close()


if __name__ == "__main__":
    export_cm_pngs()