import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Import your existing pipeline functions
from ml_trainer import load_data, encode_labels, split_data, BASE_DIR

# Create a folder to store the logs for your teacher
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def save_tuning_results(grid_search, model_name):
    """
    Extracts the detailed cross-validation results and exports them to a CSV.
    """
    results_df = pd.DataFrame(grid_search.cv_results_)

    params_df = pd.json_normalize(results_df['params'])

    summary_df = pd.DataFrame({
        'Model': model_name,
        'Mean_F1_Score': results_df['mean_test_score'],
        'Std_Dev_F1': results_df['std_test_score'],  # Shows stability across folds
        'Rank': results_df['rank_test_score'],
        'Mean_Fit_Time_sec': results_df['mean_fit_time']
    })

    # Combine the metrics with the specific parameters used
    final_df = pd.concat([summary_df, params_df], axis=1)

    # Sort by Rank so the best combinations are at the top
    final_df = final_df.sort_values('Rank')

    # Save to CSV
    filename = f"{model_name.replace(' ', '_').lower()}_tuning_results.csv"
    file_path = os.path.join(LOG_DIR, filename)
    final_df.to_csv(file_path, index=False)

    print(f"--> Saved detailed logs for {model_name} to: {file_path}\n")
    return final_df


def tune_models():
    print("Loading data for tuning...\n")
    X, y, features = load_data("combined")
    y_encoded, le = encode_labels(y)
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)


    # ==========================================
    # 3. Logistic Regression Tuning
    # ==========================================
    print("--- Tuning Logistic Regression ---")
    lr_param_grid = {
        'C': [0.1, 1.0, 10.0],
        'solver': ['lbfgs', 'saga']
    }
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr_grid = GridSearchCV(lr, lr_param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    lr_grid.fit(X_train, y_train)

    save_tuning_results(lr_grid, "Logistic Regression")


if __name__ == "__main__":
    tune_models()
    print("Tuning complete.")