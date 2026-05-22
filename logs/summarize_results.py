import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


def generate_table_pngs():
    # Find all the CSV files
    csv_files = glob.glob("*_tuning_results.csv")
    if not csv_files:
        csv_files = glob.glob("logs/*_tuning_results.csv")

    if not csv_files:
        print("No CSV files found.")
        return

    for file in csv_files:
        df = pd.read_csv(file)

        model_name = os.path.basename(file).replace('_tuning_results.csv', '').replace('_', ' ').title()

        # Take top 3
        top_3 = df.head(3).copy()

        top_3 = top_3.fillna("None")

        if 'Model' in top_3.columns:
            top_3 = top_3.drop(columns=['Model'])
        if 'Mean_Fit_Time_sec' in top_3.columns:
            top_3 = top_3.drop(columns=['Mean_Fit_Time_sec'])

        top_3['Mean_F1_Score'] = top_3['Mean_F1_Score'].round(4)
        top_3['Std_Dev_F1'] = top_3['Std_Dev_F1'].round(4)

        cols = list(top_3.columns)
        cols.insert(0, cols.pop(cols.index('Rank')))
        top_3 = top_3[cols]

        rename_map = {
            'min_samples_split': 'Min Split',
            'n_estimators': 'Estimators',
            'max_depth': 'Max Depth',
            'learning_rate': 'Learning Rate',
            'Mean_F1_Score': 'F1-Score',
            'Std_Dev_F1': 'Std Dev',
            'solver': 'Solver',
            'C': 'C (Regularization)'
        }
        top_3 = top_3.rename(columns=rename_map)

        # --- Plotting the Table as an Image ---
        fig, ax = plt.subplots(figsize=(10, 2.5))  # Made slightly taller to give breathing room

        ax.axis('tight')
        ax.axis('off')

        # Create the table
        table = ax.table(
            cellText=top_3.values,
            colLabels=top_3.columns,
            loc='center',
            cellLoc='center'
        )

        table.scale(1, 2.0)  # Stretch rows to make them taller
        table.auto_set_font_size(False)
        table.set_fontsize(11)  # Slightly bigger font

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f0f0f0')

        plt.title(f'Top 3 Hyperparameter Configurations: {model_name}', pad=20, weight='bold', fontsize=14)

        output_filename = f"{model_name.replace(' ', '_').lower()}_table.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"--> Saved pristine table image to: {output_filename}")

        plt.close()


if __name__ == "__main__":
    generate_table_pngs()