import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


def generate_comparison_table():
    # Find all the CSV files
    csv_files = glob.glob("*_tuning_results.csv")
    if not csv_files:
        csv_files = glob.glob("logs/*_tuning_results.csv")

    if not csv_files:
        print("No CSV files found. Make sure they are in the folder or 'logs' directory.")
        return

    best_rows = []

    for file in csv_files:
        df = pd.read_csv(file)

        # Grab only Rank 1
        top_row = df[df['Rank'] == 1].copy()

        # Clean up the model name
        model_name = os.path.basename(file).replace('_tuning_results.csv', '').replace('_', ' ').title()
        top_row['Algorithm'] = model_name

        # Extract only the metrics we care about comparing
        best_rows.append(top_row[['Algorithm', 'Mean_F1_Score', 'Std_Dev_F1', 'Mean_Fit_Time_sec']])

    # Combine them all into one DataFrame
    comparison_df = pd.concat(best_rows, ignore_index=True)

    # Sort by the highest F1-Score
    comparison_df = comparison_df.sort_values(by='Mean_F1_Score', ascending=False)

    # Clean up the names for presentation
    comparison_df = comparison_df.rename(columns={
        'Mean_F1_Score': 'Peak F1-Score',
        'Std_Dev_F1': 'Std Dev (Stability)',
        'Mean_Fit_Time_sec': 'Training Time (sec)'
    })

    # --- THE FIX: Force convert to numeric before rounding ---
    comparison_df['Peak F1-Score'] = pd.to_numeric(comparison_df['Peak F1-Score'], errors='coerce').round(4)
    comparison_df['Std Dev (Stability)'] = pd.to_numeric(comparison_df['Std Dev (Stability)'], errors='coerce').round(4)
    comparison_df['Training Time (sec)'] = pd.to_numeric(comparison_df['Training Time (sec)'], errors='coerce').round(2)

    # 1. Print it to the console as Markdown text
    print("\n### Ultimate Model Comparison (Rank 1 Only)")
    print(comparison_df.to_markdown(index=False))

    # 2. Save it as a beautiful PNG image
    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=comparison_df.values,
        colLabels=comparison_df.columns,
        loc='center',
        cellLoc='center'
    )

    table.scale(1, 2.0)
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Style the header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f0f0f0')

    plt.title('Overall Comparison of Optimized Algorithms', pad=20, weight='bold', fontsize=14)

    output_filename = "best_models_comparison_table.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n--> Saved comparison table image to: {output_filename}")
    plt.close()


if __name__ == "__main__":
    generate_comparison_table()