import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_all_tuning_results():
    sns.set_theme(style="whitegrid")

    csv_files = glob.glob("*_tuning_results.csv")
    if not csv_files:
        csv_files = glob.glob("logs/*_tuning_results.csv")

    for file in csv_files:
        df = pd.read_csv(file)
        model_name = os.path.basename(file).replace('_tuning_results.csv', '').replace('_', ' ').title()

        if "Random Forest" in model_name:
            df['max_depth'] = df['max_depth'].fillna('None (Unlimited)').astype(str)

            g = sns.relplot(
                data=df,
                x='n_estimators',
                y='Mean_F1_Score',
                hue='max_depth',
                col='min_samples_split',
                kind='line',
                marker='o',
                height=5,
                aspect=1,
                errorbar=None
            )

            g.fig.subplots_adjust(top=0.85)
            g.fig.suptitle('Random Forest: F1-Score by Estimators, Max Depth, and Min Samples Split', fontsize=16)
            g.set_axis_labels('Number of Estimators', 'Mean F1-Score')
            g.set_titles('Min Samples Split: {col_name}')

            output_filename = f"{model_name.replace(' ', '_').lower()}_optimization_chart.png"
            plt.savefig(output_filename, dpi=300)
            print(f"--> Saved detailed grid graph to: {output_filename}")
            plt.close()

        elif "Gradient Boosting" in model_name:
            plt.figure(figsize=(9, 6))
            g = sns.relplot(
                data=df,
                x='learning_rate',
                y='Mean_F1_Score',
                hue='n_estimators',
                col='max_depth',
                kind='line',
                marker='s',
                height=5,
                aspect=1,
                errorbar=None
            )
            g.set(xscale="log")
            g.fig.subplots_adjust(top=0.85)
            g.fig.suptitle('Gradient Boosting: F1-Score by Learning Rate, Estimators, and Max Depth', fontsize=16)
            g.set_axis_labels('Learning Rate (Log Scale)', 'Mean F1-Score')
            g.set_titles('Max Depth: {col_name}')

            output_filename = f"{model_name.replace(' ', '_').lower()}_optimization_chart.png"
            plt.savefig(output_filename, dpi=300)
            print(f"--> Saved beautiful graph to: {output_filename}")
            plt.close()

        elif "Logistic Regression" in model_name:
            plt.figure(figsize=(9, 6))
            sns.lineplot(data=df, x='C', y='Mean_F1_Score', hue='solver', marker='^', errorbar=None)
            plt.xscale('log')
            plt.title('Logistic Regression Optimization: F1-Score vs Regularization', fontsize=14)
            plt.xlabel('Inverse of Regularization Strength - C (Log Scale)', fontsize=12)
            plt.ylabel('Mean F1-Score (Cross-Validation)', fontsize=12)
            plt.legend(title='Solver')

            output_filename = f"{model_name.replace(' ', '_').lower()}_optimization_chart.png"
            plt.savefig(output_filename, dpi=300)
            print(f"--> Saved beautiful graph to: {output_filename}")
            plt.close()


if __name__ == "__main__":
    plot_all_tuning_results()