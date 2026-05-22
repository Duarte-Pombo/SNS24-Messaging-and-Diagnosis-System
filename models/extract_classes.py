import pandas as pd
import os

csv_path = os.path.join(os.path.dirname(__file__), '..', 'data/symptoms_dataset', 'dataset_pt.csv')

try:
    df = pd.read_csv(csv_path)

    unique_diseases = df['Disease'].unique()

    print("-" * 40)
    for disease in unique_diseases:
        print(f'    "{disease}": {{"color": "TODO", "level": "TODO"}},')

except Exception as e:
    print(f"Could not read the CSV file: {e}")