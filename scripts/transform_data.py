# convert the dataset into a binary disease-symptom table 
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath((__file__))))
DATASET_PATH = os.path.join(BASE_DIR, "data/symptoms_dataset/", "dataset_pt.csv")
SEVERITY_PATH = os.path.join(BASE_DIR, "data/symptoms_dataset/", "Symptom-severity_pt.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/symptoms_dataset/", "dataset_binary_pt.csv")

def transform_dataset():
    
    print("loading files...")
    df_patients = pd.read_csv(DATASET_PATH)
    df_severity = pd.read_csv(SEVERITY_PATH)

    # fetch the list of symptoms 
    official_symptoms = df_severity['Symptom'].apply(lambda x: str(x).strip()).tolist()
    
    print(f"Loaded {len(official_symptoms)} symptoms")

    # clean the symptom strings in the patient dataset
    for col in df_patients.columns[1:]:
        df_patients[col] = df_patients[col].apply(lambda x: str(x).strip() if pd.notna(x) else x)

    # create the empty dataframe using the official symptoms as columns
    ml_df = pd.DataFrame(0, index=df_patients.index, columns=['diagnosis'] + official_symptoms)
    ml_df['diagnosis'] = df_patients['Disease']

    # populate the matrix
    print("generating symptom matrix...")
    for index, row in df_patients.iterrows():
        for col in df_patients.columns[1:]:
            symptom = row[col]
            if pd.notna(symptom):
                if symptom in ml_df.columns:
                    ml_df.at[index, symptom] = 1
                else:
                    print(f"warning: '{symptom}' not in severity file")

    # save
    ml_df.to_csv(OUTPUT_PATH, index=False)
    print(f"dataset saved in: {OUTPUT_PATH}")

if __name__ == "__main__":
    transform_dataset()

