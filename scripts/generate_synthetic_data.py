import pandas as pd
import numpy as np
import os

def generate_synthetic_patients():
    # Set paths relative to the scripts/ directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '../data/symptoms_dataset')
    
    binary_dataset_path = os.path.join(data_dir, 'dataset_binary_pt.csv')
    severity_path = os.path.join(data_dir, 'Symptom-severity_pt.csv')
    synthetic_out_path = os.path.join(data_dir, 'dataset_synthetic_pt.csv')
    
    print("Loading base deterministic dataset...")
    try:
        df = pd.read_csv(binary_dataset_path)
        severity_df = pd.read_csv(severity_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Map symptoms to their severity weights
    severity_dict = dict(zip(severity_df['Symptom'], severity_df['weight']))
    
    exclude_cols = {'diagnosis', 'prognóstico'}
    features = [c for c in df.columns if c not in exclude_cols]

    # Categorize symptoms
    mild_cols = [c for c in features if severity_dict.get(c, 1) <= 4]
    severe_cols = [c for c in features if severity_dict.get(c, 1) > 4]
    
    # Extract the "perfect" base profile for each disease
    disease_profiles = df.groupby('diagnosis')[features].mean()
    
    synthetic_rows = []
    # Generate 300 unique synthetic patients per disease (41 diseases * 300 = 12,300 patients)
    samples_per_disease = 300 
    
    print(f"Generating {samples_per_disease} synthetic patients for each of the {len(disease_profiles)} diseases...")

    for diagnosis, profile in disease_profiles.iterrows():
        prob_vector = profile.copy()
        
        # Build the probability distribution for this specific disease
        for col in mild_cols:
            if prob_vector[col] > 0.5: # It's a core symptom of the disease
                prob_vector[col] = 0.85 # 15% chance the patient forgets to mention it
            else: # It's NOT a core symptom
                prob_vector[col] = 0.15 # 15% chance of background noise (e.g., random headache)
                
        for col in severe_cols:
            if prob_vector[col] > 0.5: # Core severe symptom
                prob_vector[col] = 0.95 # 5% chance it doesn't present or wasn't reported
            else:
                prob_vector[col] = 0.02 # 2% chance of mistaken severe reporting
        
        # Generate patients mathematically using the probability vector
        # np.random.rand generates a matrix of random numbers between 0 and 1.
        # If the random number is less than our probability, the symptom becomes 1 (True).
        random_matrix = np.random.rand(samples_per_disease, len(features))
        prob_matrix = np.tile(prob_vector.values, (samples_per_disease, 1))
        
        simulated_patients = (random_matrix < prob_matrix).astype(int)
        
        # Convert to DataFrame and append
        sim_df = pd.DataFrame(simulated_patients, columns=features)
        sim_df['diagnosis'] = diagnosis
        synthetic_rows.append(sim_df)

    # Combine all generated patients
    synthetic_dataset = pd.concat(synthetic_rows, ignore_index=True)
    
    # Shuffle the dataset thoroughly so diseases are mixed during ML training
    synthetic_dataset = synthetic_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    synthetic_dataset.to_csv(synthetic_out_path, index=False)
    
    print(f"Successfully generated {len(synthetic_dataset)} unique synthetic patient records!")
    print(f"Saved to: {synthetic_out_path}")

if __name__ == "__main__":
    # Seed for reproducibility
    np.random.seed(42)
    generate_synthetic_patients()
