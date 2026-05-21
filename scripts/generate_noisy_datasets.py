import pandas as pd
import numpy as np
import os

def generate_noisy_datasets():
    # Set paths relative to the scripts/ directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '../data/symptoms_dataset')
    
    binary_dataset_path = os.path.join(data_dir, 'dataset_binary_pt.csv')
    severity_path = os.path.join(data_dir, 'Symptom-severity_pt.csv')
    
    low_noise_out_path = os.path.join(data_dir, 'dataset_binary_pt_low_noise.csv')
    high_noise_out_path = os.path.join(data_dir, 'dataset_binary_pt_high_noise.csv')
    
    # Load datasets
    print("Loading datasets...")
    try:
        df = pd.read_csv(binary_dataset_path)
        severity_df = pd.read_csv(severity_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure this script is run from the 'scripts/' directory.")
        return

    # Map symptoms to their severity weights
    severity_dict = dict(zip(severity_df['Symptom'], severity_df['weight']))
    
    # Exclude non-feature columns
    exclude_cols = {'diagnosis', 'prognóstico'}
    all_symptoms = [c for c in df.columns if c not in exclude_cols]

    # Categorize symptoms into mild and severe
    # If a symptom is missing from severity_dict, assume a safe default weight of 1 (mild)
    mild_cols = [col for col in all_symptoms if severity_dict.get(col, 1) <= 4]
    severe_cols = [col for col in all_symptoms if severity_dict.get(col, 1) > 4]
    
    print(f"Total symptoms: {len(all_symptoms)}")
    print(f"Mild symptoms (Primary noise targets): {len(mild_cols)}")
    print(f"Severe symptoms (Baseline noise targets): {len(severe_cols)}")
    
    def apply_dual_noise(original_df, mild_symptoms, severe_symptoms, mild_prob, severe_prob):
        """
        Apply higher probability noise to mild symptoms and a baseline 
        probability noise to severe symptoms to prevent the model 
        from anchoring to perfect indicators.
        """
        noisy_df = original_df.copy()
        
        # Apply primary noise to mild symptoms
        for col in mild_symptoms:
            flip_mask = np.random.rand(len(noisy_df)) < mild_prob
            noisy_df.loc[flip_mask, col] = 1 - noisy_df.loc[flip_mask, col]

        # Apply baseline error noise to severe symptoms
        for col in severe_symptoms:
            flip_mask = np.random.rand(len(noisy_df)) < severe_prob
            noisy_df.loc[flip_mask, col] = 1 - noisy_df.loc[flip_mask, col]
            
        return noisy_df

    # Generate Low Noise Dataset
    # Mild symptoms: 12.5% | Severe symptoms: 3%
    low_mild_prob = 0.125
    low_severe_prob = 0.03
    print(f"\nGenerating low noise dataset ({low_mild_prob*100}% mild noise, {low_severe_prob*100}% severe error)...")
    df_low_noise = apply_dual_noise(df, mild_cols, severe_cols, low_mild_prob, low_severe_prob)
    df_low_noise.to_csv(low_noise_out_path, index=False)
    print(f"Saved to: {low_noise_out_path}")

    # Generate High Noise Dataset
    # Mild symptoms: 27.5% | Severe symptoms: 5%
    high_mild_prob = 0.275
    high_severe_prob = 0.05
    print(f"Generating high noise dataset ({high_mild_prob*100}% mild noise, {high_severe_prob*100}% severe error)...")
    df_high_noise = apply_dual_noise(df, mild_cols, severe_cols, high_mild_prob, high_severe_prob)
    df_high_noise.to_csv(high_noise_out_path, index=False)
    print(f"Saved to: {high_noise_out_path}")
    
    print("\nDataset generation complete!")

if __name__ == "__main__":
    # Seed the random number generator to ensure reproducibility
    np.random.seed(42)
    generate_noisy_datasets()
