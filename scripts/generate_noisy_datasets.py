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
    
    # Isolate mild symptoms to simulate realistic user errors
    # Select symptoms with a severity weight <= 4 (e.g., fatigue, headache)
    # Severe symptoms (weight > 4) remain unaffected
    flippable_cols = [col for col in df.columns if col in severity_dict and severity_dict[col] <= 4]
    
    print(f"Total symptoms in dataset: {len(df.columns) - 1}")
    print(f"Flippable 'realistic' symptoms (severity <= 4): {len(flippable_cols)}")
    
    def apply_noise(original_df, columns_to_flip, noise_prob):
        """
        Simulate user error by flipping bits (0 -> 1 and 1 -> 0) 
        for the specified realistic columns based on the given probability.
        """
        noisy_df = original_df.copy()
        
        for col in columns_to_flip:
            # Generate a random boolean mask for the current column
            # A True value indicates the row value should be flipped
            flip_mask = np.random.rand(len(noisy_df)) < noise_prob
            
            # Apply the mathematical flip: 1 - current_value 
            # (1 becomes 0 "forgot to mention", 0 becomes 1 "mistakenly mentioned")
            noisy_df.loc[flip_mask, col] = 1 - noisy_df.loc[flip_mask, col]
            
        return noisy_df

    # Generate Low Noise Dataset (10-15%)
    # Set 12.5% as the median probability for low noise
    low_noise_prob = 0.125
    print(f"\nGenerating low noise dataset (~{low_noise_prob*100}% noise on realistic symptoms)...")
    df_low_noise = apply_noise(df, flippable_cols, low_noise_prob)
    df_low_noise.to_csv(low_noise_out_path, index=False)
    print(f"Saved to: {low_noise_out_path}")

    # Generate High Noise Dataset (25-30%)
    # Set 27.5% as the median probability for high noise
    high_noise_prob = 0.275
    print(f"Generating high noise dataset (~{high_noise_prob*100}% noise on realistic symptoms)...")
    df_high_noise = apply_noise(df, flippable_cols, high_noise_prob)
    df_high_noise.to_csv(high_noise_out_path, index=False)
    print(f"Saved to: {high_noise_out_path}")
    print("\nDataset generation complete!")

if __name__ == "__main__":
    # Seed the random number generator to ensure reproducibility
    np.random.seed(42)
    generate_noisy_datasets()
