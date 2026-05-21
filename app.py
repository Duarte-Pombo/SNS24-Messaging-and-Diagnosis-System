import sys
import os

# Ensure the root directory is in the path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.nlp_extractor import extract_symptoms
from src.ml_trainer import predict_top3

def main():
    print("\n=== SNS24 prot. 1 ===")
    patient_text = input("Describe the symptoms: ")

    # NLP extraction
    print("\n[1] extracting symptoms ...")
    extracted_symptoms = extract_symptoms(patient_text)

    if not extracted_symptoms:
        print("no symptom recognized")
        return

    print(f"detected symptoms: {', '.join(extracted_symptoms)}")

    # Data formatting for the ML model
    feature_vector = {sym.replace(" ", "_"): 1 for sym in extracted_symptoms}

    # Model prediction
    print("\n[2] calculating possible diagnosis...")
    try:
        # predict_top3 returns [(condition, probability, placeholder_triage_val)]
        predictions = predict_top3(feature_vector)
        
        print("\n=== diagnosis results ===")
        for i, (condition, prob, _) in enumerate(predictions, 1):
            print(f"{i}. {condition} ({prob:.1f}%)")
            
    except FileNotFoundError:
        print("\n error: model not found. execute 'python src/ml_trainer.py'")

if __name__ == "__main__":
    main()
