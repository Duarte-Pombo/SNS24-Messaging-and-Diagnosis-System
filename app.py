# frontend - StreamLit CURRENT IS A PLACEHOLDER FOR TESTING
import sys
import os

# Ensure the root directory is in the path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.nlp_extractor import extract_symptoms
from src.ml_trainer import predict_top3, EXPECTED_FEATURES

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

    # data formating for the ml model
    ml_ready_symptoms = [sym.replace(" ", "_") for sym in extracted_symptoms]

    # initialize feature vector with zeros
    feature_vector = {feature: 0 for feature in EXPECTED_FEATURES}

    # set detected symptoms to 1
    for sym in ml_ready_symptoms:
        if sym in feature_vector:
            feature_vector[sym] = 1
        else:
            print(f"warning: symptom '{sym}' ignored (not included in the model training ).")

    # collect mandatory demographics
    print("\n[2] additional information")
    try:
        # adjust these inputs based on how your dataset actually encoded them
        age = int(input("age: "))
        feature_vector['age_group'] = age 
        
        gender = int(input("gender (0 = Male, 1 = Female): "))
        feature_vector['gender'] = gender
    except ValueError:
        print("warning: invalid input, usign 0 as default.")

    # model prediction
    print("\n[3] calculating possible diagnosis...")
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
