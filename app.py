# frontend - StreamLit CURRENT IS A PLACEHOLDER FOR TESTING
import sys
import os

# Ensure the root directory is in the path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.nlp_extractor import extract_symptoms
from src.ml_trainer import predict_top3, EXPECTED_FEATURES
from src.triage_logic import determine_urgency


def get_integer_input(prompt_text: str, default_val: int = 0) -> int:
    """
    Safely asks the user for a number. If they type letters,
    it asks them again instead of crashing.
    """
    while True:
        user_input = input(prompt_text)
        if user_input.strip() == "":
            print(f"  -> Using default: {default_val}")
            return default_val
        try:
            return int(user_input)
        except ValueError:
            print(" Invalid input. Please enter a number.")

def main():
    print("\n=== SNS24 prot. 1 ===")
    print("Please provide the most accurate and complete description possible of your symptoms.")
    print("For pain, please indicate the location as precisely as possible.")
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

        # collect mandatory demographics and vitals
        print("\n[2] additional information")

        feature_vector['age_group'] = get_integer_input("age: ")

        feature_vector['gender'] = get_integer_input("gender (0 = Male, 1 = Female): ")

        feature_vector['duration'] = min(get_integer_input("duration of symptoms (in days): ", default_val=0),2)

        feature_vector['pain_intensity'] = get_integer_input("pain intensity (0 to 10): ", default_val=0)

    # model prediction
    print("\n[3] calculating possible diagnosis...")
    try:
        # predict_top3 returns [(condition, probability, placeholder_triage_val)]
        predictions = predict_top3(feature_vector)
        
        print("\n=== diagnosis results ===")
        for i, (condition, prob, _) in enumerate(predictions, 1):
            print(f"{i}. {condition} ({prob:.1f}%). Urgency: {determine_urgency(condition)}")
            # Na nossa app final deve ser associada uma cor ao número (1-azul, 2-verde, 3-amarelo, 4-laranja, 5-vermelho)
            
    except FileNotFoundError:
        print("\n error: model not found. execute 'python src/ml_trainer.py'")

if __name__ == "__main__":
    main()
