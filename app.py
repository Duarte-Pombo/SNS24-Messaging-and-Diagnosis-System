import sys
import os

# Ensure the root directory is in the path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.nlp_extractor import extract_symptoms
from src.ml_trainer import predict_top3, get_differentiating_symptoms, MODEL_DIR

def select_model():
    """Scans the models directory and prompts the user to select one."""
    if not os.path.exists(MODEL_DIR):
        print(f"\n[!] Model directory not found at {MODEL_DIR}")
        return None
        
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
    
    if not models:
        print("\n[!] No .pkl models found. Please run 'python src/ml_trainer.py' first.")
        return None
        
    print("\n=== Available Models ===")
    for i, model_name in enumerate(models, 1):
        print(f"{i}. {model_name}")
        
    while True:
        try:
            choice = input(f"\nSelect a model (1-{len(models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    print("\n=== SNS24 prot. 1 ===")
    
    # 1. Ask the user which model to run
    selected_model = select_model()
    if not selected_model:
        return
        
    print(f"\n[!] Loaded '{selected_model}' successfully.")
    
    # 2. Get Symptoms
    patient_text = input("\nDescribe the symptoms: ")

    print("\n[1] extracting symptoms ...")
    extracted_symptoms = extract_symptoms(patient_text)

    if not extracted_symptoms:
        print("no symptom recognized")
        return

    print(f"detected symptoms: {', '.join(extracted_symptoms)}")

    # Data formatting for the ML model
    feature_vector = {sym.replace(" ", "_"): 1 for sym in extracted_symptoms}

    try:
        # Initial Prediction
        print("\n[2] calculating initial diagnosis...")
        predictions = predict_top3(feature_vector, model_name=selected_model)
        top_prob = predictions[0][1]
        
        # Track asked questions so we don't repeat them if the user says "no"
        asked_symptoms = set()
        question_rounds = 0
        MAX_ROUNDS = 5 # Safety limit: stop after 10 total questions so the user isn't stuck forever
        
        # Keep asking until confidence hits 70% or we hit our round limit
        while top_prob < 70.0 and question_rounds < MAX_ROUNDS:
            print(f"\n[!] Diagnosis confidence is {top_prob:.1f}%. Generating triage questions...")
            
            # Fetch just 2 highly relevant questions per round based on current top predictions
            questions = get_differentiating_symptoms(
                feature_vector, 
                model_name=selected_model, 
                max_questions=2, 
                asked_symptoms=asked_symptoms
            )
            
            if not questions:
                print("[-] No more distinguishing symptoms to ask about.")
                break
                
            print("=== Please answer a few follow-up questions ===")
            for sym in questions:
                asked_symptoms.add(sym) # Mark as asked
                readable_sym = sym.replace("_", " ")
                
                ans = input(f"Are you experiencing '{readable_sym}'? (y/n): ").strip().lower()
                if ans in ['y', 'yes', 'sim', 's']:
                    feature_vector[sym] = 1
            
            question_rounds += 1
            
            # Recalculate with new data
            print(f"\n[3] recalculating diagnosis (Round {question_rounds})...")
            predictions = predict_top3(feature_vector, model_name=selected_model)
            top_prob = predictions[0][1]

        # Output final results
        print("\n=== final diagnosis results ===")
        for i, (condition, prob, _) in enumerate(predictions, 1):
            print(f"{i}. {condition} ({prob:.1f}%)")
            
    except FileNotFoundError:
        print("\n error: model not found. execute 'python src/ml_trainer.py'")

if __name__ == "__main__":
    main()
