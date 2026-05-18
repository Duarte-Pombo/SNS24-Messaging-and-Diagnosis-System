import json
import os

RULES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'triage_rules.json')


def load_rules():
    with open(RULES_PATH, 'r') as file:
        return json.load(file)


def determine_urgency(predicted_disease: str) -> int:
    rules = load_rules()
    DEFAULT_URGENCY = 3

    if predicted_disease in rules:
        return rules[predicted_disease]["level"]

    return DEFAULT_URGENCY


# Test
if __name__ == "__main__":
    print("Running Triage Logic Tests...\n")

    test_1 = "Enfarte do Miocárdio"
    result_1 = determine_urgency(test_1)
    print(f"Condition: '{test_1}' -> Urgency Level: {result_1}")

    test_2 = "Constipação"
    result_2 = determine_urgency(test_2)
    print(f"Condition: '{test_2}' -> Urgency Level: {result_2}")

    test_3 = "Alien Virus"
    result_3 = determine_urgency(test_3)
    print(f"Condition: '{test_3}' -> Urgency Level: {result_3} (Should be the default)")