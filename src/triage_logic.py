import json
import os

RULES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'triage_rules.json')


def load_rules():
    with open(RULES_PATH, 'r') as file:
        return json.load(file)


def determine_urgency(predicted_disease: str) -> tuple:
    """
    Returns a tuple containing (urgency_level, hex_color)
    """
    rules = load_rules()

    # Defaults if the disease somehow isn't in the JSON
    DEFAULT_URGENCY = 3
    DEFAULT_COLOR = "#A3A3A3"  # Neutral grey

    if predicted_disease in rules:
        return rules[predicted_disease]["level"], rules[predicted_disease]["color"]

    return DEFAULT_URGENCY, DEFAULT_COLOR


# Test
if __name__ == "__main__":
    print("Running Triage Logic Tests...\n")

    test_1 = "Ataque Cardíaco"
    lvl_1, col_1 = determine_urgency(test_1)
    print(f"Condition: '{test_1}' -> Urgency Level: {lvl_1} | Color: {col_1}")

    test_2 = "Constipação Comum"
    lvl_2, col_2 = determine_urgency(test_2)
    print(f"Condition: '{test_2}' -> Urgency Level: {lvl_2} | Color: {col_2}")

    test_3 = "Alien Virus"
    lvl_3, col_3 = determine_urgency(test_3)
    print(f"Condition: '{test_3}' -> Urgency Level: {lvl_3} | Color: {col_3} (Defaults)")