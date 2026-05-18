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

