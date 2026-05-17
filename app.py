from src.triage_logic import get_triage
from src.routing import find_nearest_hospital

triage = get_triage(top_diagnosis)
hospital = find_nearest_hospital(city, triage["care_types"])
