import math
import pandas as pd
from pathlib import Path

# Adjust paths based on your structure
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the distance between two coordinates in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def load_facilities() -> pd.DataFrame:
    """Loads and cleans the hospitals dataset."""
    csv_path = DATA_DIR / "hospitals.csv"
    if not csv_path.exists():
        return pd.DataFrame()
        
    df = pd.read_csv(csv_path)
    
    # Parse the "Coordinates" string into float lat/lon columns
    def parse_coords(coord_str):
        try:
            parts = str(coord_str).replace('"', '').split(',')
            if len(parts) == 2:
                return float(parts[0].strip()), float(parts[1].strip())
        except Exception:
            pass
        return None, None

    df[['lat', 'lon']] = pd.DataFrame(df['Coordinates'].apply(parse_coords).tolist(), index=df.index)
    return df.dropna(subset=['lat', 'lon'])

def get_symptom_severity(symptoms: list[str]) -> int:
    """Calculates the maximum severity score from the extracted symptoms."""
    # Look for the file in the symptoms_dataset subfolder or directly in data/
    csv_path = DATA_DIR / "symptoms_dataset" / "Symptom-severity_pt.csv"
    if not csv_path.exists():
        csv_path = DATA_DIR / "Symptom-severity_pt.csv"
        if not csv_path.exists():
            return 0  # Fallback if file is missing
            
    df = pd.read_csv(csv_path)
    # Create a fast lookup dictionary { "febre_alta": 7, "arrepios": 3, ... }
    severity_map = dict(zip(df['Symptom'].str.lower(), df['weight']))
    
    max_severity = 0
    for sym in symptoms:
        formatted_sym = sym.replace(" ", "_").lower()
        sev = severity_map.get(formatted_sym, 0)
        if sev > max_severity:
            max_severity = sev
            
    return max_severity

def get_nearest_facilities(user_lat: float, user_lon: float, extracted_symptoms: list[str], top_diagnosis: str = "", max_results: int = 3) -> list[dict]:
    """Finds the most adequate and closest health centers based on symptom severity."""
    df = load_facilities()
    if df.empty:
        return []
        
    # 1. Aggressively clean strings to prevent hidden whitespace/case mismatch bugs
    df['Care Type'] = df['Care Type'].astype(str).str.strip().str.lower()
    clean_diagnosis = str(top_diagnosis).strip().lower()
    
    # Calculate distances
    df['distance_km'] = df.apply(
        lambda row: haversine(user_lat, user_lon, row['lat'], row['lon']), 
        axis=1
    )
    
    # 2. Check Severity
    severity_score = get_symptom_severity(extracted_symptoms)
    
    # Add any other severe classes your model outputs here (lowercase)
    severe_conditions = ["ataque cardíaco", "avc", "enfarte", "hemorragia"]
    
    is_emergency = (severity_score >= 5) or (clean_diagnosis in severe_conditions)
    
    # 3. Strict Routing Heuristic
    if is_emergency:
        # EMERGENCY: Only Hospitals or Urgências
        df_filtered = df[df['Care Type'].isin(['hospital', 'maternity', 'urgências'])]
    else:
        # NON-EMERGENCY: Clinics and Health Centers
        df_filtered = df[df['Care Type'].isin(['health center', 'clinic'])]
        
    # 4. Safe Fallbacks
    if df_filtered.empty:
        if is_emergency:
            # If it's an emergency and no hospitals are found, DO NOT route to a clinic.
            # We return empty so the UI doesn't give dangerous advice.
            return []
        else:
            # If it's mild and no clinics are found, it's safe to fall back to a hospital
            df_filtered = df
            
    # Restore original capitalization for the UI
    df_filtered['Care Type'] = df_filtered['Care Type'].str.title()
    
    df_sorted = df_filtered.sort_values('distance_km')
    return df_sorted.head(max_results).to_dict('records')
