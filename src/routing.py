"""Haversine distance + nearest valid hospital lookup."""

import os
from functools import lru_cache
from math import radians, sin, cos, sqrt, atan2

import pandas as pd

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOSPITALS_PATH = os.path.join(BASE_DIR, "data", "hospitals.csv")

EARTH_RADIUS_KM = 6371.0

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * atan2(sqrt(a), sqrt(1 - a))


@lru_cache(maxsize=1)
def load_hospitals() -> pd.DataFrame:
    df = pd.read_csv(HOSPITALS_PATH)

    coords = df["Coordinates"].str.split(",", n=1, expand=True)
    df["lat"] = coords[0].astype(float)
    df["lon"] = coords[1].astype(float)

    tags = df["Specialty Tags"].fillna("").str.lower()
    excluded = tags.apply(lambda s: any(k in s for k in EXCLUDED_KEYWORDS))
    return df[~excluded].reset_index(drop=True)


def _rank_by_distance(df: pd.DataFrame, user_lat: float, user_lon: float) -> pd.DataFrame:
    df = df.copy()
    df["distance_km"] = df.apply(
        lambda r: haversine(user_lat, user_lon, r["lat"], r["lon"]), axis=1
    )
    return df.sort_values("distance_km")


def _to_records(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "name": row["Hospital Name"],
            "care_type": row["Care Type"],
            "specialty_tags": row["Specialty Tags"],
            "lat": row["lat"],
            "lon": row["lon"],
            "distance_km": round(row["distance_km"], 2),
        }
        for _, row in df.iterrows()
    ]


def find_nearest_hospital(
    user_lat: float,
    user_lon: float,
    care_types: list[str],
    specialty_keywords: list[str] | None = None,
    max_results: int = 1,
) -> list[dict]:
    """Find the nearest valid hospital(s) for the user.

    1. Prefer facilities whose Specialty Tags match any clinical keyword AND
       whose Care Type is in care_types (best clinical fit).
    2. Fall back through care_types in order, ignoring specialty.
    """
    df = load_hospitals()
    specialty_keywords = specialty_keywords or []

    if specialty_keywords:
        tags = df["Specialty Tags"].fillna("").str.lower()
        specialty_mask = tags.apply(lambda s: any(k in s for k in specialty_keywords))
        type_mask = df["Care Type"].isin(care_types)
        preferred = df[specialty_mask & type_mask]
        if not preferred.empty:
            return _to_records(_rank_by_distance(preferred, user_lat, user_lon).head(max_results))

    for care_type in care_types:
        subset = df[df["Care Type"] == care_type]
        if subset.empty:
            continue
        return _to_records(_rank_by_distance(subset, user_lat, user_lon).head(max_results))

    return []
