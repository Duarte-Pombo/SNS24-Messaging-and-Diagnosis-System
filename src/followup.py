"""
Adaptive follow-up questionnaire.

When the first ML pass returns a top prediction with confidence < 90%,
this module selects up to 5 yes/no questions that best discriminate
between the top-3 candidate conditions, using per-condition symptom
frequencies computed from the training dataset.

The caller owns the loop state (current feature vector, round counter,
predictions). This module is stateless and pure.
"""

from functools import lru_cache

import pandas as pd

from src.ml_trainer import DATA_PATH, EXPECTED_FEATURES, FEATURE_LABELS_PT, TARGET


FOLLOWUP_THRESHOLD = 90.0  # top probability % below which follow-up triggers
MAX_QUESTIONS = 5          # questions per round
MAX_ROUNDS = 5             # caller enforces this

_NON_SYMPTOM_FEATURES = frozenset({"age_group", "gender", "duration", "pain_intensity"})

SYMPTOM_FEATURES = [f for f in EXPECTED_FEATURES if f not in _NON_SYMPTOM_FEATURES]


def needs_followup(predictions):
    """True when the top prediction probability is below FOLLOWUP_THRESHOLD."""
    if not predictions:
        return False
    top_prob = predictions[0][1]
    return top_prob < FOLLOWUP_THRESHOLD


@lru_cache(maxsize=1)
def _load_all_profiles():
    """
    Load the training CSV once and return {condition: Series(symptom -> mean)}
    for every diagnosis class. Cached for the lifetime of the process.
    """
    df = pd.read_csv(DATA_PATH)
    symptom_cols = [f for f in SYMPTOM_FEATURES if f in df.columns]
    return {
        cond: group[symptom_cols].mean()
        for cond, group in df.groupby(TARGET)
    }


def _profiles_for(conditions):
    """
    Return per-condition symptom-frequency Series for the given conditions.
    Unknown conditions get a uniform 0.5 profile so scoring degrades
    gracefully instead of crashing.
    """
    all_profiles = _load_all_profiles()
    symptom_cols = [f for f in SYMPTOM_FEATURES if f in next(iter(all_profiles.values())).index]
    fallback = pd.Series({col: 0.5 for col in symptom_cols})
    return {c: all_profiles.get(c, fallback) for c in conditions}


def _score_zero_features(feature_vector, profiles, top1, competitors, asked_features):
    """
    Score each currently-unreported symptom by how much it favours `top1`
    over the strongest competitor:
        score = freq(top1) - max(freq(c) for c in competitors)

    Features already in `asked_features` are excluded so the same question
    is not repeated across rounds (even when the user answered "no").

    Returns list of (feature_name, score) sorted descending by score,
    with alphabetical tie-breaking for determinism.
    """
    top1_profile = profiles.get(top1)
    if top1_profile is None:
        return []

    zero_features = [
        f for f in SYMPTOM_FEATURES
        if f not in asked_features
        and feature_vector.get(f, 0) == 0
        and f in top1_profile.index
    ]

    scores = []
    for feat in zero_features:
        freq_top1 = float(top1_profile.get(feat, 0.0))
        competitor_freqs = [
            float(profiles[c].get(feat, 0.0))
            for c in competitors
            if c in profiles
        ]
        max_competitor = max(competitor_freqs) if competitor_freqs else 0.0
        scores.append((feat, freq_top1 - max_competitor))

    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores


def get_followup_questions(feature_vector, predictions, asked_features=frozenset()):
    """
    Select up to MAX_QUESTIONS yes/no questions that best discriminate
    between the top-3 candidate conditions.

    `asked_features` is the set of feature keys already asked in earlier
    rounds; they are excluded so the same question is never repeated.

    Returns a list of dicts:
        {"feature": "fraqueza_facial", "question_pt": "Tem fraqueza facial?"}

    Returns [] when there are no unreported symptoms left to ask about.
    """
    if not predictions:
        return []

    top1 = predictions[0][0]
    competitors = [p[0] for p in predictions[1:]]
    all_conditions = [p[0] for p in predictions]

    profiles = _profiles_for(all_conditions)
    scored = _score_zero_features(
        feature_vector, profiles, top1, competitors, asked_features
    )

    return [
        {
            "feature": feat,
            "question_pt": f"Tem {FEATURE_LABELS_PT.get(feat, feat)}?",
        }
        for feat, _ in scored[:MAX_QUESTIONS]
    ]


def apply_answers(feature_vector, answers):
    """
    Merge yes/no answers into the feature vector. Returns a new dict;
    does not mutate the input. Values are clamped to {0, 1}; unknown
    feature keys are silently ignored.
    """
    updated = dict(feature_vector)
    for feat, val in answers.items():
        if feat in EXPECTED_FEATURES:
            updated[feat] = max(0, min(1, int(val)))
    return updated
