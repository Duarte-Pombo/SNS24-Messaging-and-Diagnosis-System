"""
Extracts recognised medical symptoms from a Portuguese string using
spaCy's PhraseMatcher.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path
from typing import Sequence

import spacy
from spacy.matcher import PhraseMatcher

# ---------------------------------------------------------------------------
# Column names that are NOT symptoms (metadata / target)
# ---------------------------------------------------------------------------
NON_SYMPTOM_COLS: frozenset[str] = frozenset(
    {"age_group", "gender", "duration", "pain_intensity", "diagnosis"}
)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_symptoms_from_csv(path: str | Path) -> list[str]:
    """
    Derive the symptom vocabulary from the header row of ../data/symptoms_data.csv.
    """

    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        headers = next(reader)  # only the header row is needed

    return [
        col.replace("_", " ")
        for col in headers
        if col not in NON_SYMPTOM_COLS
    ]


# ---------------------------------------------------------------------------
# Default symptom vocabulary
# ---------------------------------------------------------------------------

def _bootstrap_default_symptoms() -> list[str]:
    """Try to load from a co-located CSV; fall back to a minimal built-in list."""
    candidate = Path(__file__).parent / "../data/symptoms_data.csv"
    if candidate.exists():
        return load_symptoms_from_csv(candidate)
    # Minimal fallback so the module works without the CSV.
    return [
        "febre", "calafrios", "tosse", "falta de ar", "dor de garganta",
        "dor no peito", "fadiga", "náusea", "vômito", "diarreia",
        "dor abdominal", "espirros", "cefaleia", "tontura", "inchaço",
        "palpitações", "suores frios", "dificuldade falar", "visão turva",
        "confusão mental", "fraqueza facial", "rigidez abdominal",
        "perda apetite", "ardor ao urinar", "frequência urinária",
        "pieira", "expectoração", "dor ao respirar", "dor abdominal difusa",
        "dor abdominal qid", "sensibilidade luz", "cefaleia pulsátil",
        "cefaleia súbita", "irradiação braço", "dor localizada",
    ]


DEFAULT_SYMPTOMS: list[str] = _bootstrap_default_symptoms()


def set_default_symptoms(symptoms: Sequence[str]) -> None:
    """
    Replace the module-level default vocabulary and invalidate the cache.
    """

    global DEFAULT_SYMPTOMS, _matcher, _key_to_phrase
    DEFAULT_SYMPTOMS = list(symptoms)
    # Invalidate cache so the next extract_symptoms() call rebuilds.
    _matcher = None
    _key_to_phrase = {}


def _strip_accents(text: str) -> str:
    """Return *text* with diacritics removed (NFC-safe)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def build_matcher(
    nlp: spacy.language.Language,
    symptoms: Sequence[str],
    *,
    case_insensitive: bool = True,
    accent_insensitive: bool = True,
) -> tuple[PhraseMatcher, dict[str, str]]:
    """
    Compile a PhraseMatcher from *symptoms* and return it together with
    a mapping from the normalised match key back to the original phrase.
    """
    attr = "LOWER" if case_insensitive else "TEXT"
    matcher = PhraseMatcher(nlp.vocab, attr=attr)
    key_to_phrase: dict[str, str] = {}

    for phrase in symptoms:
        normalised = _strip_accents(phrase.lower()) if accent_insensitive else phrase
        key = normalised.replace(" ", "_")  # spaCy rule keys cannot contain spaces
        key_to_phrase[key] = phrase

        # Build both the original and accent-stripped patterns so the matcher
        # catches typed text regardless of whether the user used diacritics.
        patterns = [phrase]
        if accent_insensitive and normalised != phrase.lower():
            patterns.append(_strip_accents(phrase))

        for pattern_text in patterns:
            doc_pattern = nlp.make_doc(pattern_text)
            # add_patterns accepts duplicates; guard to avoid spaCy warnings.
            if key not in matcher:
                matcher.add(key, [doc_pattern])
            else:
                matcher.add(key, [doc_pattern])  # spaCy merges lists automatically

    return matcher, key_to_phrase


# ---------------------------------------------------------------------------
# Module-level singletons (built once, reused on every call)
# ---------------------------------------------------------------------------
_nlp: spacy.language.Language | None = None
_matcher: PhraseMatcher | None = None
_key_to_phrase: dict[str, str] = {}


def _get_pipeline(
    symptoms: Sequence[str] | None = None,
) -> tuple[spacy.language.Language, PhraseMatcher, dict[str, str]]:
    """Return (or lazily initialise) the shared NLP pipeline."""
    global _nlp, _matcher, _key_to_phrase

    vocab_changed = symptoms is not None and set(symptoms) != set(
        _key_to_phrase.values()
    )

    if _nlp is None or _matcher is None or vocab_changed:
        _nlp = spacy.blank("pt")
        _nlp.max_length = 2_000_000
        vocab = symptoms if symptoms is not None else DEFAULT_SYMPTOMS
        _matcher, _key_to_phrase = build_matcher(_nlp, vocab)

    return _nlp, _matcher, _key_to_phrase


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_symptoms(
    text: str,
    *,
    symptoms: Sequence[str] | None = None,
    preserve_duplicates: bool = False,
) -> list[str]:
    """
    Identify medical symptoms in a Portuguese *text* string.
    """
    if not text or not text.strip():
        return []

    nlp, matcher, key_to_phrase = _get_pipeline(symptoms)

    # Run the matcher on both the original text and its accent-stripped version
    # so we catch typed forms like "febre" and "naúsea" (common typos).
    doc = nlp(text)
    matches = matcher(doc)

    found: list[str] = []
    seen: set[str] = set()

    for match_id, _start, _end in matches:
        key = nlp.vocab.strings[match_id]
        original = key_to_phrase.get(key, key.replace("_", " "))
        if preserve_duplicates:
            found.append(original)
        elif original not in seen:
            seen.add(original)
            found.append(original)

    return found
