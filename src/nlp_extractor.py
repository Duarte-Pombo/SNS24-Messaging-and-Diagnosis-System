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
        "dor de cabeça", "dor no peito", "fadiga", "náusea", "vômito", "diarreia",
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


# ---------------------------------------------------------------------------
# Plural generation
# ---------------------------------------------------------------------------

def _pt_plurals(phrase: str) -> list[str]:
    """
    Return a list of likely Portuguese plural surface forms for *phrase*.
    """
    words = phrase.split()
    last = words[-1].lower()
    prefix = " ".join(words[:-1])  # everything before the last word

    candidates: list[str] = []

    def make(new_last: str) -> str:
        return (prefix + " " + new_last).strip()

    if last.endswith("ão"):
        stem = last[:-2]
        candidates += [make(stem + s) for s in ("ões", "ães", "ãos")]
    elif last.endswith("al"):
        candidates.append(make(last[:-2] + "ais"))
    elif last.endswith("el"):
        candidates.append(make(last[:-2] + "eis"))
    elif last.endswith("ol"):
        candidates.append(make(last[:-2] + "óis"))
    elif last.endswith("ul"):
        candidates.append(make(last[:-2] + "uis"))
    elif last.endswith("il"):
        candidates.append(make(last[:-2] + "is"))
    elif last.endswith("m"):
        candidates.append(make(last[:-1] + "ns"))
    elif last.endswith(("r", "z", "n")):
        candidates.append(make(last + "es"))
    elif last.endswith("s"):
        pass  # already plural / invariant
    else:
        candidates.append(make(last + "s"))

    return [c for c in candidates if c != phrase]


# ---------------------------------------------------------------------------
# Synonym map  
# ---------------------------------------------------------------------------

SYNONYM_MAP: dict[str, str] = {
    # ── Fatigue ──────────────────────────────────────────────────────────────
    "cansado":              "fadiga",
    "cansada":              "fadiga",
    "cansaço":              "fadiga",
    "cansaco":              "fadiga",
    "esgotado":             "fadiga",
    "esgotada":             "fadiga",
    "esgotamento":          "fadiga",
    "exausto":              "fadiga",
    "exausta":              "fadiga",
    "exaustão":             "fadiga",
    "sem energia":          "fadiga",
    "sem forças":           "fadiga",
    # ── Headache ─────────────────────────────────────────────────────────────
    "dores de cabeça":      "cefaleia",  # also caught by plural gen
    "dor de cabeça":        "cefaleia",
    "enxaqueca":            "cefaleia",
    "migrânea":             "cefaleia",
    "migranea":             "cefaleia",
    # ── Chest pain ───────────────────────────────────────────────────────────
    "dores no peito":       "dor no peito",
    "aperto no peito":      "dor no peito",
    "pressão no peito":     "dor no peito",
    "pressao no peito":     "dor no peito",
    # ── Nausea ───────────────────────────────────────────────────────────────
    "enjoado":              "náusea",
    "enjoada":              "náusea",
    "enjoo":                "náusea",
    "enjôo":                "náusea",
    "mal estar":            "náusea",
    # ── Fever ────────────────────────────────────────────────────────────────
    "temperatura":          "febre",
    "febril":               "febre",
    "estado febril":        "febre",
    # ── Cough ────────────────────────────────────────────────────────────────
    "tossindo":             "tosse",
    "tussia":               "tosse",
    # ── Dizziness ────────────────────────────────────────────────────────────
    "tonto":                "tontura",
    "tonta":                "tontura",
    "vertigem":             "tontura",
    "zonzo":                "tontura",
    "zonza":                "tontura",
    # ── Shortness of breath ──────────────────────────────────────────────────
    "sem fôlego":           "falta de ar",
    "sem folego":           "falta de ar",
    "ofegante":             "falta de ar",
    "respiração difícil":   "falta de ar",
    "respiracao dificil":   "falta de ar",
    # ── Abdominal pain ───────────────────────────────────────────────────────
    "dores abdominais":     "dor abdominal",
    "dor de barriga":       "dor abdominal",
    "barriga doendo":       "dor abdominal",
    "cólica":               "dor abdominal",
    "colica":               "dor abdominal",
    # ── Throat ───────────────────────────────────────────────────────────────
    "dores de garganta":    "dor de garganta",
    "garganta inflamada":   "dor de garganta",
    "garganta irritada":    "dor de garganta",
    # ── Swelling ─────────────────────────────────────────────────────────────
    "inchado":              "inchaço",
    "inchada":              "inchaço",
    "edema":                "inchaço",
    # ── Palpitations ─────────────────────────────────────────────────────────
    "coração acelerado":    "palpitações",
    "coração disparado":    "palpitações",
    "taquicardia":          "palpitações",
    # ── Fever / chills ───────────────────────────────────────────────────────
    "arrepios":             "calafrios",
    "tremores":             "calafrios",
}


def build_matcher(
    nlp: spacy.language.Language,
    symptoms: Sequence[str],
    *,
    case_insensitive: bool = True,
    accent_insensitive: bool = True,
) -> tuple[PhraseMatcher, dict[str, str]]:
    """
    Compile a PhraseMatcher from *symptoms* (plus their plural variants and
    synonyms) and return it together with a mapping from each match key back
    to the canonical symptom phrase.

    Every surface form — original, plural, synonym, accent-stripped — is
    registered under the **same key** so ``extract_symptoms`` always returns
    the canonical name from the vocabulary.
    """
    attr = "LOWER" if case_insensitive else "TEXT"
    matcher = PhraseMatcher(nlp.vocab, attr=attr)
    key_to_phrase: dict[str, str] = {}

    def _register(key: str, canonical: str, surface_forms: list[str]) -> None:
        """Add *surface_forms* to the matcher under *key* → *canonical*."""
        key_to_phrase[key] = canonical
        all_forms: list[str] = []
        for form in surface_forms:
            all_forms.append(form)
            if accent_insensitive:
                stripped = _strip_accents(form)
                if stripped != form:
                    all_forms.append(stripped)
        patterns = [nlp.make_doc(f) for f in dict.fromkeys(all_forms)]  # dedup, keep order
        if key in matcher:
            matcher.add(key, patterns)
        else:
            matcher.add(key, patterns)

    # 1. Register every canonical symptom + its auto-generated plural forms.
    for phrase in symptoms:
        normalised = _strip_accents(phrase.lower()) if accent_insensitive else phrase.lower()
        key = normalised.replace(" ", "_")
        forms = [phrase] + _pt_plurals(phrase)
        _register(key, phrase, forms)

    # 2. Register synonym surface forms, pointing at their canonical symptom.
    for synonym, canonical in SYNONYM_MAP.items():
        # Find the key of the canonical symptom (it must exist in step 1).
        canon_key = _strip_accents(canonical.lower()).replace(" ", "_")
        if canon_key not in key_to_phrase:
            import warnings
            warnings.warn(
                f"SYNONYM_MAP: canonical '{canonical}' (for synonym '{synonym}') "
                f"is not in the active vocabulary — entry ignored. "
                f"Add '{canonical}' to the symptom list or fix the mapping.",
                stacklevel=2,
            )
            continue
        # Synonyms also get plural variants.
        forms = [synonym] + _pt_plurals(synonym)
        # Add to the existing canonical key (don't create a duplicate key).
        all_forms: list[str] = []
        for form in forms:
            all_forms.append(form)
            if accent_insensitive:
                stripped = _strip_accents(form)
                if stripped != form:
                    all_forms.append(stripped)
        patterns = [nlp.make_doc(f) for f in dict.fromkeys(all_forms)]
        matcher.add(canon_key, patterns)

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


# ---------------------------------------------------------------------------
# CLI — python nlp_extractor.py --test
# ---------------------------------------------------------------------------

def _run_test_mode() -> None:
    """Interactive test loop: type a sentence, see the matched symptoms."""
    import json

    print("=" * 60)
    print("  NLP Symptom Extractor — interactive test mode")
    print("  Type a Portuguese sentence and press Enter.")
    print("  Leave the line empty and press Enter to quit.")
    print("=" * 60)

    # Warm up the pipeline once so the first query feels instant.
    _get_pipeline()

    print(f"\nActive vocabulary ({len(DEFAULT_SYMPTOMS)} symptoms, "
          f"{len(SYNONYM_MAP)} synonyms):")
    for s in DEFAULT_SYMPTOMS:
        print(f"  • {s}")
    print(f"\nSynonym map ({len(SYNONYM_MAP)} entries):")
    for alt, canon in SYNONYM_MAP.items():
        print(f"  • {alt!r:30s} → {canon!r}")

    while True:
        try:
            text = input("\nInput : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not text:
            print("Goodbye.")
            break

        result = extract_symptoms(text)
        print(f"Output: {json.dumps(result, ensure_ascii=False)}")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv or "-test" in sys.argv:
        _run_test_mode()
    else:
        print("Usage: python nlp_extractor.py --test")
        sys.exit(1)
