"""
Extracts recognised medical symptoms from a Portuguese string using
spaCy's PhraseMatcher.
"""

from __future__ import annotations

import csv
import itertools
import unicodedata
import warnings
from dataclasses import dataclass
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
        headers = next(reader)

    return [
        col.replace("_", " ")
        for col in headers
        if col not in NON_SYMPTOM_COLS
    ]


# ---------------------------------------------------------------------------
# Default symptom vocabulary  (85 canonical phrases, organised by system)
# ---------------------------------------------------------------------------

def _bootstrap_default_symptoms() -> list[str]:
    """Try to load from a co-located CSV; fall back to the built-in list."""
    candidate = Path(__file__).parent / "../data/symptoms_data.csv"
    if candidate.exists():
        return load_symptoms_from_csv(candidate)
    return [
        # ── General / Constitutional ─────────────────────────────────────────
        "febre", "febre baixa", "febre alta", "calafrios", "fadiga",
        "fraqueza", "mal-estar geral", "sudorese", "suores frios",
        "suores noturnos", "emagrecimento", "ganho de peso", "perda de apetite",
        # ── Neurological / Psychiatric ───────────────────────────────────────
        "cefaleia", "cefaleia pulsátil", "cefaleia súbita",
        "dor de cabeça", "tontura", "vertigem", "síncope", "pré-síncope",
        "confusão mental", "desorientação", "amnésia", "convulsão", "tremor",
        "ataxia", "afasia", "dificuldade falar", "fraqueza facial",
        "dormência", "formigueiro", "parestesia",
        "ansiedade", "depressão", "irritabilidade", "agitação",
        "alucinação", "insônia", "sonolência excessiva",
        # ── Eyes / Vision ────────────────────────────────────────────────────
        "visão turva", "diplopia", "perda de visão", "fotofobia",
        "fotopsia", "olho vermelho",
        # ── ENT / Head ───────────────────────────────────────────────────────
        "zumbido", "otalgia", "dor no ouvido", "epistaxe",
        "sangramento nasal", "anosmia", "ageusia", "dor de garganta",
        "odinofagia", "disfagia", "congestão nasal", "rinorreia", "espirros",
        # ── Respiratory ──────────────────────────────────────────────────────
        "tosse", "tosse seca", "tosse produtiva", "falta de ar", "dispneia",
        "pieira", "sibilância", "expectoração", "hemoptise",
        "dor ao respirar", "dor pleurítica", "crepitações",
        # ── Cardiovascular ───────────────────────────────────────────────────
        "dor no peito", "palpitações", "irradiação braço",
        "taquicardia", "bradicardia", "hipertensão", "hipotensão",
        # ── Gastrointestinal ─────────────────────────────────────────────────
        "náusea", "vômito", "diarreia", "constipação", "obstipação",
        "dor abdominal", "dor abdominal difusa", "rigidez abdominal",
        "distensão abdominal", "pirose", "regurgitação",
        "flatulência", "melena", "hematêmese",
        # ── Urinary ──────────────────────────────────────────────────────────
        "ardor ao urinar", "disúria", "frequência urinária",
        "poliúria", "oligúria", "hematúria",
        # ── Musculoskeletal ──────────────────────────────────────────────────
        "artralgia", "mialgia", "lombalgia", "cervicalgia",
        "dor no ombro", "dor no joelho", "dor no quadril",
        "dor localizada", "inchaço", "rigidez articular",
        # ── Skin ─────────────────────────────────────────────────────────────
        "prurido", "erupção cutânea", "urticária",
        "icterícia", "cianose", "palidez", "rubor",
        # ── Metabolic / Endocrine ────────────────────────────────────────────
        "polidipsia", "polifagia", "hipoglicemia",
    ]


DEFAULT_SYMPTOMS: list[str] = _bootstrap_default_symptoms()


def set_default_symptoms(symptoms: Sequence[str]) -> None:
    """Replace the module-level default vocabulary and invalidate the cache."""
    global DEFAULT_SYMPTOMS, _matcher, _key_to_phrase
    DEFAULT_SYMPTOMS = list(symptoms)
    _matcher = None
    _key_to_phrase = {}


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _strip_accents(text: str) -> str:
    """Return *text* with diacritics removed (NFC-safe)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


# ---------------------------------------------------------------------------
# Plural generation
# ---------------------------------------------------------------------------

def _pt_plurals(phrase: str) -> list[str]:
    """
    Return likely Portuguese plural surface forms for *phrase* by inflecting
    the **last** content word.  Multi-word plurals that don't follow these
    rules (e.g. "dores de cabeça") should be entered in SYNONYM_MAP instead.
    """
    words = phrase.split()
    last = words[-1].lower()
    prefix = " ".join(words[:-1])

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
    elif last.endswith("gem"):
        candidates.append(make(last[:-2] + "ns"))
    elif last.endswith("m") and not last.endswith("gem"):
        candidates.append(make(last[:-1] + "ns"))
    elif last.endswith("nte"):
        candidates.append(make(last + "s"))
    elif last.endswith(("r", "z", "n")):
        candidates.append(make(last + "es"))
    elif last.endswith("s"):
        pass
    else:
        candidates.append(make(last + "s"))

    return [c for c in candidates if c != phrase]


# ---------------------------------------------------------------------------
# Preposition normalisation  (NEW in v2)
# ---------------------------------------------------------------------------

# These prepositions / contractions are treated as fully interchangeable
# inside symptom phrases.  "dor de peito" == "dor no peito" == "dor do peito".
_PREP_GROUP: frozenset[str] = frozenset(
    {"de", "do", "da", "dos", "das", "no", "na", "nos", "nas"}
)


def _prep_variants(phrase: str) -> list[str]:
    """
    Return all surface variants of *phrase* obtained by independently swapping
    every preposition word that belongs to _PREP_GROUP with every other member.
    """
    words = phrase.split()

    # Identify positions that carry a swappable preposition and list all
    # alternatives (including the original) for that position.
    swap_specs: list[tuple[int, list[str]]] = []
    for i, w in enumerate(words):
        if w.lower() in _PREP_GROUP:
            swap_specs.append((i, sorted(_PREP_GROUP)))

    if not swap_specs:
        return [phrase]

    positions   = [pos  for pos,  _    in swap_specs]
    option_sets = [opts for _,    opts in swap_specs]

    variants: set[str] = set()
    for combo in itertools.product(*option_sets):
        new_words = words[:]
        for idx, prep in zip(positions, combo):
            new_words[idx] = prep
        variants.add(" ".join(new_words))

    return list(variants)


# ---------------------------------------------------------------------------
# Negation detection
# ---------------------------------------------------------------------------

# Tokens that negate the symptom that follows them.
_NEGATION_TOKENS: frozenset[str] = frozenset({
    "não", "nao", "sem", "nunca", "jamais",
    "ausência", "ausencia", "ausente",
    "nega", "nega-se", "descarta",
    "nenhum", "nenhuma", "inexistente",
})

# How many tokens before the match start to scan for a negation cue.
_NEGATION_WINDOW: int = 5


def _negation_before(doc: spacy.tokens.Doc, match_start: int,
                     window: int = _NEGATION_WINDOW) -> bool:
    """
    Return True when any negation token appears within *window* positions
    immediately before token index *match_start*.
    """
    for i in range(max(0, match_start - window), match_start):
        if doc[i].lower_ in _NEGATION_TOKENS:
            return True
    return False


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------

def _resolve_overlaps(
    matches: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """
    Given a list of (match_id, start, end) tuples, return only the longest
    non-overlapping match at each token position.
    """
    # Sort: longer spans first; ties broken by earlier start
    sorted_m = sorted(matches, key=lambda m: (-(m[2] - m[1]), m[1]))
    occupied: set[int] = set()
    result: list[tuple[int, int, int]] = []
    for mid, start, end in sorted_m:
        span = set(range(start, end))
        if not span & occupied:
            result.append((mid, start, end))
            occupied |= span
    # Restore document order
    result.sort(key=lambda m: m[1])
    return result


# ---------------------------------------------------------------------------
# Rich output type
# ---------------------------------------------------------------------------

@dataclass
class SymptomMatch:
    """A single matched symptom with full provenance information."""
    symptom: str    # canonical vocabulary name
    text:    str    # the exact surface form found in the input
    start:   int    # token index (inclusive) in the spaCy Doc
    end:     int    # token index (exclusive) in the spaCy Doc
    negated: bool   # True when a negation token precedes the match


# ---------------------------------------------------------------------------
# Synonym map  (expanded in v2 — 150+ entries)
# ---------------------------------------------------------------------------

SYNONYM_MAP: dict[str, str] = {
    # ── Fatigue ──────────────────────────────────────────────────────────────
    "cansado":                  "fadiga",
    "cansada":                  "fadiga",
    "cansaço":                  "fadiga",
    "cansaco":                  "fadiga",
    "esgotado":                 "fadiga",
    "esgotada":                 "fadiga",
    "esgotamento":              "fadiga",
    "exausto":                  "fadiga",
    "exausta":                  "fadiga",
    "exaustão":                 "fadiga",
    "sem energia":              "fadiga",
    "sem forças":               "fadiga",
    "moleza":                   "fadiga",
    # ── Headache ─────────────────────────────────────────────────────────────
    "dores de cabeça":          "cefaleia",
    "dor de cabeça":            "cefaleia",
    "enxaqueca":                "cefaleia",
    "migrânea":                 "cefaleia",
    "migranea":                 "cefaleia",
    "cabeça doendo":            "cefaleia",
    "cabeça pesada":            "cefaleia",
    "cabeça latejando":         "cefaleia pulsátil",
    # ── Chest pain ───────────────────────────────────────────────────────────
    "dores no peito":           "dor no peito",
    "dores de peito":           "dor no peito",
    "aperto no peito":          "dor no peito",
    "aperto de peito":          "dor no peito",
    "pressão no peito":         "dor no peito",
    "pressao no peito":         "dor no peito",
    "pressão de peito":         "dor no peito",
    "dor torácica":             "dor no peito",
    "dor toracica":             "dor no peito",
    "queimação no peito":       "dor no peito",
    # ── Nausea ───────────────────────────────────────────────────────────────
    "enjoado":                  "náusea",
    "enjoada":                  "náusea",
    "enjoo":                    "náusea",
    "enjôo":                    "náusea",
    "mal estar":                "náusea",
    "estômago embrulhado":      "náusea",
    "estomago embrulhado":      "náusea",
    "vontade de vomitar":       "náusea",
    # ── Vomiting ─────────────────────────────────────────────────────────────
    "vomitando":                "vômito",
    "vomitei":                  "vômito",
    "vomitar":                  "vômito",
    "vomitou":                  "vômito",
    # ── Fever ────────────────────────────────────────────────────────────────
    "temperatura":              "febre",
    "febril":                   "febre",
    "estado febril":            "febre",
    "temperatura elevada":      "febre alta",
    "febrinha":                 "febre baixa",
    "subfebril":                "febre baixa",
    # ── Chills ───────────────────────────────────────────────────────────────
    "arrepios":                 "calafrios",
    "tremores":                 "calafrios",
    "agitação por frio":        "calafrios",
    # ── Cough ────────────────────────────────────────────────────────────────
    "tossindo":                 "tosse",
    "tussia":                   "tosse",
    "pigarro":                  "tosse seca",
    "tosse com catarro":        "tosse produtiva",
    "tosse com muco":           "tosse produtiva",
    # ── Shortness of breath ──────────────────────────────────────────────────
    "sem fôlego":               "falta de ar",
    "sem folego":               "falta de ar",
    "ofegante":                 "falta de ar",
    "ofegância":                "falta de ar",
    "respiração difícil":       "falta de ar",
    "respiracao dificil":       "falta de ar",
    "respiração curta":         "falta de ar",
    "dificuldade respirar":     "falta de ar",
    "dificuldade de respirar":  "falta de ar",
    # ── Dizziness ────────────────────────────────────────────────────────────
    "tonto":                    "tontura",
    "tonta":                    "tontura",
    "vertigem":                 "tontura",
    "zonzo":                    "tontura",
    "zonza":                    "tontura",
    "cabeça rodando":           "tontura",
    "quase desmaiei":           "pré-síncope",
    "sensação de desmaio":      "pré-síncope",
    "desmaio":                  "síncope",
    "desmaiei":                 "síncope",
    "perda de consciência":     "síncope",
    # ── Abdominal pain ───────────────────────────────────────────────────────
    "dores abdominais":         "dor abdominal",
    "dor de barriga":           "dor abdominal",
    "barriga doendo":           "dor abdominal",
    "cólica":                   "dor abdominal",
    "colica":                   "dor abdominal",
    "dor na barriga":           "dor abdominal",
    "barriga dura":             "rigidez abdominal",
    "barriga inchada":          "distensão abdominal",
    "abdômen distendido":       "distensão abdominal",
    "abdome distendido":        "distensão abdominal",
    "gases":                    "flatulência",
    "muito gás":                "flatulência",
    # ── Heartburn / Reflux ───────────────────────────────────────────────────
    "azia":                     "pirose",
    "queimação no estômago":    "pirose",
    "queimacao no estomago":    "pirose",
    "refluxo":                  "pirose",
    "acidez":                   "pirose",
    "queimação de estômago":    "pirose",
    # ── Swallowing ───────────────────────────────────────────────────────────
    "dificuldade engolir":      "disfagia",
    "dificuldade de engolir":   "disfagia",
    "engasgando":               "disfagia",
    "dor ao engolir":           "odinofagia",
    "dor de engolir":           "odinofagia",
    # ── Throat ───────────────────────────────────────────────────────────────
    "dores de garganta":        "dor de garganta",
    "garganta inflamada":       "dor de garganta",
    "garganta irritada":        "dor de garganta",
    "garganta doendo":          "dor de garganta",
    # ── Swelling ─────────────────────────────────────────────────────────────
    "inchado":                  "inchaço",
    "inchada":                  "inchaço",
    "edema":                    "inchaço",
    "perna inchada":            "inchaço",
    "tornozelo inchado":        "inchaço",
    # ── Palpitations ─────────────────────────────────────────────────────────
    "coração acelerado":        "palpitações",
    "coração disparado":        "palpitações",
    "coração pulando":          "palpitações",
    "taquicardia":              "palpitações",
    "coração batendo forte":    "palpitações",
    # ── Back / Musculoskeletal ───────────────────────────────────────────────
    "dor nas costas":           "lombalgia",
    "dores nas costas":         "lombalgia",
    "dor lombar":               "lombalgia",
    "coluna doendo":            "lombalgia",
    "dor no pescoço":           "cervicalgia",
    "dor de pescoço":           "cervicalgia",
    "pescoço duro":             "cervicalgia",
    "dor muscular":             "mialgia",
    "dor nos músculos":         "mialgia",
    "dor nos musculos":         "mialgia",
    "dor nas articulações":     "artralgia",
    "dor nas articulacoes":     "artralgia",
    "dor nas juntas":           "artralgia",
    "juntas doendo":            "artralgia",
    # ── Numbness / Tingling ──────────────────────────────────────────────────
    "adormecido":               "dormência",
    "adormecida":               "dormência",
    "braço adormecido":         "dormência",
    "perna adormecida":         "dormência",
    "formigando":               "formigueiro",
    "formiga no braço":         "formigueiro",
    "agulhadas":                "formigueiro",
    "picadas":                  "parestesia",
    # ── Smell / Taste ────────────────────────────────────────────────────────
    "sem olfato":               "anosmia",
    "perdi o cheiro":           "anosmia",
    "não consigo cheirar":      "anosmia",
    "sem paladar":              "ageusia",
    "perdi o gosto":            "ageusia",
    "não sinto sabor":          "ageusia",
    # ── Tinnitus ─────────────────────────────────────────────────────────────
    "zunido":                   "zumbido",
    "apito no ouvido":          "zumbido",
    "campainha no ouvido":      "zumbido",
    # ── Ear pain ─────────────────────────────────────────────────────────────
    "dor de ouvido":            "otalgia",
    "ouvido doendo":            "otalgia",
    # ── Nosebleed ────────────────────────────────────────────────────────────
    "sangramento pelo nariz":   "epistaxe",
    "nariz sangrando":          "epistaxe",
    "nariz a sangrar":          "epistaxe",
    # ── Runny nose ───────────────────────────────────────────────────────────
    "coriza":                   "rinorreia",
    "nariz escorrendo":         "rinorreia",
    "nariz a pingar":           "rinorreia",
    "catarro nasal":            "rinorreia",
    "corrimento nasal":         "rinorreia",
    # ── Nasal congestion ────────────────────────────────────────────────────
    "nariz entupido":           "congestão nasal",
    "nariz tapado":             "congestão nasal",
    "nariz congestionado":      "congestão nasal",
    # ── Light sensitivity ───────────────────────────────────────────────────
    "sensibilidade à luz":      "fotofobia",
    "sensibilidade a luz":      "fotofobia",
    "luz incomoda":             "fotofobia",
    "luz me incomoda":          "fotofobia",
    # ── Urinary ─────────────────────────────────────────────────────────────
    "queimação ao urinar":      "ardor ao urinar",
    "ardor urinário":           "ardor ao urinar",
    "dor ao fazer xixi":        "ardor ao urinar",
    "urinando muito":           "frequência urinária",
    "muita urina":              "poliúria",
    "sangue na urina":          "hematúria",
    "urina com sangue":         "hematúria",
    # ── Skin ────────────────────────────────────────────────────────────────
    "coceira":                  "prurido",
    "comichão":                 "prurido",
    "coçando":                  "prurido",
    "manchas na pele":          "erupção cutânea",
    "brotoejas":                "urticária",
    "urticária":                "erupção cutânea",
    "vermelhidão na pele":      "erupção cutânea",
    "pele amarelada":           "icterícia",
    "olho amarelo":             "icterícia",
    "lábios roxos":             "cianose",
    "pálido":                   "palidez",
    "pálida":                   "palidez",
    # ── Sleep / Psych ────────────────────────────────────────────────────────
    "não consigo dormir":       "insônia",
    "dificuldade dormir":       "insônia",
    "dificuldade de dormir":    "insônia",
    "noites sem dormir":        "insônia",
    "muito sono":               "sonolência excessiva",
    "dormindo de dia":          "sonolência excessiva",
    "nervoso":                  "ansiedade",
    "nervosa":                  "ansiedade",
    "angústia":                 "ansiedade",
    "angustia":                 "ansiedade",
    "agoniado":                 "ansiedade",
    "agoniada":                 "ansiedade",
    # ── Metabolic ────────────────────────────────────────────────────────────
    "muita sede":               "polidipsia",
    "sede excessiva":           "polidipsia",
    "bebendo muito":            "polidipsia",
    "perdendo peso":            "emagrecimento",
    "emagreci":                 "emagrecimento",
    "baixa de açúcar":          "hipoglicemia",
    "baixa de acucar":          "hipoglicemia",
    "açúcar baixo":             "hipoglicemia",
    # ── Blood pressure ──────────────────────────────────────────────────────
    "pressão alta":             "hipertensão",
    "pressao alta":             "hipertensão",
    "pressão elevada":          "hipertensão",
    "pressão baixa":            "hipotensão",
    "pressao baixa":            "hipotensão",
    "pressão caída":            "hipotensão",
    # ── Sweating ─────────────────────────────────────────────────────────────
    "suando muito":             "sudorese",
    "suor excessivo":           "sudorese",
    "transpiração excessiva":   "sudorese",
}


# ---------------------------------------------------------------------------
# Matcher builder
# ---------------------------------------------------------------------------

def build_matcher(
    nlp: spacy.language.Language,
    symptoms: Sequence[str],
    *,
    case_insensitive: bool = True,
    accent_insensitive: bool = True,
) -> tuple[PhraseMatcher, dict[str, str]]:
    """
    Compile a PhraseMatcher from *symptoms* (plus their plural variants,
    preposition variants, synonym surface forms, and accent-stripped versions)
    and return it together with a mapping from each match key back to the
    canonical symptom phrase.
    """
    attr = "LOWER" if case_insensitive else "TEXT"
    matcher = PhraseMatcher(nlp.vocab, attr=attr)
    key_to_phrase: dict[str, str] = {}

    def _expand(forms: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for form in forms:
            for pv in _prep_variants(form):
                seen.setdefault(pv, None)
                if accent_insensitive:
                    stripped = _strip_accents(pv)
                    if stripped != pv:
                        seen.setdefault(stripped, None)
        return list(seen)

    def _register(key: str, canonical: str, base_forms: list[str]) -> None:
        """Add expanded surface forms to the matcher under *key*."""
        key_to_phrase[key] = canonical
        all_forms = _expand(base_forms)
        patterns = [nlp.make_doc(f) for f in all_forms]
        matcher.add(key, patterns)

    # 1. Register every canonical symptom + its auto-generated plural forms.
    for phrase in symptoms:
        normalised = _strip_accents(phrase.lower()) if accent_insensitive else phrase.lower()
        key = normalised.replace(" ", "_")
        forms = [phrase] + _pt_plurals(phrase)
        _register(key, phrase, forms)

    # 2. Register synonym surface forms, pointing at their canonical symptom.
    for synonym, canonical in SYNONYM_MAP.items():
        canon_key = _strip_accents(canonical.lower()).replace(" ", "_")
        if canon_key not in key_to_phrase:
            warnings.warn(
                f"SYNONYM_MAP: canonical '{canonical}' (for synonym '{synonym}') "
                f"is not in the active vocabulary — entry ignored. "
                f"Add '{canonical}' to the symptom list or fix the mapping.",
                stacklevel=2,
            )
            continue
        forms = _expand([synonym] + _pt_plurals(synonym))
        patterns = [nlp.make_doc(f) for f in forms]
        matcher.add(canon_key, patterns)

    return matcher, key_to_phrase


# ---------------------------------------------------------------------------
# Module-level singletons (built once, reused on every call)
# ---------------------------------------------------------------------------

_nlp:          spacy.language.Language | None = None
_matcher:      PhraseMatcher | None           = None
_key_to_phrase: dict[str, str]                = {}


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
    symptoms:           Sequence[str] | None = None,
    preserve_duplicates: bool  = False,
    negation_aware:      bool  = True,
    negation_window:     int   = _NEGATION_WINDOW,
    rich:                bool  = False,
) -> list[str] | list[SymptomMatch]:
    """
    Identify medical symptoms in a Portuguese *text* string.
    """
    if not text or not text.strip():
        return []

    nlp, matcher, key_to_phrase = _get_pipeline(symptoms)
    doc = nlp(text)

    raw_matches: list[tuple[int, int, int]] = matcher(doc)
    resolved    = _resolve_overlaps(raw_matches)

    found_plain: list[str]         = []
    found_rich:  list[SymptomMatch] = []
    seen: set[str] = set()

    for match_id, start, end in resolved:
        key      = nlp.vocab.strings[match_id]
        canon    = key_to_phrase.get(key, key.replace("_", " "))
        negated  = negation_aware and _negation_before(doc, start, negation_window)

        if rich:
            span_text = doc[start:end].text
            sm = SymptomMatch(
                symptom=canon,
                text=span_text,
                start=start,
                end=end,
                negated=negated,
            )
            if preserve_duplicates or canon not in seen:
                seen.add(canon)
                found_rich.append(sm)
        else:
            if negated:
                continue
            if preserve_duplicates:
                found_plain.append(canon)
            elif canon not in seen:
                seen.add(canon)
                found_plain.append(canon)

    return found_rich if rich else found_plain


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
