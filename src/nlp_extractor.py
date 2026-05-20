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
    {"diagnosis"}
)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_symptoms_from_csv(path: str | Path) -> list[str]:
    """
    Derive the symptom vocabulary from the first column of Symptom-severity_pt.csv.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader) 

        return [
            row[0].replace("_", " ")
            for row in reader
            if row and row[0] not in NON_SYMPTOM_COLS
        ]


# ---------------------------------------------------------------------------
# Default symptom vocabulary  (85 canonical phrases, organised by system)
# ---------------------------------------------------------------------------

def _bootstrap_default_symptoms() -> list[str]:
    """Load from a co-located CSV and intelligently merge with built-in list."""
    hardcoded = [
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
    
    candidate = Path(__file__).parent / "../data/symptoms_dataset/Symptom-severity_pt.csv"
    if candidate.exists():
        csv_symptoms = load_symptoms_from_csv(candidate)
        
        # Prevent terms defined as synonyms from overwriting hardcoded map behavior
        all_syn_forms = {syn.lower() for syns in SYNONYM_MAP.values() for syn in syns}
        
        for s in csv_symptoms:
            if s.lower() not in all_syn_forms and s not in hardcoded:
                hardcoded.append(s)
                
    return hardcoded


# ---------------------------------------------------------------------------
# Base Synonym Dictionary 
# ---------------------------------------------------------------------------
SYNONYM_MAP: dict [str, list[str]] ={
    "comichão": ["coceira", "prurido", "vontade de me coçar", "a coçar", "comixão"],
    "erupção cutânea": ["erupção", "manchas na pele", "borbulhas", "rash", "alergia na pele", "pintinhas vermelhas", "empolamento", "vermelhidão"],
    "erupções cutâneas nodulares": ["caroços na pele", "nódulos na pele", "altos na pele", "caroços vermelhos"],
    "espirros contínuos": ["espirro", "a espirrar", "ataque de espirros", "fartar de espirrar", "espirros de seguida", "espirros constantes", "espirros frequentes"],
    "tremores": ["tremor", "a tremer", "tremedeira", "tremiliques", "tremuras", "tremelique"],
    "arrepios": ["arrepio", "frio na espinha", "corpo arrepiado", "pele de galinha"],
    "dor nas articulações": ["dor nas juntas", "dor na junta", "artralgia", "dor articular", "juntas a doer", "dor na articulação", "juntas estão a doer"],
    "dor de estômago": ["dor no estômago", "estômago a doer", "gastralgia", "dor na boca do estômago", "pontada no estômago", "pontada de estomago"],
    "acidez": ["azia", "ardor no estômago", "fervura", "refluxo", "acidez estomacal", "azedume", "queimor"],
    "úlceras na língua": ["afta", "ferida na boca", "ferida na língua", "chaga na boca", "úlcera na língua", "boca magoada"],
    "atrofia muscular": ["perda de músculo", "músculo fraco", "fraqueza no músculo", "perda de massa muscular", "amiotrofia", "músculos a secar"],
    "vómitos": ["vómito", "a vomitar", "bolçar", "vomitar", "emese", "a deitar tudo fora", "gregar", "chamar o gregório", "disposição para vomitar", "deito tudo fora o que como"],
    "micção ardente": ["ardor ao urinar", "dor a urinar", "ardência ao urinar", "chichi a arder", "urina a arder", "ardor na urina", "disúria", "mijo a arder"],
    "perdas urinárias": ["incontinência urinária", "perda de urina", "urina solta", "fugas de urina", "urinar nas calças", "não aguentar a urina", "descair a urina"],
    "fadiga": ["cansaço", "exaustão", "moleza", "sem energia", "fadiga extrema", "esgotamento", "prostração", "astenia", "corpo pesado", "estafado", "derreado", "sem forças"],
    "aumento de peso": ["engordar", "ganho de peso", "engordei", "a ganhar peso", "peso a mais", "ganhei peso", "obesidade"],
    "ansiedade": ["nervosismo", "ansioso", "ansiosa", "stress", "angústia", "inquietação", "ataque de pânico", "crise de ansiedade", "nervos em franja"],
    "mãos e pés frios": ["extremidades frias", "mão e pé gelados", "pés gelados", "mãos geladas", "pé frio", "mão fria", "dedos gelados", "pés estão gelados"],
    "alterações de humor": ["mudança de humor", "bipolaridade", "humor instável", "mudanças de humor", "humor a variar", "irritação repentina"],
    "perda de peso": ["perda de quilos", "a emagrecer", "perdi peso", "ficar na chapa", "perdi muitos quilos"],
    "agitação": ["inquietação", "irrequieto", "agitado", "agitada", "não para quieto", "hiperativo", "desassossegado", "em pulgas"],
    "letargia": ["preguiça", "apatia", "falta de força", "moleza", "sonolência", "letárgico", "torpor", "amorfia"],
    "manchas na garganta": ["pontos brancos na garganta", "placas na garganta", "garganta manchada", "pus na garganta", "garganta infetada"],
    "nível de açúcar irregular": ["glicose alta", "diabetes desregulada", "açúcar no sangue", "glicemia", "glicose", "açúcar alto"],
    "tosse": ["a tossir", "encatarrado", "escarro", "flegma"],
    "febre alta": ["temperatura alta", "quentura", "corpo a ferver", "pirexia", "febre forte", "febrão"],
    "olhos encovados": ["olheiras profundas", "olhos fundos", "olho encovado", "olhos para dentro", "rosto chupado", "olheiras cavadas"],
    "falta de ar": ["dificuldade em respirar", "aflição para respirar", "fôlego curto", "falta de fôlego", "asfixia", "ofegante", "cansaço para respirar"],
    "sudorese": ["suor", "a transpirar", "suores", "transpiração", "a suar muito", "suor excessivo", "hiperidrose", "banhado em suor"],
    "desidratação": ["falta de água", "boca muito seca", "secura", "sede extrema"],
    "indigestão": ["enfartamento", "mal estar no estômago", "empanzinado", "dispepsia", "estômago pesado", "a comida caiu mal", "comida parada"],
    "dor de cabeça": ["enxaqueca", "cabeça a doer", "cabeça a latejar", "dor na cabeça", "cabeça pesada", "dói-me a cabeça", "dores de cabeça"],
    "cefaleia súbita": ["dor de cabeça súbita"],
    "pele amarelada": ["pele amarela", "amarelada", "olhos amarelos", "cor de açafrão", "amarelo", "olhos estão amarelos"],
    "urina escura": ["chichi escuro", "urina com cor escura", "urina castanha", "urina muito amarela", "urina cor de coca-cola", "urina forte", "chichi muito escuro"],
    "náuseas": ["enjoo", "enjoos", "vontade de vomitar", "enjoada", "enjoado", "estômago embrulhado", "engulhos"],
    "perda de apetite": ["falta de apetite", "sem fome", "inapetência", "não tenho fome", "falta de vontade de comer", "anorexia", "não consigo comer", "sem apetite"],
    "dor atrás dos olhos": ["dor no fundo do olho", "pressão nos olhos", "olhos a doer", "dor ocular", "dor atrás do olho"],
    "dor nas costas": ["dor lombar", "dor na coluna", "dores nas costas", "lombalgia", "dor nos rins", "dor nas cruzes", "dor no lombo"],
    "obstipação": ["prisão de ventre", "intestino preso", "constipação intestinal", "dificuldade em evacuar", "fezes duras", "entupido", "não consigo obrar"],
    "dor abdominal": ["dores abdominais", "dor no baixo ventre", "desconforto abdominal", "dói-me a barriga"],
    "dor na barriga": ["barriga a doer", "cólicas na barriga", "dor no ventre", "cólica", "pontada na barriga", "dor de barriga"],
    "diarreia": ["caganeira", "fezes moles", "desarranjo", "intestino solto", "frouxeira"],
    "vasos sanguíneos dilatados": ["veias saltadas", "varizes", "derrames", "vasos dilatados", "veias aparentes"],
    "rosto e olhos inchados": ["cara inchada", "olho inchado", "rosto inchado", "edema facial", "papos nos olhos", "cara gorda"],
    "tiróide aumentada": ["bócio", "papo", "tiróide grande", "inchaço no pescoço", "garganta inchada", "tireóide"],
    "unhas frágeis": ["unhas fracas", "unhas quebradiças", "unha rachada", "unhas a lascar", "unhas a quebrar"],
    "extremidades inchadas": ["pés inchados", "mãos inchadas", "pernas inchadas", "dedos inchados", "edema nas extremidades", "tornozelos inchados"],
    "fome excessiva": ["muita fome", "fome a toda a hora", "apetite voraz", "polifagia", "fome de leão", "esganado", "sempre com fome"],
    "contactos extraconjugais": ["traição", "amante", "relação extraconjugal", "parceiros múltiplos", "DST", "relações fora do casamento", "risco sexual", "pular a cerca"],
    "lábios secos e com formigueiro": ["boca seca e dormente", "lábio rachado e dormente", "formigueiro nos lábios", "lábios ressequidos", "boca dormente"],
    "discurso arrastado": ["fala enrolada", "dificuldade para falar", "voz arrastada", "fala arrastada", "disartria", "a falar embolado", "voz pastosa"],
    "dor no joelho": ["dor nos joelhos", "joelho a doer", "joelho dorido", "dor na rótula", "joelho a latejar"],
    "dor na articulação da anca": ["dor na anca", "dor na bacia", "dor na coxa", "anca a doer", "dor no osso da bacia"],
    "fraqueza muscular": ["fraqueza", "músculos fracos", "sem força nos músculos", "miastenia", "corpo fraco", "fraqueza nas pernas", "pernas a tremer"],
    "rigidez cervical": ["dor de pescoço", "torcicolo", "pescoço duro", "rigidez no pescoço", "dor na nuca", "nuca dura", "pescoço preso"],
    "inchaço das articulações": ["juntas inchadas", "articulações inchadas", "edema articular", "joelho inchado", "cotovelo inchado"],
    "rigidez de movimentos": ["dificuldade em me mover", "corpo duro", "travado", "movimentos encravados", "rigidez articular", "corpo preso"],
    "movimentos de rotação": ["tontura", "vertigem", "tudo a andar à roda", "cabeça a andar à roda", "mundo a rodar", "labirintite", "cabeça parece que anda à roda"],
    "perda de equilíbrio": ["desequilíbrio", "a cair", "sem equilíbrio", "zonzura", "zonzo", "a cambalear"],
    "instabilidade": ["a cambalear", "instável", "pernas bambas", "falta de firmeza", "pernas a fraquejar"],
    "fraqueza num lado do corpo": ["um lado fraco", "metade do corpo fraca", "hemiparesia", "dormência de um lado", "braço e perna fracos", "paralisado de um lado"],
    "perda de olfato": ["não sinto os cheiros", "sem cheiro", "anosmia", "perda do olfato", "falta de cheiro", "perdi o olfato", "perdi olfato"],
    "desconforto na bexiga": ["dor na bexiga", "bexiga pesada", "pressão na bexiga", "dor no baixo ventre ao urinar", "bexiga tão pesada"],
    "odor fétido na urina": ["urina com mau cheiro", "chichi a cheirar mal", "chichi com cheiro forte", "urina mal cheirosa", "urina fedorenta", "urina de cheiro forte", "fedorento"],
    "sensação contínua de urinar": ["vontade de urinar a toda a hora", "vontade de fazer chichi", "urina frequente", "bexiga cheia", "polaciúria"],
    "passagem de gases": ["gases", "peidos", "traques", "flatulência", "dar traques", "meteorismo", "barriga inchada de gases", "ventosidades"],
    "comichão interna": ["coceira por dentro", "comichão no corpo todo", "prurido interno"],
    "aspeto tóxico (tifo)": ["cara de doente", "aspecto tóxico", "muito abatido", "feições encovadas", "ar cadavérico", "má cara"],
    "depressão": ["tristeza profunda", "depressivo", "deprimido", "muito triste", "vontade de morrer", "melancolia", "isolamento", "em baixo"],
    "irritabilidade": ["irritado", "irritada", "sem paciência", "pavio curto", "nervoso", "mau humor", "passar-se dos nervos"],
    "dor muscular": ["mialgia", "dor nos músculos", "corpo dorido", "dores no corpo", "dor na carne", "músculos a doer", "dor no corpo todo"],
    "alteração do sensório": ["confusão mental", "delírio", "desorientação", "mente confusa", "alucinação", "fala desconexa"],
    "manchas vermelhas pelo corpo": ["pontos vermelhos", "pintas vermelhas", "manchas no corpo", "eritema", "placas vermelhas", "borbulhas vermelhas"],
    "menstruação anormal": ["menstruação irregular", "sangramento fora de hora", "ciclo desregulado", "regras anormais", "período atrasado", "amenorreia", "muito sangue", "hemorragia menstrual"]
}

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
    Return likely Portuguese plural surface forms for *phrase*.
    Handles single words, and multi-word phrases by inflecting the first word, 
    the last word, and both.
    """
    words = phrase.split()
    if not words:
        return []

    def pluralize_word(w: str) -> str:
        last = w.lower()
        if last.endswith("ão"):
            return last[:-2] + "ões" 
        elif last.endswith("al"): return last[:-2] + "ais"
        elif last.endswith("el"): return last[:-2] + "eis"
        elif last.endswith("ol"): return last[:-2] + "óis"
        elif last.endswith("ul"): return last[:-2] + "uis"
        elif last.endswith("il"): return last[:-2] + "is"
        elif last.endswith("m"): return last[:-1] + "ns"
        elif last.endswith(("r", "z", "n")): return last + "es"
        elif last.endswith("s"): return last
        else: return last + "s"

    candidates: set[str] = set()

    # single word
    if len(words) == 1:
        candidates.add(pluralize_word(words[0]))
        return list(candidates - {phrase})

    # pluralize only the last word
    candidates.add(" ".join(words[:-1] + [pluralize_word(words[-1])]))
    
    # pluralize only the first word
    candidates.add(" ".join([pluralize_word(words[0])] + words[1:]))
    
    # pluralize both first and last
    candidates.add(" ".join([pluralize_word(words[0])] + words[1:-1] + [pluralize_word(words[-1])]))

    return [c for c in candidates if c != phrase]


# ---------------------------------------------------------------------------
# Preposition normalisation  (NEW in v2)
# ---------------------------------------------------------------------------

_PREP_GROUP: frozenset[str] = frozenset(
    {"de", "do", "da", "dos", "das", "no", "na", "nos", "nas"}
)


def _prep_variants(phrase: str) -> list[str]:
    """
    Return all surface variants of *phrase* obtained by independently swapping
    every preposition word that belongs to _PREP_GROUP with every other member.
    """
    words = phrase.split()
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

_NEGATION_TOKENS: frozenset[str] = frozenset({
    "não", "nao", "sem", "nunca", "jamais",
    "ausência", "ausencia", "ausente",
    "nega", "nega-se", "descarta",
    "nenhum", "nenhuma", "inexistente",
    "nem", "tampouco"
})

_CLAUSE_BOUNDARIES: frozenset[str] = frozenset({
    ",", ".", ";", ":", "!", "?",
    "mas", "porém", "porem", "contudo", "todavia",
    "e", "ou", "embora", "apesar", "logo", "pois", "porque"
})

_NEGATION_WINDOW: int = 5


def _negation_before(doc: spacy.tokens.Doc, match_start: int,
                     window: int = _NEGATION_WINDOW) -> bool:
    """
    Return True when any negation token appears within *window* positions
    immediately before token index *match_start*, respecting clause boundaries.
    """
    start_scan = max(0, match_start - window)
    for i in range(match_start - 1, start_scan - 1, -1):
        token_lower = doc[i].lower_
        
        if token_lower in _CLAUSE_BOUNDARIES:
            break
            
        if token_lower in _NEGATION_TOKENS:
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
    sorted_m = sorted(matches, key=lambda m: (-(m[2] - m[1]), m[1]))
    occupied: set[int] = set()
    result: list[tuple[int, int, int]] = []
    
    for mid, start, end in sorted_m:
        span = set(range(start, end))
        if not span & occupied:
            result.append((mid, start, end))
            occupied |= span
            
    result.sort(key=lambda m: m[1])
    return result


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
        key_to_phrase[key] = canonical
        all_forms = _expand(base_forms)
        patterns = [nlp.make_doc(f) for f in all_forms]
        matcher.add(key, patterns)

    for phrase in symptoms:
        normalised = _strip_accents(phrase.lower()) if accent_insensitive else phrase.lower()
        key = normalised.replace(" ", "_")
        forms = [phrase] + _pt_plurals(phrase)
        _register(key, phrase, forms)

    for canonical, synonyms in SYNONYM_MAP.items():
        canon_key = _strip_accents(canonical.lower()).replace(" ", "_")
        if canon_key not in key_to_phrase:
            warnings.warn(
                f"SYNONYM_MAP: canonical '{canonical}' is not in the active vocabulary "
                f"— entry ignored. Add '{canonical}' to the symptom list or fix the mapping.",
                stacklevel=2,
            )
            continue
            
        all_synonym_forms = []
        for syn in synonyms:
            all_synonym_forms.append(syn)
            all_synonym_forms.extend(_pt_plurals(syn))
            
        forms = _expand(all_synonym_forms)
        patterns = [nlp.make_doc(f) for f in forms]
        matcher.add(canon_key, patterns)
        
    return matcher, key_to_phrase


# ---------------------------------------------------------------------------
# Module-level singletons 
# ---------------------------------------------------------------------------

_nlp:          spacy.language.Language | None = None
_matcher:      PhraseMatcher | None           = None
_key_to_phrase: dict[str, str]                = {}


def _get_pipeline(
    symptoms: Sequence[str] | None = None,
) -> tuple[spacy.language.Language, PhraseMatcher, dict[str, str]]:
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

    text = _strip_accents(text)

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

    if not rich:
        # Guarantee alphabetical determinism for standard lists
        found_plain.sort()
        return found_plain
        
    found_rich.sort(key=lambda m: m.symptom)
    return found_rich


# ---------------------------------------------------------------------------
# CLI — python nlp_extractor.py --test
# ---------------------------------------------------------------------------

def _run_test_mode() -> None:
    import json

    print("=" * 60)
    print("  NLP Symptom Extractor — interactive test mode")
    print("=" * 60)

    _get_pipeline()

    print(f"\nActive vocabulary ({len(DEFAULT_SYMPTOMS)} symptoms, "
          f"{len(SYNONYM_MAP)} synonyms):")
    for s in DEFAULT_SYMPTOMS:
        print(f"  • {s}")
    print(f"\nSynonym map ({sum(len(v) for v in SYNONYM_MAP.values())} total synonyms):")
    for canon, synonyms in SYNONYM_MAP.items():
        print(f"  • {canon!r:30s} ← {', '.join(synonyms)}")

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
