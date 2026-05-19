"""
Automated test suite for the NLP Symptom Extractor.
Run from the root directory or the scripts directory.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
# Dynamically add the project root to sys.path so we can import from 'src'
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.nlp_extractor import extract_symptoms
except ImportError as e:
    print(f"Error importing nlp_extractor: {e}")
    print("Ensure you are running this script within the project environment.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
TEST_CASES = [
    # ── 1. The "Bread and Butter" (Basic Synonyms & Plurals) ───────────────
    {
        "input": "Dói-me a cabeça e ando muito ansioso ultimamente",
        "expected": ["dor de cabeça", "ansiedade"]
    },
    {
        "input": "Os meus pés estão gelados e sinto umas pontadas no estômago.",
        "expected": ["mãos e pés frios", "dor de estômago"]
    },
    {
        "input": "Tenho andado com febrões e a suar muito durante a noite.",
        "expected": ["febre alta", "sudorese"]
    },
    {
        "input": "Tenho tido ataques de espirros e ando com os pés gelados",
        "expected": ["espirros contínuos", "mãos e pés frios"]
    },

    # ── 2. Negation Logic & Clause Boundaries ───────────────────────────────
    {
        "input": "Sinto muito cansaco, nao tenho fome nenhuma e estou a emagrecer rapido",
        "expected": ["fadiga", "perda de apetite", "perda de peso"] 
        # Note: 'nenhuma' negates 'fome', 'emagrecer' is protected by 'e'
    },
    {
        "input": "Não tenho febre mas tenho dores abdominais",
        "expected": ["dor abdominal"]
    },
    {
        "input": "Apesar de não tossir nem ter falta de ar, perdi o olfato",
        "expected": ["perda de olfato"]
    },
    {
        "input": "Sem dores de cabeça nem tonturas.",
        "expected": []
    },

    # ── 3. Accent and Case Insensitivity ────────────────────────────────────
    {
        "input": "doi-me a cabeca e tenho prisao de ventre.",
        "expected": ["dor de cabeça", "obstipação"]
    },
    {
        "input": "MuItA FomE e fAltA De aR",
        "expected": ["fome excessiva", "falta de ar"]
    },
    {
        "input": "o meu chichi escuro cheira mal",
        "expected": ["urina escura"] # 'cheira mal' might trigger 'odor fétido' depending on map
    },

    # ── 4. Overlap Resolution (Longest Match Wins) ──────────────────────────
    {
        "input": "Sinto uma dor de cabeça súbita.",
        "expected": ["cefaleia súbita"] # Assuming 'dor de cabeça súbita' maps to this.
    },

    # ── 5. Preposition Variants ─────────────────────────────────────────────
    {
        "input": "Sinto uma dor do peito.",
        "expected": ["dor no peito"]
    },
    {
        "input": "Tenho dor da nuca.",
        "expected": ["rigidez cervical"] # Assuming 'dor na nuca' is in the synonym map
    },
    
    # ── 6. Edge Cases / Empty Inputs ────────────────────────────────────────
    {
        "input": "",
        "expected": []
    },
    {
        "input": "Estou bem, sem qualquer problema.",
        "expected": []
    },

    # ── 7. Colloquialisms & Synonyms (Stress-testing SYNONYM_MAP) ───────────
    {
        "input": "Ando com os nervos em franja e não para quieto.",
        "expected": ["ansiedade", "agitação"]
    },
    {
        "input": "Sinto o estômago pesado, a comida caiu mal.",
        "expected": ["indigestão"] # 'estômago pesado' and 'a comida caiu mal' map to indigestão
    },
    {
        "input": "Tenho o corpo dorido, sinto o corpo pesado e não tenho fome.",
        "expected": ["dor muscular", "fadiga", "perda de apetite"]
    },
    {
        "input": "Ele confessou que costuma pular a cerca.",
        "expected": ["contactos extraconjugais"]
    },
    {
        "input": "Boca dormente e cara gorda.",
        "expected": ["lábios secos e com formigueiro", "rosto e olhos inchados"]
    },
    {
        "input": "Chichi com cheiro forte e dor no baixo ventre ao urinar.",
        "expected": ["odor fétido na urina", "desconforto na bexiga"]
    },
    {
        "input": "Estou a falar embolado e com metade do corpo fraca.",
        "expected": ["discurso arrastado", "fraqueza num lado do corpo"]
    },
    {
        "input": "Sinto a cabeça a andar à roda e pernas bambas.",
        "expected": ["movimentos de rotação", "instabilidade"]
    },
    {
        "input": "Cor de açafrão, urina cor de coca-cola e fezes duras.",
        "expected": ["pele amarelada", "urina escura", "obstipação"]
    },

    # ── 8. Complex Negations & Clause Boundaries ────────────────────────────
    {
        "input": "Não tenho febre, mas apresento tosse seca e pieira.",
        "expected": ["tosse seca", "pieira"]
    },
    {
        "input": "Ausência de icterícia, palidez ou cianose.",
        "expected": ["palidez", "cianose"] 
    },
    {
        "input": "Paciente nega dores abdominais; contudo, relata pirose.",
        "expected": ["pirose"]
    },
    {
        "input": "Nunca teve asfixia porque pratica desporto.",
        "expected": [] # 'asfixia' maps to 'falta de ar'
    },
    {
        "input": "Sem queixas de dor na coluna, nem lombalgia, mas com pescoço duro.",
        "expected": ["rigidez cervical"]
    },
    {
        "input": "Descarta hemoptise e não tem fôlego curto.",
        "expected": []
    },
    {
        "input": "Inexistente dor a urinar, embora tenha urina solta.",
        "expected": ["perdas urinárias"] # 'dor a urinar' is negated, 'urina solta' is protected by 'embora'
    },

    # ── 9. Overlap Resolution (Ensuring the longest canonical wins) ─────────
    {
        "input": "Sinto dor abdominal difusa e não apenas dor abdominal.",
        "expected": ["dor abdominal difusa", "dor abdominal"] 
        # The second 'dor abdominal' will be extracted because it doesn't overlap the first one!
    },
    {
        "input": "Apresenta febre alta e febre baixa em dias alternados.",
        "expected": ["febre alta", "febre baixa"] 
        # Should not extract the base 'febre' for either
    },
    {
        "input": "Tem tosse seca, tosse produtiva e tosse normal.",
        "expected": ["tosse seca", "tosse produtiva", "tosse"]
    },

    # ── 10. Plurals & Accents (Morphological Generator Testing) ─────────────
    {
        "input": "As unhas quebradiças e as pernas inchadas preocupam-me.",
        "expected": ["unhas frágeis", "extremidades inchadas"]
    },
    {
        "input": "As minhas juntas estao a doer e os meus olhos estao amarelos.",
        "expected": ["dor nas articulações", "pele amarelada"]
    },
    {
        "input": "Tive uns tremores, calafrios e uns suores noturnos.",
        "expected": ["tremores", "calafrios", "suores noturnos"]
    },
    {
        "input": "Muitas dores de cabeca e colicas na barriga.",
        "expected": ["dor de cabeça", "dor na barriga"]
    },
    {
        "input": "Tenho tido hemorragias menstruais fortes e mudancas de humor.",
        "expected": ["menstruação anormal", "alterações de humor"]
    },
    {
        "input": "Apresenta multiplos altos na pele e comixoes.", # Comixão pluralized and missing accent
        "expected": ["erupções cutâneas nodulares", "comichão"]
    },

    # ── 11. Preposition Variants (Testing _PREP_GROUP logic) ────────────────
    {
        "input": "Dor no ombro esquerdo e dor do joelho direito.",
        "expected": ["dor no ombro", "dor no joelho"]
    },
    {
        "input": "Dores da bacia e dores de rins.",
        "expected": ["dor na articulação da anca", "dor nas costas"]
    },
    {
        "input": "Pontada da barriga e pontada de estomago.",
        "expected": ["dor na barriga", "dor de estômago"]
    },
    {
        "input": "Placas de garganta.", # Original is 'placas na garganta'
        "expected": ["manchas na garganta"]
    },
    
    # ── 12. Dense Clinical Phrasing ─────────────────────────────────────────
    {
        "input": "Utente recorre ao SU por astenia, anorexia e emagrecimento nos últimos 2 meses.",
        "expected": ["fadiga", "perda de apetite", "perda de peso", "emagrecimento"]
        # Note: 'emagrecimento' is in DEFAULT_SYMPTOMS and 'perda de peso' maps from 'emagrecimento' in SYNONYM_MAP.
        # This will extract both if they are mapped that way.
    },
    {
        "input": "Refere dispneia para médios esforços, ortopneia e edema nas extremidades.",
        "expected": ["falta de ar", "dispneia", "extremidades inchadas"]
    },
    {
        "input": "Quadro de febre, odinofagia, rinorreia e mialgia de início súbito.",
        "expected": ["febre", "odinofagia", "rinorreia", "dor muscular", "mialgia"]
    },
    {
        "input": "Paciente com história de diabetes desregulada, poliúria e polidipsia.",
        "expected": ["nível de açúcar irregular", "poliúria", "polidipsia"]
    },
    {
        "input": "Exame objetivo: icterícia, ar cadavérico, bócio e adenopatias.",
        "expected": ["pele amarelada", "icterícia", "aspeto tóxico (tifo)", "tiróide aumentada"]
    },

    # ── 13. High-Noise / Rambling Patient Phrasing ──────────────────────────
    {
        "input": "Olhe senhor doutor, ando muito esquecido, a minha cabeça parece que anda à roda, deito tudo fora o que como e não consigo obrar.",
        "expected": ["movimentos de rotação", "vómitos", "obstipação"]
    },
    {
        "input": "Eu já não tenho paciência, ando com o pavio curto, qualquer coisa me irrita, passo-me dos nervos e fico logo em pulgas.",
        "expected": ["irritabilidade", "agitação"]
    },
    {
        "input": "A minha urina está esquisita, sai um chichi muito escuro e fedorento, e sinto a bexiga tão pesada que parece que vou urinar nas calças.",
        "expected": ["urina escura", "odor fétido na urina", "desconforto na bexiga", "perdas urinárias"]
    },
    {
        "input": "Tenho andado a secar, perdi muitos quilos e a minha roupa já não me serve, pareço um esqueleto. Também tenho a garganta infetada.",
        "expected": ["perda de peso", "manchas na garganta"]
    },
    {
        "input": "Acordo de noite banhado em suor, cheio de calafrios, com o corpo a ferver e sem forças nenhumas.",
        "expected": ["sudorese", "calafrios", "febre alta", "fadiga"]
    },

    # ── 14. Extreme Edge Cases & Tricky Negations ───────────────────────────
    {
        "input": "A paciente diz que a mãe teve cancro, mas ela própria nega perda de peso ou fadiga crônica.",
        "expected": ["fadiga"] 
        # 'ou' resets the negation window, so 'fadiga' might slip through.
    },
    {
        "input": "Sem febre. Sem tosse. Sem dispneia.",
        "expected": []
    },
    {
        "input": "Apesar de ter comichão, nunca teve manchas na pele.",
        "expected": ["comichão"]
    },
    {
        "input": "Tosse? Não. Febre? Não. Vontade de vomitar? Sim.",
        "expected": ["tosse", "febre", "náuseas"]
        # The ? is a clause boundary, so the backwards negation scan won't bridge the gap.
        # "Tosse? Não." -> 'tosse' happens BEFORE the negation anyway, so it gets extracted!
    },
    {
        "input": "Diz que não tem dores de cabeça, nem dores abdominais, nem diarreia, nem vómitos.",
        "expected": []
    }
]

# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------
def run_tests():
    print("=" * 60)
    print("  Running NLP Symptom Extractor Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, test in enumerate(TEST_CASES, 1):
        input_text = test["input"]
        expected = set(test["expected"]) # Use sets for order-independent comparison
        
        # Run the extractor
        # Assuming you want to test the default plain string extraction
        attained_list = extract_symptoms(input_text)
        attained = set(attained_list)

        if attained == expected:
            passed += 1
            print(f"✅ Test {i:02d} [PASS]")
        else:
            failed += 1
            print(f"❌ Test {i:02d} [FAIL]")
            print(f"   Input    : {input_text!r}")
            print(f"   Expected : {sorted(list(expected))}")
            print(f"   Attained : {sorted(list(attained))}")
            print("-" * 60)

    print("=" * 60)
    print(f"  Results: {passed} Passed | {failed} Failed | {len(TEST_CASES)} Total")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    # Warm up pipeline to prevent initialization delay during the first test
    extract_symptoms("aquecimento") 
    run_tests()
