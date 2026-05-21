import sys
import os
import streamlit as st

# Ensure the root directory is in the path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.nlp_extractor import extract_symptoms
from src.ml_trainer import predict_top3, EXPECTED_FEATURES
from src.triage_logic import determine_urgency
from src.followup import (
    needs_followup,
    get_followup_questions,
    apply_answers,
    MAX_ROUNDS,
    FOLLOWUP_THRESHOLD,
)

st.set_page_config(page_title="SNS24 - Triagem e Diagnóstico", page_icon="assets/sns24-logo.png")

# Theming and CSS injections for "secondary" colors
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Roboto', 'Segoe UI', sans-serif;
    }
    .urgency-1 { color: #ffffff; background-color: #d73512; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .urgency-2 { color: #ffffff; background-color: #eabd35; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .urgency-3 { color: #333333; background-color: #f7e04f; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .urgency-4 { color: #ffffff; background-color: #0f9e59; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .urgency-5 { color: #ffffff; background-color: #0c72ba; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def init_state():
    if 'step' not in st.session_state:
        st.session_state.step = 'symptoms'
    if 'extracted_symptoms' not in st.session_state:
        st.session_state.extracted_symptoms = []
    if 'feature_vector' not in st.session_state:
        st.session_state.feature_vector = {feature: 0 for feature in EXPECTED_FEATURES}
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'asked_features' not in st.session_state:
        st.session_state.asked_features = set()
    if 'round_num' not in st.session_state:
        st.session_state.round_num = 1
    if 'followup_questions' not in st.session_state:
        st.session_state.followup_questions = []

def reset_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def main():
    init_state()
    
    st.logo("assets/sns24-logo.png", size="large")
    
    col1, col2 = st.columns([1.6, 4], vertical_alignment="center")
    with col1:
        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
        st.image("assets/sns24-logo.png", use_container_width=True)
    with col2:
        st.title("Triagem e Diagnóstico")

    if st.session_state.step == "symptoms":
        st.subheader("Passo 1: Descrição dos Sintomas")
        st.write("Por favor, forneça uma descrição o mais precisa e completa possível dos seus sintomas. No caso de dor, indique a localização da forma mais precisa que conseguir.")
        
        with st.form("symptoms_form"):
            patient_text = st.text_area("Descreva os seus sintomas:", placeholder="Ex: Tenho dor de cabeça há 2 dias...", value=None)
            submitted = st.form_submit_button("Extrair Sintomas")
            
            if submitted:
                if not patient_text or not patient_text.strip():
                    st.error("Por favor, insira uma descrição dos sintomas.")
                else:
                    with st.spinner("A extrair sintomas..."):
                        extracted = extract_symptoms(patient_text)
                    if not extracted:
                        st.warning("Nenhum sintoma reconhecido. Por favor, tente novamente com termos diferentes.")
                    else:
                        st.session_state.extracted_symptoms = extracted
                        st.session_state.step = "demographics"
                        st.rerun()

    elif st.session_state.step == 'demographics':
        st.subheader("Passo 2: Informações Adicionais")
        st.success(f"Sintomas detetados: {', '.join(st.session_state.extracted_symptoms)}")
        st.write("Por favor, preencha alguns detalhes adicionais para ajudar no diagnóstico.")
        
        with st.form("demographics_form"):
            age = st.number_input("Idade", min_value=0, max_value=120, value=None, step=1, placeholder="Introduza a sua idade...")
            gender_choice = st.selectbox("Género", options=["Masculino", "Feminino"], index=None, placeholder="Selecione o género...")
            duration = st.number_input("Duração dos sintomas (em dias)", min_value=0, max_value=365, value=None, step=1, placeholder="Introduza a duração em dias...")
            pain_intensity = st.slider("Intensidade da dor (0 = Sem dor, 10 = Dor máxima)", min_value=0, max_value=10, value=0, step=1)
            
            submitted = st.form_submit_button("Prever Diagnóstico")
            if submitted:
                if age is None or gender_choice is None or duration is None:
                    st.error("Por favor, preencha todos os campos do formulário.")
                else:
                    ml_ready_symptoms = [sym.replace(" ", "_") for sym in st.session_state.extracted_symptoms]
                    for sym in ml_ready_symptoms:
                        if sym in st.session_state.feature_vector:
                            st.session_state.feature_vector[sym] = 1
                    
                    st.session_state.feature_vector['age_group'] = age
                    st.session_state.feature_vector['gender'] = 0 if gender_choice == "Masculino" else 1
                    st.session_state.feature_vector['duration'] = min(duration, 2)
                    st.session_state.feature_vector['pain_intensity'] = pain_intensity

                    with st.spinner("A calcular possíveis diagnósticos..."):
                        try:
                            st.session_state.predictions = predict_top3(st.session_state.feature_vector)
                            if needs_followup(st.session_state.predictions):
                                st.session_state.step = "followup"
                            else:
                                st.session_state.step = "results"
                            st.rerun()
                        except FileNotFoundError:
                            st.error("Modelo não encontrado. Por favor treine o modelo primeiro.")

    elif st.session_state.step == "followup":
        st.subheader(f"Passo 3: Questões de Acompanhamento (Ronda {st.session_state.round_num} de {MAX_ROUNDS})")
        
        st.info("Resultados provisórios atuais:")
        for condition, prob, _ in st.session_state.predictions:
            st.write(f"- **{condition}**: {prob:.1f}%")
        
        if not st.session_state.followup_questions:
            st.session_state.followup_questions = get_followup_questions(
                st.session_state.feature_vector, 
                st.session_state.predictions, 
                st.session_state.asked_features
            )
            if not st.session_state.followup_questions:
                st.session_state.step = "results"
                st.rerun()
        
        st.write("Para melhorar o diagnóstico, por favor responda às seguintes questões:")
        
        with st.form(f"followup_form_{st.session_state.round_num}"):
            answers = {}
            for q in st.session_state.followup_questions:
                ans = st.radio(q['question_pt'], options=["Sim", "Não"], index=None, key=f"q_{q['feature']}")
                answers[q['feature']] = ans
            
            submitted = st.form_submit_button("Submeter Respostas")
            if submitted:
                if None in answers.values():
                    st.error("Por favor, responda a todas as questões.")
                else:
                    processed_answers = {k: (1 if v == "Sim" else 0) for k, v in answers.items()}
                    for feat in processed_answers.keys():
                        st.session_state.asked_features.add(feat)
                    
                    st.session_state.feature_vector = apply_answers(st.session_state.feature_vector, processed_answers)
                    st.session_state.predictions = predict_top3(st.session_state.feature_vector)
                    
                    st.session_state.round_num += 1
                    st.session_state.followup_questions = []
                    
                    if st.session_state.round_num > MAX_ROUNDS or not needs_followup(st.session_state.predictions):
                        st.session_state.step = "results"
                    st.rerun()

    elif st.session_state.step == "results":
        st.subheader("=== Resultados do Diagnóstico ===")
        top_prob = st.session_state.predictions[0][1]
        
        if top_prob < FOLLOWUP_THRESHOLD:
            st.warning(f"A confiança está abaixo de {FOLLOWUP_THRESHOLD:.0f}% — resultado apenas indicativo.")
            
        urgency_colors = {
            1: ("Emergência (Vermelho)", "urgency-1"),
            2: ("Muito Urgente (Laranja)", "urgency-2"),
            3: ("Urgente (Amarelo)", "urgency-3"),
            4: ("Pouco Urgente (Verde)", "urgency-4"),
            5: ("Não Urgente (Azul)", "urgency-5")
        }
        
        for i, (condition, prob, _) in enumerate(st.session_state.predictions, 1):
            urgency_level = determine_urgency(condition)
            urgency_name, urgency_class = urgency_colors.get(urgency_level, ("Desconhecido", ""))
            
            st.markdown(f"**{i}. {condition} ({prob:.1f}%)**")
            st.markdown(f"<span class='{urgency_class}'>Nível de Urgência {urgency_level}: {urgency_name}</span>", unsafe_allow_html=True)
            st.write("---")

        if st.button("Começar de Novo"):
            reset_state()
            st.rerun()

if __name__ == '__main__':
    main()
