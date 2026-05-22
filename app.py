"""
SNS24 — Sistema de Triagem Automática
Streamlit Dashboard  ·  app.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.triage_logic import determine_urgency

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config  (must be FIRST Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SNS24 · Triagem e Encaminhamento",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports after path setup ─────────────────────────────────────────────
try:
    from src.nlp_extractor import extract_symptoms
    from src.routing import get_nearest_facilities
    from streamlit_geolocation import streamlit_geolocation
    from src.ml_trainer import (
        MODEL_DIR,
        encode_labels,
        get_differentiating_symptoms,
        load_data,
        load_model,
        predict_top3,
        split_data,
    )
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    _IMPORTS_OK = True
    _IMPORT_ERR: str | None = None
except Exception as _exc:
    _IMPORTS_OK = False
    _IMPORT_ERR = str(_exc)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_LABELS: dict[str, str] = {
    "random_forest.pkl":      "Random Forest",
    "gradient_boosting.pkl":  "Gradient Boosting",
    "logistic_regression.pkl": "Logistic Regression",
}
MODEL_COLORS: dict[str, str] = {
    "random_forest.pkl":      "#38BDF8",  # Sky blue accent
    "gradient_boosting.pkl":  "#818CF8",  # Indigo accent
    "logistic_regression.pkl": "#34D399",  # Emerald accent
}
MAX_ROUNDS            = 5
CONFIDENCE_THRESHOLD  = 70.0
QUESTIONS_PER_ROUND   = 2


# ══════════════════════════════════════════════════════════════════════════════
# CSS  —  Pure Black Theme
# ══════════════════════════════════════════════════════════════════════════════
def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg:       #000000; /* Pure Black */
    --bg2:      #0A0A0A; /* Very dark gray for sidebar */
    --bg3:      #171717; /* Subtle highlight */
    --card:     #0A0A0A; /* Card background */
    --border:   #262626; /* Neutral border */
    --accent:   #38BDF8; /* Sky 400 */
    --accent2:  #818CF8;
    --accent3:  #34D399;
    --text:     #FAFAFA; /* Near white */
    --text2:    #A3A3A3; /* Neutral gray */
    --success:  #10B981;
    --warn:     #F59E0B;
    --danger:   #EF4444;
    --font:     'Roboto', sans-serif;
    --mono:     'JetBrains Mono', monospace;
}

/* ── Reset / Global ─────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}
* { box-sizing: border-box; }

/* ── Sidebar ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* ── Typography ─────────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    color: var(--text) !important;
    font-family: var(--font) !important;
}
p, label, span, div { font-family: var(--font); }

/* ── Buttons ────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: var(--font) !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    border: 1px solid var(--border) !important;
    background: var(--bg3) !important;
    color: var(--text) !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--bg2) !important;
}
.btn-primary .stButton > button {
    background: var(--accent) !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 600 !important;
}
.btn-primary .stButton > button:hover {
    background: #7DD3FC !important;
    color: #000000 !important;
}
.btn-yes .stButton > button {
    background: rgba(16,185,129,0.1) !important;
    border-color: rgba(16,185,129,0.3) !important;
    color: #34D399 !important;
}
.btn-yes .stButton > button:hover {
    background: rgba(16,185,129,0.2) !important;
    color: #10B981 !important;
}
.btn-no .stButton > button {
    background: rgba(239,68,68,0.1) !important;
    border-color: rgba(239,68,68,0.3) !important;
    color: #F87171 !important;
}
.btn-no .stButton > button:hover {
    background: rgba(239,68,68,0.2) !important;
    color: #EF4444 !important;
}

/* ── Text area ──────────────────────────────────────────────────────── */
.stTextArea textarea {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
    font-size: 0.95rem !important;
    border-radius: 6px !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}
.stTextArea label { color: var(--text2) !important; font-size: 0.85rem !important; }

/* ── Radio ──────────────────────────────────────────────────────────── */
.stRadio label { color: var(--text2) !important; font-size: 0.85rem !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
.stRadio div[role="radiogroup"] label { 
    padding: 8px 12px !important;
    border-radius: 6px !important;
    margin: 2px 0 !important;
}

/* ── Hide chrome ────────────────────────────────────────────────────── */
#MainMenu, footer { visibility: hidden !important; }
header { background: transparent !important; }
[data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 2.5rem !important; }

/* ── Custom layout components ────────────────────────────────────────  */
.sns-logo-area {
    padding: 24px 20px 16px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}
.sns-subtitle {
    color: var(--text2);
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 4px 0 0;
}

.nav-section-label {
    color: var(--text2);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 16px 20px 8px;
}

.sidebar-model-status {
    margin: 0 16px;
    padding: 12px 14px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.8rem;
}
.sms-label { color: var(--text2); font-size: 0.7rem; text-transform: uppercase; margin-bottom: 4px; }
.sms-model { color: var(--text); font-weight: 500; }

.status-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.dot-ok  { background: var(--success); }
.dot-off { background: var(--text2); }

.page-header { margin-bottom: 28px; }
.page-title  {
    font-size: 1.6rem; font-weight: 600;
    color: var(--text); margin: 0; line-height: 1.2;
}
.page-sub    { font-size: 0.9rem; color: var(--text2); margin-top: 6px; }

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
}
.card-accent { border-top: 3px solid var(--accent); }

/* symptom chips */
.symptom-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 0.8rem;
    color: var(--text);
    margin: 4px 4px 4px 0;
    font-family: var(--mono);
}

/* diagnosis cards */
.diag-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.diag-card.top {
    border-color: var(--accent);
    background: var(--bg);
}
.diag-name { font-size: 1.05rem; font-weight: 500; color: var(--text); }
.diag-prob { font-family: var(--mono); font-size: 1.2rem; font-weight: 600; }
.diag-bar-bg  {
    height: 4px; background: var(--bg3);
    border-radius: 2px; margin-top: 12px; overflow: hidden;
}
.diag-bar-fill { height: 100%; border-radius: 2px; }

/* confidence display */
.conf-display {
    text-align: center;
    padding: 20px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 16px;
}
.conf-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    font-weight: 600;
    color: var(--text2);
    margin-bottom: 8px;
}
.conf-value          { font-family: var(--mono); font-size: 2.2rem; font-weight: 600; color: var(--warn); }
.conf-value.high     { color: var(--success); }
.conf-subtext        { font-size: 0.8rem; color: var(--text2); margin-top: 4px; }

/* ── Flashcard ───────────────────────────────────────────────────────── */
.flashcard {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px;
    position: relative;
}
.flashcard-badge {
    position: absolute;
    top: -10px; left: 20px;
    background: var(--bg3);
    color: var(--text);
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 4px;
    border: 1px solid var(--border);
}
.flashcard-round {
    font-size: 0.75rem;
    color: var(--text2);
    margin-bottom: 12px;
}
.flashcard-q {
    font-size: 1.1rem;
    font-weight: 500;
    color: var(--text);
    line-height: 1.6;
    margin: 12px 0 24px;
}
.flashcard-sym {
    color: var(--accent);
    font-weight: 600;
}
.fp-dots { display: flex; gap: 4px; margin-bottom: 16px; }
.fp-dot  { height: 4px; border-radius: 2px; flex: 1; background: var(--bg3); }
.fp-dot.done    { background: var(--accent); }
.fp-dot.current { background: var(--accent); opacity: 0.4; }

/* ── Results ─────────────────────────────────────────────────────────── */
.result-rank { 
    font-size: 0.85rem; 
    color: var(--text2); 
    font-weight: 600; 
    margin-right: 12px;
}
.warning-box {
    background: rgba(245, 158, 11, 0.05);
    border-left: 3px solid var(--warn);
    padding: 16px;
    margin-top: 24px;
    border-radius: 0 4px 4px 0;
}

/* ── Benchmarks ──────────────────────────────────────────────────────── */
.bench-model-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 16px;
}
.bench-color-dot {
    width: 12px; height: 12px;
    border-radius: 2px;
}
.bench-model-name { font-size: 1.1rem; font-weight: 500; color: var(--text); }

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px;
    text-align: center;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text2);
    margin-bottom: 8px;
}
.metric-value {
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 600;
}

.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 40px 0 30px;
}
.section-title {
    font-size: 1.1rem;
    color: var(--text);
    margin-bottom: 20px;
    font-weight: 500;
}

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar        { width: 6px; height: 6px; }
::-webkit-scrollbar-track  { background: var(--bg); }
::-webkit-scrollbar-thumb  { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text2); }

/* ── Streamlit overrides ─────────────────────────────────────────────── */
[data-testid="stMetricValue"] { font-family: var(--mono) !important; }
.streamlit-expanderHeader {
    background: var(--card) !important;
    border-color: var(--border) !important;
    border-radius: 6px !important;
}
.stAlert { border-radius: 6px !important; }
div[data-testid="stVerticalBlock"] > div > div > div { background: transparent !important; }
</style>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def init_state() -> None:
    defaults: dict = {
        "page":               "diagnosis",
        "phase":              "setup",
        "selected_model":     None,
        "patient_text":       "",
        "extracted_symptoms": [],
        "feature_vector":     {},
        "questions":          [],
        "current_q_idx":      0,
        "asked_symptoms":     set(),
        "predictions":        [],
        "top_prob":           0.0,
        "question_rounds":    0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_diagnosis() -> None:
    for k in [
        "phase", "patient_text", "extracted_symptoms", "feature_vector",
        "questions", "current_q_idx", "asked_symptoms",
        "predictions", "top_prob", "question_rounds",
    ]:
        if k in st.session_state:
            del st.session_state[k]


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def get_available_models() -> list[str]:
    model_dir = Path(MODEL_DIR)
    if not model_dir.exists():
        return []
    return sorted(f for f in os.listdir(model_dir) if f.endswith(".pkl"))


@st.cache_data(show_spinner=False)
def evaluate_all_models() -> dict:
    """Evaluate every saved .pkl on the held-out test split. Cached."""
    out: dict = {}
    try:
        X, y, features = load_data("combined")
        y_enc, le = encode_labels(y)
        _, X_test, _, y_test = split_data(X, y_enc)
        classes = le.classes_
    except Exception as exc:
        return {"_error": str(exc)}

    for fname in get_available_models():
        label = MODEL_LABELS.get(fname, fname.replace(".pkl", "").replace("_", " ").title())
        try:
            clf, _le, _feat = load_model(fname)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            pre = precision_score(y_test, y_pred, average="macro", zero_division=0)
            cm  = confusion_matrix(y_test, y_pred)
            per_class_f1  = f1_score(y_test, y_pred, average=None, zero_division=0)
            per_class_rec = recall_score(y_test, y_pred, average=None, zero_division=0)
            per_class_pre = precision_score(y_test, y_pred, average=None, zero_division=0)

            out[fname] = {
                "label": label, "accuracy": acc, "f1": f1, "recall": rec,
                "precision": pre, "cm": cm, "classes": classes,
                "per_class_f1": per_class_f1,
                "per_class_rec": per_class_rec,
                "per_class_pre": per_class_pre,
            }
        except Exception as exc:
            out[fname] = {"label": label, "_error": str(exc)}

    return out


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar() -> None:
    with st.sidebar:
        # ── Logo ──────────────────────────────────────────────────────────────
        st.markdown('<div class="sns-logo-area">', unsafe_allow_html=True)
        
        logo_path = ROOT / "assets" / "sns24-logo.png"
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
        else:
            st.markdown(
                '<span style="font-size:1.4rem;font-weight:700;'
                'color:var(--text);letter-spacing:0.02em;">SNS24</span>',
                unsafe_allow_html=True,
            )
            
        st.markdown(
            '<p class="sns-subtitle">Triagem e Encaminhamento</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Navigation ────────────────────────────────────────────────────────
        st.markdown('<p class="nav-section-label">Navegação</p>', unsafe_allow_html=True)
        page = st.session_state.page

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(
                "Sintomas", use_container_width=True,
                type="primary" if page == "diagnosis" else "secondary",
            ):
                st.session_state.page = "diagnosis"
                st.rerun()
        with col_b:
            if st.button(
                "Métricas", use_container_width=True,
                type="primary" if page == "benchmarks" else "secondary",
            ):
                st.session_state.page = "benchmarks"
                st.rerun()

        st.markdown("<hr class='section-divider' style='margin: 20px 0;'>", unsafe_allow_html=True)

        # ── Session status ────────────────────────────────────────────────────
        if st.session_state.phase not in ("setup",):
            model_name = st.session_state.selected_model or ""
            label = MODEL_LABELS.get(model_name, model_name)

            top_prob = st.session_state.top_prob
            conf_col = "var(--success)" if top_prob >= CONFIDENCE_THRESHOLD else "var(--warn)" if top_prob > 0 else "var(--text2)"

            st.markdown(
                f"""
                <div class="sidebar-model-status">
                    <div class="sms-label">Modelo Ativo</div>
                    <div class="sms-model">{label or "—"}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if top_prob > 0:
                st.markdown(
                    f"""
                    <div class="sidebar-model-status" style="margin-top:8px;">
                        <div class="sms-label">Certeza do Sistema</div>
                        <div class="sms-model" style="color:{conf_col};
                             font-family:var(--mono);">{top_prob:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Iniciar Nova Triagem", use_container_width=True):
                reset_diagnosis()
                st.rerun()

        # ── Model availability ────────────────────────────────────────────────
        st.markdown("<hr class='section-divider' style='margin: 20px 0;'>", unsafe_allow_html=True)
        st.markdown('<p class="nav-section-label">Estado do Sistema</p>', unsafe_allow_html=True)

        models = get_available_models()
        if models:
            for fname in ["random_forest.pkl", "gradient_boosting.pkl", "logistic_regression.pkl"]:
                found = fname in models
                dot = "dot-ok" if found else "dot-off"
                lbl = MODEL_LABELS.get(fname, fname)
                status_txt = "Online" if found else "Offline"
                st.markdown(
                    f'<p style="font-size:0.8rem;color:var(--text2);margin:6px 16px;">'
                    f'<span class="status-dot {dot}"></span>{lbl}'
                    f'<span style="float:right;font-size:0.75rem;">{status_txt}</span></p>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<p style="font-size:0.8rem;color:var(--danger);margin:6px 16px;">'
                "Sistemas temporariamente offline.</p>",
                unsafe_allow_html=True,
            )

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="position:fixed;bottom:16px;left:0;width:240px;
                        text-align:center;color:var(--text2);font-size:0.7rem;">
                SNS24 · Triagem Automática
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSIS  —  phase router
# ══════════════════════════════════════════════════════════════════════════════
def render_diagnosis() -> None:
    phase = st.session_state.phase
    {
        "setup":       _phase_setup,
        "input":       _phase_input,
        "extracted":   _phase_extracted,
        "questioning": _phase_questioning,
        "results":     _phase_results,
    }.get(phase, _phase_setup)()


# ── Phase 0: model selection ───────────────────────────────────────────────────
def _phase_setup() -> None:
    models = get_available_models()

    st.markdown(
        """
        <div class="page-header">
            <p class="page-title">Triagem Digital de Sintomas</p>
            <p class="page-sub">Sistema de análise rápida para o recomendar ao serviço de saúde mais adequado</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not models:
        st.error("Sistemas offline no momento. Por favor, contacte a linha 808 24 24 24.")
        return

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(
            '<div style="border-top: 3px solid var(--accent); padding-top: 16px; margin-bottom: 12px;">'
            '<p style="font-weight:500;color:var(--text);margin:0;">Selecione o Motor de Análise</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        selected = st.radio(
            "modelo",
            models,
            format_func=lambda m: MODEL_LABELS.get(m, m),
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        if st.button("Iniciar Avaliação", use_container_width=True):
            st.session_state.selected_model = selected
            st.session_state.phase = "input"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="card">
                <p style="font-weight:500;color:var(--text);margin-bottom:12px;">Como utilizar?</p>
                <p style="color:var(--text2);font-size:0.9rem;line-height:1.8;margin:0;">
                    <b style="color:var(--text);">1.</b> Descreva, pelas suas próprias palavras, o que está a sentir.<br>
                    <b style="color:var(--text);">2.</b> O nosso sistema vai procurar compreender a sua situação.<br>
                    <b style="color:var(--text);">3.</b> Avaliaremos cenários possíveis baseados na sua descrição.<br>
                    <b style="color:var(--text);">4.</b> Faremos algumas perguntas extra para termos mais a certeza.<br>
                    <b style="color:var(--text);">5.</b> Mostramos-lhe as opções prováveis para decidir os próximos passos.
                </p>
            </div>
            <div class="warning-box" style="margin-top:0;">
                <span style="color:var(--warn);font-weight:600;font-size:0.85rem;">Em caso de emergência ligue 112</span>
                <p style="color:var(--text2);font-size:0.85rem;margin:6px 0 0;line-height:1.5;">
                    Este sistema é um assistente automático de avaliação de sintomas. Não substitui o conselho ou diagnóstico de um médico ou enfermeiro.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Phase 1: symptom text input ────────────────────────────────────────────────
def _phase_input() -> None:
    model   = st.session_state.selected_model
    label   = MODEL_LABELS.get(model, model)
    color   = MODEL_COLORS.get(model, "var(--accent)")

    st.markdown(
        f"""
        <div class="page-header">
            <p class="page-title">Como o podemos ajudar?</p>
            <p class="page-sub">Motor ativo: 
                <b style="color:{color};">{label}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 1], gap="large")

    with left:
        text = st.text_area(
            "Descreva a condição clínica",
            placeholder=(
                "Ex: «Tenho tido febre alta há dois dias, dores de cabeça muito intensas e sinto-me extremamente cansado. "
                "Também tenho algumas dores nas articulações, falta de apetite e sinto-me enjoado.»"
            ),
            height=180,
            label_visibility="collapsed",
            key="symptom_text_input",
        )

        c1, _ = st.columns([1, 3])
        with c1:
            st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
            analyse = st.button("Analisar Sintomas", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="card">
                <p style="font-weight:500;color:var(--text);font-size:0.9rem;margin-bottom:12px;">Dicas</p>
                <p style="color:var(--text2);font-size:0.85rem;line-height:1.7;margin:0;">
                    • Escreva como se estivesse a falar com um médico.<br>
                    • Tente mencionar há quanto tempo começaram os sintomas.<br>
                    • Refira a zona do corpo, se for dor.<br>
                    • O sistema tenta perceber mesmo as palavras mais informais.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyse:
        if not text.strip():
            st.warning("Por favor, descreva o que está a sentir antes de continuar.")
            return
        with st.spinner("A analisar os seus sintomas..."):
            try:
                symptoms = extract_symptoms(text)
            except Exception as exc:
                st.error(f"Falha no processamento: {exc}")
                return

        if not symptoms:
            st.error(
                "Não conseguimos identificar os sintomas exatos da sua descrição. Tente usar outras palavras."
            )
            return

        st.session_state.patient_text       = text
        st.session_state.extracted_symptoms = symptoms
        st.session_state.feature_vector     = {s.replace(" ", "_"): 1 for s in symptoms}
        st.session_state.phase              = "extracted"
        st.rerun()


# ── Phase 2: NLP results + initial prediction ──────────────────────────────────
def _phase_extracted() -> None:
    symptoms = st.session_state.extracted_symptoms
    fv       = st.session_state.feature_vector
    model    = st.session_state.selected_model

    if not st.session_state.predictions:
        with st.spinner("A avaliar cenários possíveis..."):
            try:
                preds = predict_top3(fv, model_name=model)
                st.session_state.predictions = preds
                st.session_state.top_prob    = preds[0][1]
            except Exception as exc:
                st.error(f"Erro de inferência: {exc}")
                return

    preds    = st.session_state.predictions
    top_prob = st.session_state.top_prob

    st.markdown(
        """
        <div class="page-header">
            <p class="page-title">O que detetámos até agora</p>
            <p class="page-sub">Sintomas identificados e possíveis cenários</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([55, 40], gap="large")

    with left:
        st.markdown(
            '<p style="font-size:0.75rem;text-transform:uppercase;font-weight:600;'
            'color:var(--text2);margin-bottom:12px;">Sintomas Registados</p>',
            unsafe_allow_html=True,
        )
        chips = " ".join(
            f'<span class="symptom-chip">{s.replace("_"," ").capitalize()}</span>'
            for s in symptoms
        )
        st.markdown(f'<div style="margin-bottom:28px;">{chips}</div>', unsafe_allow_html=True)

        st.markdown(
            '<p style="font-size:0.75rem;text-transform:uppercase;font-weight:600;'
            'color:var(--text2);margin-bottom:12px;">Cenários Possíveis</p>',
            unsafe_allow_html=True,
        )
        bar_colors = ["var(--accent)", "var(--accent2)", "var(--text2)"]
        for i, (cond, prob, _) in enumerate(preds):
            bc   = bar_colors[i]
            top  = "top" if i == 0 else ""
            st.markdown(
                f"""
                <div class="diag-card {top}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span class="result-rank">{i+1}º</span>
                            <span class="diag-name">{cond}</span>
                        </div>
                        <span class="diag-prob" style="color:{bc};">{prob:.1f}%</span>
                    </div>
                    <div class="diag-bar-bg">
                        <div class="diag-bar-fill" style="width:{min(prob,100):.1f}%;
                             background:{bc};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        conf_cls  = "high" if top_prob >= CONFIDENCE_THRESHOLD else ""
        conf_status = "Nível de certeza adequado" if top_prob >= CONFIDENCE_THRESHOLD else "Precisamos de mais informações"

        st.markdown(
            f"""
            <div class="conf-display">
                <div class="conf-label">Certeza do Sistema (Top 1)</div>
                <div class="conf-value {conf_cls}">{top_prob:.1f}%</div>
                <div class="conf-subtext">{conf_status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if top_prob >= CONFIDENCE_THRESHOLD:
            st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
            if st.button("Ver Sugestão de Encaminhamento", use_container_width=True):
                st.session_state.phase = "results"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div class="card" style="margin:0 0 16px;">
                    <p style="color:var(--text2);font-size:0.85rem;line-height:1.6;margin:0;">
                        O sistema regista uma certeza de apenas <b style="color:var(--warn);">{top_prob:.1f}%</b>.<br>
                        Para podermos recomendar o melhor encaminhamento de forma segura, precisamos de lhe fazer algumas questões rápidas.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                qs = get_differentiating_symptoms(
                    fv, model,
                    max_questions=QUESTIONS_PER_ROUND,
                    asked_symptoms=set(),
                )

                if not qs:
                    st.info("O modelo já alcançou a confiança máxima possível. Não há mais perguntas.")
                else:
                    if st.button("Responder a Perguntas", use_container_width=True, type="primary"):
                        st.session_state.questions = qs
                        st.session_state.current_q_idx = 0
                        st.session_state.asked_symptoms = set()
                        st.session_state.question_rounds = 0
                        st.session_state.phase = "questioning"
                        st.rerun()
            with c2:
                if st.button("Avançar para Sugestão", use_container_width=True):
                    st.session_state.phase = "results"
                    st.rerun()


# ── Phase 3: adaptive flashcard triage ─────────────────────────────────────────
def _phase_questioning() -> None:
    questions = st.session_state.questions
    idx       = st.session_state.current_q_idx
    rounds    = st.session_state.question_rounds
    preds     = st.session_state.predictions
    top_prob  = st.session_state.top_prob
    symptoms  = st.session_state.extracted_symptoms

    if idx >= len(questions):
        with st.spinner("A processar as suas respostas..."):
            try:
                new_preds = predict_top3(
                    st.session_state.feature_vector,
                    model_name=st.session_state.selected_model,
                )
                st.session_state.predictions  = new_preds
                st.session_state.top_prob     = new_preds[0][1]
                st.session_state.question_rounds += 1
            except Exception as exc:
                st.error(f"Erro no recálculo: {exc}")
                return

        new_top      = st.session_state.top_prob
        rounds_done  = st.session_state.question_rounds

        if new_top >= CONFIDENCE_THRESHOLD or rounds_done >= MAX_ROUNDS:
            st.session_state.phase = "results"
        else:
            new_qs = get_differentiating_symptoms(
                st.session_state.feature_vector,
                st.session_state.selected_model,
                max_questions=QUESTIONS_PER_ROUND,
                asked_symptoms=st.session_state.asked_symptoms,
            )
            if not new_qs:
                st.session_state.phase = "results"
            else:
                st.session_state.questions     = new_qs
                st.session_state.current_q_idx = 0
        st.rerun()
        return

    st.markdown(
        f"""
        <div class="page-header">
            <p class="page-title">Questões de Confirmação</p>
            <p class="page-sub">Ronda {rounds + 1} de {MAX_ROUNDS} — 
               Pergunta {idx + 1} de {len(questions)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([55, 45], gap="large")

    with left:
        conf_cls = "high" if top_prob >= CONFIDENCE_THRESHOLD else ""
        st.markdown(
            f"""
            <div class="conf-display" style="padding: 16px;">
                <div class="conf-label" style="margin-bottom:4px;">Certeza Atual</div>
                <div class="conf-value {conf_cls}" style="font-size:1.8rem;">{top_prob:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p style="font-size:0.75rem;text-transform:uppercase;font-weight:600;'
            'color:var(--text2);margin:20px 0 10px;">A reavaliar cenários em tempo real</p>',
            unsafe_allow_html=True,
        )
        bar_colors = ["var(--accent)", "var(--accent2)", "var(--text2)"]
        for i, (cond, prob, _) in enumerate(preds[:3]):
            bc = bar_colors[i]
            st.markdown(
                f"""
                <div class="diag-card" style="padding:12px 16px;margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:0.9rem;color:var(--text);">{cond}</span>
                        <span style="font-family:var(--mono);color:{bc};
                              font-weight:600;font-size:0.95rem;">{prob:.1f}%</span>
                    </div>
                    <div class="diag-bar-bg" style="margin-top:8px;">
                        <div class="diag-bar-fill"
                             style="width:{min(prob,100):.1f}%;background:{bc};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        q         = questions[idx]
        q_label   = q.replace("_", " ").capitalize()
        fc_key    = f"fc_{rounds}_{idx}"

        dots = "".join(
            f'<div class="fp-dot {"done" if i < idx else "current" if i == idx else ""}"></div>'
            for i in range(len(questions))
        )

        st.markdown(
            f"""
            <div class="flashcard" id="{fc_key}">
                <div class="flashcard-badge">Avaliação Rápida &nbsp;·&nbsp; {idx + 1}/{len(questions)}</div>
                <div class="flashcard-round">Responda para melhorarmos a nossa sugestão</div>
                <div class="fp-dots">{dots}</div>
                <div class="flashcard-q">
                    Além do que já indicou, também está a sentir<br>
                    <span class="flashcard-sym">"{q_label}"</span>?
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2, _ = st.columns([2, 2, 1])
        with b1:
            st.markdown('<div class="btn-yes">', unsafe_allow_html=True)
            if st.button("Sim", key=f"yes_{fc_key}", use_container_width=True):
                st.session_state.feature_vector[q] = 1
                st.session_state.asked_symptoms.add(q)
                st.session_state.current_q_idx += 1
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with b2:
            st.markdown('<div class="btn-no">', unsafe_allow_html=True)
            if st.button("Não", key=f"no_{fc_key}", use_container_width=True):
                st.session_state.asked_symptoms.add(q)
                st.session_state.current_q_idx += 1
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def _phase_results() -> None:
    preds = st.session_state.predictions
    symptoms = st.session_state.extracted_symptoms
    top_prob = st.session_state.top_prob
    rounds = st.session_state.question_rounds
    model = st.session_state.selected_model
    m_label = MODEL_LABELS.get(model, model)
    m_color = MODEL_COLORS.get(model, "var(--accent)")

    conf_col = "var(--success)" if top_prob >= CONFIDENCE_THRESHOLD else "var(--warn)"

    st.markdown(
        f"""
        <div class="page-header">
            <p class="page-title">Resultado da Triagem</p>
            <p class="page-sub">
                Motor: <b style="color:{m_color};">{m_label}</b>
                &nbsp;·&nbsp; Rondas de Perguntas: <b>{rounds}</b>
                &nbsp;·&nbsp; Certeza Global: <b style="color:{conf_col};">{top_prob:.1f}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        for i, (cond, prob, _) in enumerate(preds):
            top = "top" if i == 0 else ""

            # 1. Use YOUR function to get the level and color!
            u_level, bc = determine_urgency(cond)

            st.markdown(
                f"""
                <div class="diag-card {top}" style="border-left: 4px solid {bc};">
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                margin-bottom:12px;">
                        <div style="display:flex;flex-direction:column;">
                            <div style="display:flex;align-items:center;">
                                <span class="result-rank">{i + 1}º</span>
                                <span style="font-size:1.1rem;font-weight:500;">{cond}</span>
                            </div>
                            <span style="font-size:0.85rem;font-weight:600;color:{bc};margin-top:4px;">
                                Nível de Urgência (Manchester): {u_level}
                            </span>
                        </div>
                        <span style="font-family:var(--mono);font-size:1.4rem;
                                     font-weight:600;color:{bc};">{prob:.1f}%</span>
                    </div>
                    <div class="diag-bar-bg">
                        <div class="diag-bar-fill"
                             style="width:{min(prob, 100):.1f}%;background:{bc};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="warning-box">
                <span style="color:var(--warn);font-weight:600;font-size:0.85rem;">Indicação Importante</span>
                <p style="color:var(--text2);font-size:0.85rem;margin:8px 0 0;line-height:1.6;">
                    Estes resultados são sugestões geradas automaticamente com base no que descreveu. Este sistema não faz diagnósticos oficiais nem substitui a opinião de um profissional de saúde.<br><br>
                    Se necessitar de falar com um profissional imediatamente, ligue <b>808 24 24 24</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if preds:
            names = [p[0] for p in preds]
            probs = [p[1] for p in preds]

            # 2. Use YOUR function to extract just the color (index 1 of the tuple) for the pie chart!
            pie_colors = [determine_urgency(name)[1] for name in names]

            fig = go.Figure(
                go.Pie(
                    labels=names,
                    values=probs,
                    hole=0.65,
                    textinfo="none",
                    marker=dict(
                        colors=pie_colors,
                        line=dict(color="#000000", width=2),
                    ),
                )
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                height=220,
                annotations=[
                    dict(
                        text=f"<b>{probs[0]:.0f}%</b>",
                        x=0.5, y=0.5,
                        font=dict(size=26, color="#FAFAFA", family="JetBrains Mono"),
                        showarrow=False,
                    )
                ],
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            '<p style="font-size:0.75rem;text-transform:uppercase;font-weight:600;'
            'color:var(--text2);margin:20px 0 10px;">Sintomas Finalizados</p>',
            unsafe_allow_html=True,
        )
        chips = " ".join(
            f'<span class="symptom-chip">{s.replace("_", " ").capitalize()}</span>'
            for s in symptoms
        )
        st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="margin-bottom: 16px;">
                <p class="section-title" style="margin-bottom: 4px;">Unidades de Saúde Recomendadas</p>
                <p style="color:var(--text2);font-size:0.85rem;">Partilhe a sua localização para encontrarmos a unidade mais adequada para o seu diagnóstico.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Initialize geolocation widget
        location = streamlit_geolocation()

        if location and location.get('latitude') and location.get('longitude'):
            user_lat = location['latitude']
            user_lon = location['longitude']
            top_condition = preds[0][0] if preds else ""

            with st.spinner("A procurar unidades na sua zona..."):
                facilities = get_nearest_facilities(user_lat, user_lon, top_condition)

            if facilities:
                st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
                for fac in facilities:
                    st.markdown(
                        f"""
                        <div class="diag-card" style="border-left: 3px solid var(--accent);">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <div>
                                    <span class="diag-name">{fac.get('Hospital Name', 'Unidade de Saúde')}</span><br>
                                    <span style="font-size:0.8rem;color:var(--text2);text-transform:uppercase;letter-spacing:0.05em;">
                                        {fac.get('Care Type', 'Clínica')} • {fac.get('Specialty Tags', 'Geral')}
                                    </span>
                                </div>
                                <div style="text-align:right;">
                                    <span style="font-family:var(--mono);font-size:1.4rem;font-weight:600;color:var(--text);">
                                        {fac.get('distance_km', 0):.1f}
                                    </span>
                                    <span style="font-size:0.8rem;color:var(--text2);">km</span>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Não foram encontradas unidades de saúde compatíveis na sua zona.")

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARKS PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_benchmarks() -> None:
    st.markdown(
        """
        <div class="page-header">
            <p class="page-title">Métricas do Sistema</p>
            <p class="page-sub">
                Transparência dos motores de triagem usados (informação técnica)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("A gerar grelha de métricas..."):
        results = evaluate_all_models()

    if "_error" in results:
        st.error(f"Falha no processamento da avaliação: {results['_error']}")
        return

    valid = {k: v for k, v in results.items() if "_error" not in v}
    errored = {k: v for k, v in results.items() if "_error" in v}

    if errored:
        for fname, res in errored.items():
            st.warning(f"Inconsistência identificada no modelo {MODEL_LABELS.get(fname, fname)}: {res['_error']}")

    if not valid:
        st.error("Dados de avaliação insuficientes.")
        return

    metrics_keys   = ["accuracy", "f1", "recall", "precision"]
    metrics_labels = {
        "accuracy": "Accuracy", "f1": "F1-Score (Macro)",
        "recall": "Recall (Macro)", "precision": "Precision (Macro)",
    }

    # ── Per-model metric cards ────────────────────────────────────────────────
    for fname, res in valid.items():
        color = MODEL_COLORS.get(fname, "#38BDF8")
        lbl   = res["label"]
        st.markdown(
            f"""
            <div class="bench-model-header">
                <div class="bench-color-dot" style="background:{color};"></div>
                <span class="bench-model-name">{lbl}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns(4, gap="small")
        for j, mk in enumerate(metrics_keys):
            with cols[j]:
                val = res[mk]
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{metrics_labels[mk]}</div>
                        <div class="metric-value" style="color:{color};">{val:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Grouped comparison bar chart ──────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Análise Comparativa de Desempenho Global</p>', unsafe_allow_html=True)

    fig = go.Figure()
    for fname, res in valid.items():
        fig.add_trace(
            go.Bar(
                name=res["label"],
                x=[metrics_labels[m] for m in metrics_keys],
                y=[res[m] for m in metrics_keys],
                marker_color=MODEL_COLORS.get(fname, "#cccccc"),
                marker_line_width=0,
                text=[f"{res[m]:.3f}" for m in metrics_keys],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11, color="#FAFAFA"),
            )
        )
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#A3A3A3"),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color="#A3A3A3", size=12),
            linecolor="#262626",
        ),
        yaxis=dict(
            range=[0, 1.15],
            showgrid=True, gridcolor="#171717",
            tickfont=dict(color="#A3A3A3"),
            tickformat=".0%",
        ),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=1.1,
            font=dict(color="#FAFAFA", size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── New Addition: F1-Score Distribution (Box Plot) ────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Distribuição de F1-Score por Condição</p>', unsafe_allow_html=True)
    
    box_fig = go.Figure()
    for fname, res in valid.items():
        if "per_class_f1" in res:
            box_fig.add_trace(go.Box(
                y=res["per_class_f1"],
                name=res["label"],
                marker_color=MODEL_COLORS.get(fname, "#cccccc"),
                boxpoints='all',
                jitter=0.4,
                pointpos=-1.8,
                line=dict(width=1),
            ))
            
    box_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#A3A3A3"),
        yaxis=dict(
            title="F1-Score",
            showgrid=True, gridcolor="#171717",
            zeroline=False,
            tickfont=dict(color="#A3A3A3")
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color="#FAFAFA", size=12)
        ),
        showlegend=False,
        margin=dict(l=50, r=20, t=20, b=40),
        height=380,
    )
    st.plotly_chart(box_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Espectro de Performance Multidimensional</p>', unsafe_allow_html=True)

    radar_fig = go.Figure()
    categories = [metrics_labels[m] for m in metrics_keys] + [metrics_labels[metrics_keys[0]]]
    for fname, res in valid.items():
        vals = [res[m] for m in metrics_keys] + [res[metrics_keys[0]]]
        hex_color = MODEL_COLORS.get(fname, "#cccccc")
        h = hex_color.lstrip("#")
        if len(h) == 3:
            h = "".join(c*2 for c in h)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        radar_fig.add_trace(
            go.Scatterpolar(
                r=vals, theta=categories,
                fill="toself", name=res["label"],
                line=dict(color=hex_color, width=1.5),
                fillcolor=f"rgba({r},{g},{b},0.05)",
            )
        )
    radar_fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickfont=dict(color="#A3A3A3", size=10),
                gridcolor="#171717", linecolor="#171717",
            ),
            angularaxis=dict(
                tickfont=dict(color="#FAFAFA", size=11),
                gridcolor="#171717", linecolor="#171717",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#A3A3A3"),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=-0.15,
            font=dict(color="#FAFAFA", size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=40, r=40, t=20, b=70),
        height=380,
    )
    st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Matriz de Confusão Normalizada (Top 30 Condições)</p>', unsafe_allow_html=True)

    valid_cm = [(fn, r) for fn, r in valid.items() if "cm" in r]
    if valid_cm:
        cm_cols = st.columns(len(valid_cm), gap="small")
        for col, (fname, res) in zip(cm_cols, valid_cm):
            with col:
                cm      = res["cm"]
                classes = res["classes"]

                class_counts = cm.sum(axis=1)
                top_idx      = np.argsort(class_counts)[::-1][:30]
                cm_sub       = cm[np.ix_(top_idx, top_idx)]
                cls_sub      = [classes[i] for i in top_idx]

                row_sums = cm_sub.sum(axis=1, keepdims=True).astype(float)
                cm_norm  = np.divide(cm_sub, row_sums, where=row_sums > 0)

                short_cls = [c[:16] + "…" if len(c) > 16 else c for c in cls_sub]

                fig = go.Figure(
                    go.Heatmap(
                        z=cm_norm, x=short_cls, y=short_cls,
                        colorscale=[[0, "#000000"], [0.5, "#1E40AF"], [1, "#38BDF8"]],
                        showscale=False,
                        hovertemplate="Real: %{y}<br>Previsto: %{x}<br>Score: %{z:.2f}<extra></extra>",
                    )
                )
                fig.update_layout(
                    title=dict(
                        text=res["label"],
                        font=dict(size=12, color="#FAFAFA", family="Inter"),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#A3A3A3", size=8),
                    xaxis=dict(showgrid=False, tickangle=45, tickfont=dict(size=8)),
                    yaxis=dict(showgrid=False, tickfont=dict(size=8), autorange="reversed"),
                    margin=dict(l=10, r=10, t=40, b=100),
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Summary comparison table ──────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Quadro Resumo de Métricas</p>', unsafe_allow_html=True)

    rows = []
    for fname, res in valid.items():
        rows.append({
            "Modelo":     res["label"],
            "Accuracy":   f"{res['accuracy']:.4f}",
            "F1-Score":   f"{res['f1']:.4f}",
            "Recall":     f"{res['recall']:.4f}",
            "Precision":  f"{res['precision']:.4f}",
        })

    df_summary = pd.DataFrame(rows).set_index("Modelo")
    st.dataframe(
        df_summary,
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    inject_css()
    init_state()

    if not _IMPORTS_OK:
        st.error(
            f"**Falha de importação:** `{_IMPORT_ERR}`\n\n"
            "Valide a integridade do pacote `src/` e assegure a instalação das "
            "dependências do ambiente virtual (`pip install -r requirements.txt`)."
        )
        st.stop()

    render_sidebar()

    page = st.session_state.page
    if page == "diagnosis":
        render_diagnosis()
    elif page == "benchmarks":
        render_benchmarks()


if __name__ == "__main__":
    main()
