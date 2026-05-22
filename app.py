"""
SNS24 — Sistema de Apoio ao Diagnóstico Inteligente
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

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config  (must be FIRST Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SNS24 · Diagnóstico Inteligente",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports after path setup ─────────────────────────────────────────────
try:
    from src.nlp_extractor import extract_symptoms
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
    "random_forest.pkl":      "#00C2A8",
    "gradient_boosting.pkl":  "#4F8EF7",
    "logistic_regression.pkl": "#A78BFA",
}
MAX_ROUNDS            = 5
CONFIDENCE_THRESHOLD  = 70.0
QUESTIONS_PER_ROUND   = 2


# ══════════════════════════════════════════════════════════════════════════════
# CSS  —  full dark medical theme
# ══════════════════════════════════════════════════════════════════════════════
def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg:       #070C18;
    --bg2:      #0D1427;
    --bg3:      #131D35;
    --card:     #0F1929;
    --border:   #1C2E4A;
    --accent:   #00C2A8;
    --accent2:  #4F8EF7;
    --accent3:  #A78BFA;
    --text:     #DDE8FF;
    --text2:    #6878A0;
    --success:  #10B981;
    --warn:     #F5A623;
    --danger:   #EF4444;
    --font:     'Plus Jakarta Sans', sans-serif;
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
    font-weight: 600 !important;
    border-radius: 9px !important;
    border: 1px solid var(--border) !important;
    background: var(--bg3) !important;
    color: var(--text2) !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(0,194,168,0.07) !important;
    transform: translateY(-1px) !important;
}
.btn-primary .stButton > button {
    background: var(--accent) !important;
    color: #040810 !important;
    border: none !important;
    font-weight: 700 !important;
}
.btn-primary .stButton > button:hover {
    background: #00D9BC !important;
    color: #040810 !important;
    transform: translateY(-1px) !important;
}
.btn-yes .stButton > button {
    background: rgba(16,185,129,0.12) !important;
    border-color: #10B981 !important;
    color: #10B981 !important;
    font-size: 1rem !important;
    padding: 0.55rem 0 !important;
}
.btn-yes .stButton > button:hover {
    background: rgba(16,185,129,0.24) !important;
    color: #10B981 !important;
}
.btn-no .stButton > button {
    background: rgba(239,68,68,0.1) !important;
    border-color: #EF4444 !important;
    color: #EF4444 !important;
    font-size: 1rem !important;
    padding: 0.55rem 0 !important;
}
.btn-no .stButton > button:hover {
    background: rgba(239,68,68,0.22) !important;
    color: #EF4444 !important;
}

/* ── Text area ──────────────────────────────────────────────────────── */
.stTextArea textarea {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
    font-size: 0.95rem !important;
    border-radius: 10px !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,194,168,0.18) !important;
}
.stTextArea label { color: var(--text2) !important; font-size: 0.85rem !important; }

/* ── Radio ──────────────────────────────────────────────────────────── */
.stRadio label { color: var(--text2) !important; font-size: 0.85rem !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
.stRadio div[role="radiogroup"] label { 
    padding: 8px 14px !important;
    border-radius: 8px !important;
    margin: 3px 0 !important;
}

/* ── Hide chrome ────────────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 2rem !important; }

/* ── Custom layout components ────────────────────────────────────────  */

.sns-logo-area {
    padding: 26px 22px 18px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 4px;
}
.sns-subtitle {
    color: var(--text2);
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 5px 0 0;
}

.nav-section-label {
    color: var(--text2);
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 14px 22px 6px;
}

.sidebar-model-status {
    margin: 0 14px;
    padding: 12px 16px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 10px;
    font-size: 0.78rem;
}
.sms-label { color: var(--text2); font-size: 0.7rem; text-transform: uppercase;
             letter-spacing: 0.08em; margin-bottom: 3px; }
.sms-model { color: var(--text); font-weight: 600; }

.status-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}
.dot-ok  { background: var(--success); box-shadow: 0 0 6px #10B981; }
.dot-off { background: var(--text2); }

.page-header { margin-bottom: 24px; }
.page-title  {
    font-size: 1.75rem; font-weight: 800;
    color: var(--text); margin: 0; line-height: 1.2;
}
.page-sub    { font-size: 0.88rem; color: var(--text2); margin-top: 5px; }

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 12px;
}
.card-accent { border-left: 3px solid var(--accent); }

/* symptom chips */
.symptom-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 11px;
    background: rgba(79,142,247,0.11);
    border: 1px solid rgba(79,142,247,0.28);
    border-radius: 20px;
    font-size: 0.78rem;
    color: #7CB9FF;
    margin: 3px 2px;
    font-family: var(--mono);
    letter-spacing: 0.01em;
}

/* diagnosis cards */
.diag-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 10px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.diag-card.top {
    border-color: var(--accent);
    box-shadow: 0 0 20px rgba(0,194,168,0.1);
    background: linear-gradient(135deg, rgba(0,194,168,0.04), var(--card));
}
.diag-name { font-size: 1rem; font-weight: 600; color: var(--text); }
.diag-prob { font-family: var(--mono); font-size: 1.45rem; font-weight: 700; }
.diag-bar-bg  {
    height: 5px; background: var(--bg3);
    border-radius: 3px; margin-top: 10px; overflow: hidden;
}
.diag-bar-fill { height: 100%; border-radius: 3px; }

/* confidence display */
.conf-display {
    text-align: center;
    padding: 18px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 14px;
}
.conf-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text2);
    margin-bottom: 5px;
}
.conf-value          { font-family: var(--mono); font-size: 2.4rem; font-weight: 700; color: var(--warn); }
.conf-value.high     { color: var(--success); }
.conf-subtext        { font-size: 0.73rem; color: var(--text2); margin-top: 3px; }

/* ── Flashcard ───────────────────────────────────────────────────────── */
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(55px) scale(0.97); }
    to   { opacity: 1; transform: translateX(0)   scale(1);    }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 4px 24px rgba(0,194,168,0.12); }
    50%       { box-shadow: 0 4px 36px rgba(0,194,168,0.26); }
}

.flashcard {
    background: var(--card);
    border: 1px solid var(--accent);
    border-radius: 18px;
    padding: 30px 26px 24px;
    position: relative;
    animation: slideInRight 0.38s cubic-bezier(0.25,0.8,0.25,1) both,
               pulseGlow 3s ease-in-out 0.5s infinite;
}
.flashcard-badge {
    position: absolute;
    top: -11px; left: 22px;
    background: var(--accent);
    color: #040810;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 12px;
    border-radius: 12px;
}
.flashcard-round {
    font-size: 0.7rem;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.flashcard-q {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.55;
    margin: 10px 0 20px;
}
.flashcard-sym {
    color: var(--accent);
    font-style: italic;
}
.fp-dots { display: flex; gap: 6px; margin-bottom: 18px; }
.fp-dot  { height: 3px; border-radius: 2px; flex: 1; background: var(--border); }
.fp-dot.done    { background: var(--accent); }
.fp-dot.current { background: var(--accent); opacity: 0.5; }

/* ── Results ─────────────────────────────────────────────────────────── */
.result-medal { font-size: 1.5rem; }
.warning-box {
    background: rgba(245,166,35,0.07);
    border: 1px solid rgba(245,166,35,0.3);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 20px;
}

/* ── Benchmarks ──────────────────────────────────────────────────────── */
.bench-model-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 12px;
}
.bench-color-dot {
    width: 10px; height: 10px;
    border-radius: 3px;
    flex-shrink: 0;
}
.bench-model-name { font-size: 1.05rem; font-weight: 700; color: var(--text); }

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 14px;
    text-align: center;
}
.metric-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text2);
    margin-bottom: 7px;
}
.metric-value {
    font-family: var(--mono);
    font-size: 1.85rem;
    font-weight: 700;
}

.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 28px 0;
}
.section-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--text2);
    margin-bottom: 16px;
    font-weight: 600;
}

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar        { width: 5px; height: 5px; }
::-webkit-scrollbar-track  { background: var(--bg2); }
::-webkit-scrollbar-thumb  { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text2); }

/* ── Streamlit overrides ─────────────────────────────────────────────── */
[data-testid="stMetricValue"] { font-family: var(--mono) !important; }
.streamlit-expanderHeader {
    background: var(--card) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
}
.stAlert { border-radius: 10px !important; }
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
    """Evaluate every saved .pkl on the held-out test split.  Cached."""
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
        logo_path = ROOT / "assets" / "sns24-logo.png"
        st.markdown('<div class="sns-logo-area">', unsafe_allow_html=True)
        if logo_path.exists():
            st.image(str(logo_path), width=120)
        else:
            st.markdown(
                '<span style="font-size:1.5rem;font-weight:800;'
                'color:var(--accent);letter-spacing:-0.02em;">SNS24</span>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<p class="sns-subtitle">Sistema de Apoio ao Diagnóstico</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Navigation ────────────────────────────────────────────────────────
        st.markdown('<p class="nav-section-label">Navegação</p>', unsafe_allow_html=True)
        page = st.session_state.page

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(
                "⚕ Diagnóstico", use_container_width=True,
                type="primary" if page == "diagnosis" else "secondary",
            ):
                st.session_state.page = "diagnosis"
                st.rerun()
        with col_b:
            if st.button(
                "📊 Benchmarks", use_container_width=True,
                type="primary" if page == "benchmarks" else "secondary",
            ):
                st.session_state.page = "benchmarks"
                st.rerun()

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── Session status ────────────────────────────────────────────────────
        if st.session_state.phase not in ("setup",):
            model_name = st.session_state.selected_model or ""
            label = MODEL_LABELS.get(model_name, model_name)

            top_prob = st.session_state.top_prob
            conf_col = "#10B981" if top_prob >= CONFIDENCE_THRESHOLD else "#F5A623" if top_prob > 0 else "#6878A0"

            st.markdown(
                f"""
                <div class="sidebar-model-status">
                    <div class="sms-label">Modelo activo</div>
                    <div class="sms-model">{label or "—"}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if top_prob > 0:
                st.markdown(
                    f"""
                    <div class="sidebar-model-status" style="margin-top:8px;">
                        <div class="sms-label">Confiança actual</div>
                        <div class="sms-model" style="color:{conf_col};
                             font-family:var(--mono);">{top_prob:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("↺ Novo Diagnóstico", use_container_width=True):
                reset_diagnosis()
                st.rerun()

        # ── Model availability ────────────────────────────────────────────────
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown('<p class="nav-section-label">Modelos Disponíveis</p>', unsafe_allow_html=True)

        models = get_available_models()
        if models:
            for fname in ["random_forest.pkl", "gradient_boosting.pkl", "logistic_regression.pkl"]:
                found = fname in models
                dot = "dot-ok" if found else "dot-off"
                lbl = MODEL_LABELS.get(fname, fname)
                status_txt = "Carregado" if found else "Não encontrado"
                st.markdown(
                    f'<p style="font-size:0.75rem;color:var(--text2);margin:5px 14px;">'
                    f'<span class="status-dot {dot}"></span>{lbl}'
                    f'<span style="float:right;font-size:0.68rem;">{status_txt}</span></p>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<p style="font-size:0.75rem;color:var(--danger);margin:5px 14px;">'
                "Nenhum modelo encontrado</p>",
                unsafe_allow_html=True,
            )

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="position:fixed;bottom:14px;left:0;width:238px;
                        text-align:center;color:#243550;font-size:0.68rem;">
                SNS24 · Protótipo Clínico · v1.0
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
            <p class="page-title">⚕ Sistema de Diagnóstico</p>
            <p class="page-sub">Triagem inteligente baseada em NLP + Machine Learning</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not models:
        st.error("Nenhum modelo encontrado. Execute `python src/ml_trainer.py` primeiro.")
        return

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-weight:700;color:var(--text);margin-bottom:12px;">Escolha o Modelo</p>',
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
        if st.button("Iniciar Diagnóstico →", use_container_width=True):
            st.session_state.selected_model = selected
            st.session_state.phase = "input"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="card">
                <p style="font-weight:600;color:var(--text);margin-bottom:10px;">Como funciona?</p>
                <p style="color:var(--text2);font-size:0.85rem;line-height:1.8;margin:0;">
                    <b style="color:var(--accent);">①</b> Descreve os sintomas em linguagem natural<br>
                    <b style="color:var(--accent);">②</b> O NLP extrai os sintomas relevantes<br>
                    <b style="color:var(--accent);">③</b> O modelo ML calcula hipóteses de diagnóstico<br>
                    <b style="color:var(--accent);">④</b> Triagem adaptativa refina a confiança<br>
                    <b style="color:var(--accent);">⑤</b> Diagnóstico final com probabilidades
                </p>
            </div>
            <div class="warning-box" style="margin-top:0;">
                <span style="color:var(--warn);font-weight:600;font-size:0.85rem;">⚠ Aviso Clínico</span>
                <p style="color:var(--text2);font-size:0.8rem;margin:6px 0 0;line-height:1.6;">
                    Este é um protótipo de apoio à decisão. Não substitui avaliação médica profissional.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Phase 1: symptom text input ────────────────────────────────────────────────
def _phase_input() -> None:
    model   = st.session_state.selected_model
    label   = MODEL_LABELS.get(model, model)
    color   = MODEL_COLORS.get(model, "#00C2A8")

    st.markdown(
        f"""
        <div class="page-header">
            <p class="page-title">Descrição dos Sintomas</p>
            <p class="page-sub">Modelo seleccionado: 
                <b style="color:{color};">{label}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 1], gap="large")

    with left:
        text = st.text_area(
            "Descreva os seus sintomas",
            placeholder=(
                "Ex: «Tenho febre alta há dois dias, dores de cabeça intensas e muita fadiga. "
                "Também sinto dores nas articulações, não tenho apetite e tenho estado com náuseas.»"
            ),
            height=170,
            label_visibility="collapsed",
            key="symptom_text_input",
        )

        c1, _ = st.columns([1, 3])
        with c1:
            st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
            analyse = st.button("Analisar Sintomas →", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="card">
                <p style="font-weight:600;color:var(--text);font-size:0.85rem;margin-bottom:10px;">Dicas de Descrição</p>
                <p style="color:var(--text2);font-size:0.78rem;line-height:1.9;margin:0;">
                    • Use linguagem natural em Português<br>
                    • Inclua duração e intensidade<br>
                    • Mencione todos os sintomas<br>
                    • Termos informais são reconhecidos<br>
                    • Pode incluir localização da dor
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyse:
        if not text.strip():
            st.warning("Por favor, descreva os sintomas antes de continuar.")
            return
        with st.spinner("A extrair sintomas via NLP..."):
            try:
                symptoms = extract_symptoms(text)
            except Exception as exc:
                st.error(f"Erro no NLP: {exc}")
                return

        if not symptoms:
            st.error(
                "Nenhum sintoma reconhecido. Tente usar outros termos ou descreva de forma diferente."
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

    # Run initial prediction (only if not already computed for this phase)
    if not st.session_state.predictions:
        with st.spinner("A calcular diagnóstico inicial..."):
            try:
                preds = predict_top3(fv, model_name=model)
                st.session_state.predictions = preds
                st.session_state.top_prob    = preds[0][1]
            except Exception as exc:
                st.error(f"Erro no modelo: {exc}")
                return

    preds    = st.session_state.predictions
    top_prob = st.session_state.top_prob

    st.markdown(
        """
        <div class="page-header">
            <p class="page-title">Análise Preliminar</p>
            <p class="page-sub">Sintomas extraídos e hipóteses iniciais</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([55, 40], gap="large")

    # ── Left: symptoms + predictions ────────────────────────────────────────
    with left:
        st.markdown(
            '<p style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;'
            'color:var(--text2);margin-bottom:8px;">Sintomas Identificados</p>',
            unsafe_allow_html=True,
        )
        chips = " ".join(
            f'<span class="symptom-chip">◆ {s.replace("_"," ").capitalize()}</span>'
            for s in symptoms
        )
        st.markdown(f'<div style="margin-bottom:22px;">{chips}</div>', unsafe_allow_html=True)

        st.markdown(
            '<p style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;'
            'color:var(--text2);margin-bottom:8px;">Hipóteses de Diagnóstico</p>',
            unsafe_allow_html=True,
        )
        bar_colors = ["var(--accent)", "var(--accent2)", "var(--text2)"]
        for i, (cond, prob, _) in enumerate(preds):
            bc   = bar_colors[i]
            top  = "top" if i == 0 else ""
            rank = ["①", "②", "③"][i]
            st.markdown(
                f"""
                <div class="diag-card {top}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="color:var(--text2);font-size:0.85rem;">{rank}</span>
                            <span class="diag-name" style="margin-left:8px;">{cond}</span>
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

    # ── Right: confidence + actions ───────────────────────────────────────────
    with right:
        conf_cls  = "high" if top_prob >= CONFIDENCE_THRESHOLD else ""
        conf_icon = "✓ Confiança suficiente" if top_prob >= CONFIDENCE_THRESHOLD else "↑ Pode ser melhorado com triagem"

        st.markdown(
            f"""
            <div class="conf-display">
                <div class="conf-label">Confiança do diagnóstico</div>
                <div class="conf-value {conf_cls}">{top_prob:.1f}%</div>
                <div class="conf-subtext">{conf_icon}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if top_prob >= CONFIDENCE_THRESHOLD:
            st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
            if st.button("Ver Diagnóstico Final →", use_container_width=True):
                st.session_state.phase = "results"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div class="card" style="margin:0 0 12px;">
                    <p style="color:var(--text2);font-size:0.83rem;line-height:1.65;margin:0;">
                        Confiança actual: <b style="color:var(--warn);">{top_prob:.1f}%</b>.<br>
                        A triagem adaptativa faz perguntas específicas para aumentar a precisão do diagnóstico.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
                if st.button("Iniciar Triagem →", use_container_width=True):
                    qs = get_differentiating_symptoms(
                        fv, model,
                        max_questions=QUESTIONS_PER_ROUND,
                        asked_symptoms=set(),
                    )
                    st.session_state.questions       = qs
                    st.session_state.current_q_idx   = 0
                    st.session_state.asked_symptoms  = set()
                    st.session_state.question_rounds = 0
                    st.session_state.phase           = "questioning"
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                if st.button("Resultado Actual →", use_container_width=True):
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

    # ── All questions answered → recalculate and decide ───────────────────────
    if idx >= len(questions):
        with st.spinner("A recalcular diagnóstico..."):
            try:
                new_preds = predict_top3(
                    st.session_state.feature_vector,
                    model_name=st.session_state.selected_model,
                )
                st.session_state.predictions  = new_preds
                st.session_state.top_prob     = new_preds[0][1]
                st.session_state.question_rounds += 1
            except Exception as exc:
                st.error(f"Erro: {exc}")
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

    # ── Show current question ─────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="page-header">
            <p class="page-title">Triagem Adaptativa</p>
            <p class="page-sub">Ronda {rounds + 1} de {MAX_ROUNDS} — 
               Pergunta {idx + 1} de {len(questions)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([55, 45], gap="large")

    # ── Left: live prediction + symptoms ─────────────────────────────────────
    with left:
        conf_cls = "high" if top_prob >= CONFIDENCE_THRESHOLD else ""
        st.markdown(
            f"""
            <div class="conf-display">
                <div class="conf-label">Confiança actual</div>
                <div class="conf-value {conf_cls}">{top_prob:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p style="font-size:0.73rem;text-transform:uppercase;letter-spacing:0.1em;'
            'color:var(--text2);margin:16px 0 8px;">Hipóteses em Curso</p>',
            unsafe_allow_html=True,
        )
        bar_colors = ["var(--accent)", "var(--accent2)", "var(--text2)"]
        for i, (cond, prob, _) in enumerate(preds[:3]):
            bc = bar_colors[i]
            st.markdown(
                f"""
                <div class="diag-card" style="padding:13px 18px;margin-bottom:7px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:0.88rem;color:var(--text);">{cond}</span>
                        <span style="font-family:var(--mono);color:{bc};
                              font-weight:700;font-size:0.95rem;">{prob:.1f}%</span>
                    </div>
                    <div class="diag-bar-bg" style="margin-top:7px;">
                        <div class="diag-bar-fill"
                             style="width:{min(prob,100):.1f}%;background:{bc};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            '<p style="font-size:0.73rem;text-transform:uppercase;letter-spacing:0.1em;'
            'color:var(--text2);margin:16px 0 8px;">Sintomas Base</p>',
            unsafe_allow_html=True,
        )
        chips = " ".join(
            f'<span class="symptom-chip">◆ {s.replace("_"," ").capitalize()}</span>'
            for s in symptoms
        )
        st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)

    # ── Right: animated flashcard ─────────────────────────────────────────────
    with right:
        q         = questions[idx]
        q_label   = q.replace("_", " ").capitalize()
        fc_key    = f"fc_{rounds}_{idx}"

        # Build progress dots
        dots = "".join(
            f'<div class="fp-dot {"done" if i < idx else "current" if i == idx else ""}"></div>'
            for i in range(len(questions))
        )

        st.markdown(
            f"""
            <div class="flashcard" id="{fc_key}">
                <div class="flashcard-badge">Triagem &nbsp;·&nbsp; {idx + 1}/{len(questions)}</div>
                <div class="flashcard-round">Ronda {rounds + 1} — Pergunta de diferenciação</div>
                <div class="fp-dots">{dots}</div>
                <div class="flashcard-q">
                    Está a experienciar<br>
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
            if st.button("✓  Sim", key=f"yes_{fc_key}", use_container_width=True):
                st.session_state.feature_vector[q] = 1
                st.session_state.asked_symptoms.add(q)
                st.session_state.current_q_idx += 1
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with b2:
            st.markdown('<div class="btn-no">', unsafe_allow_html=True)
            if st.button("✗  Não", key=f"no_{fc_key}", use_container_width=True):
                st.session_state.asked_symptoms.add(q)
                st.session_state.current_q_idx += 1
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


# ── Phase 4: final results ──────────────────────────────────────────────────────
def _phase_results() -> None:
    preds    = st.session_state.predictions
    symptoms = st.session_state.extracted_symptoms
    top_prob = st.session_state.top_prob
    rounds   = st.session_state.question_rounds
    model    = st.session_state.selected_model
    m_label  = MODEL_LABELS.get(model, model)
    m_color  = MODEL_COLORS.get(model, "#00C2A8")

    conf_col = "#10B981" if top_prob >= CONFIDENCE_THRESHOLD else "#F5A623"

    st.markdown(
        f"""
        <div class="page-header">
            <p class="page-title">Diagnóstico Final</p>
            <p class="page-sub">
                Modelo: <b style="color:{m_color};">{m_label}</b>
                &nbsp;·&nbsp; Rondas de triagem: <b>{rounds}</b>
                &nbsp;·&nbsp; Confiança: <b style="color:{conf_col};">{top_prob:.1f}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        medals     = ["🥇", "🥈", "🥉"]
        bar_colors = ["var(--accent)", "var(--accent2)", "var(--accent3)"]

        for i, (cond, prob, _) in enumerate(preds):
            top  = "top" if i == 0 else ""
            bc   = bar_colors[i]
            st.markdown(
                f"""
                <div class="diag-card {top}">
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                margin-bottom:10px;">
                        <div style="display:flex;align-items:center;gap:10px;">
                            <span class="result-medal">{medals[i]}</span>
                            <span style="font-size:1.05rem;font-weight:600;">{cond}</span>
                        </div>
                        <span style="font-family:var(--mono);font-size:1.5rem;
                                     font-weight:700;color:{bc};">{prob:.1f}%</span>
                    </div>
                    <div class="diag-bar-bg">
                        <div class="diag-bar-fill"
                             style="width:{min(prob,100):.1f}%;background:{bc};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="warning-box">
                <span style="color:var(--warn);font-weight:600;font-size:0.85rem;">⚠ Aviso Importante</span>
                <p style="color:var(--text2);font-size:0.82rem;margin:7px 0 0;line-height:1.65;">
                    Este resultado é gerado por um sistema de apoio clínico e
                    <b>não substitui avaliação médica profissional</b>.
                    Consulte um médico ou ligue para o <b style="color:var(--text);">SNS 24</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        # Donut chart
        if preds:
            names = [p[0] for p in preds]
            probs = [p[1] for p in preds]
            fig   = go.Figure(
                go.Pie(
                    labels=names,
                    values=probs,
                    hole=0.60,
                    textinfo="none",
                    marker=dict(
                        colors=["#00C2A8", "#4F8EF7", "#2A3E6A"],
                        line=dict(color="#070C18", width=2),
                    ),
                )
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                height=190,
                annotations=[
                    dict(
                        text=f"<b>{probs[0]:.0f}%</b>",
                        x=0.5, y=0.5,
                        font=dict(size=24, color="#00C2A8", family="JetBrains Mono"),
                        showarrow=False,
                    )
                ],
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Symptom chips
        st.markdown(
            '<p style="font-size:0.73rem;text-transform:uppercase;letter-spacing:0.1em;'
            'color:var(--text2);margin-bottom:8px;">Sintomas detectados</p>',
            unsafe_allow_html=True,
        )
        chips = " ".join(
            f'<span class="symptom-chip">◆ {s.replace("_"," ").capitalize()}</span>'
            for s in symptoms
        )
        st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARKS PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_benchmarks() -> None:
    st.markdown(
        """
        <div class="page-header">
            <p class="page-title">📊 Benchmarks dos Modelos</p>
            <p class="page-sub">
                Comparação de desempenho — Random Forest · Gradient Boosting · Logistic Regression
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("A avaliar modelos no conjunto de teste..."):
        results = evaluate_all_models()

    if "_error" in results:
        st.error(f"Erro ao carregar dados de avaliação: {results['_error']}")
        return

    valid = {k: v for k, v in results.items() if "_error" not in v}
    errored = {k: v for k, v in results.items() if "_error" in v}

    if errored:
        for fname, res in errored.items():
            st.warning(f"{MODEL_LABELS.get(fname, fname)}: {res['_error']}")

    if not valid:
        st.error("Nenhum modelo foi avaliado com sucesso.")
        return

    metrics_keys   = ["accuracy", "f1", "recall", "precision"]
    metrics_labels = {
        "accuracy": "Accuracy", "f1": "F1-Score (Macro)",
        "recall": "Recall (Macro)", "precision": "Precision (Macro)",
    }

    # ── Per-model metric cards ────────────────────────────────────────────────
    for fname, res in valid.items():
        color = MODEL_COLORS.get(fname, "#00C2A8")
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
    st.markdown('<p class="section-title">Comparação Visual de Métricas</p>', unsafe_allow_html=True)

    fig = go.Figure()
    for fname, res in valid.items():
        fig.add_trace(
            go.Bar(
                name=res["label"],
                x=[metrics_labels[m] for m in metrics_keys],
                y=[res[m] for m in metrics_keys],
                marker_color=MODEL_COLORS.get(fname, "#ccc"),
                marker_line_width=0,
                text=[f"{res[m]:.3f}" for m in metrics_keys],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=10, color="#DDE8FF"),
            )
        )
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#6878A0"),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color="#DDE8FF", size=11),
            linecolor="#1C2E4A",
        ),
        yaxis=dict(
            range=[0, 1.12],
            showgrid=True, gridcolor="#1C2E4A",
            tickfont=dict(color="#6878A0"),
            tickformat=".0%",
        ),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=1.06,
            font=dict(color="#DDE8FF", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Gráfico Radar — Perfil de Cada Modelo</p>', unsafe_allow_html=True)

    radar_fig = go.Figure()
    categories = [metrics_labels[m] for m in metrics_keys] + [metrics_labels[metrics_keys[0]]]
    for fname, res in valid.items():
        vals = [res[m] for m in metrics_keys] + [res[metrics_keys[0]]]
        radar_fig.add_trace(
            go.Scatterpolar(
                r=vals, theta=categories,
                fill="toself", name=res["label"],
                line=dict(color=MODEL_COLORS.get(fname, "#ccc"), width=2),
                fillcolor=MODEL_COLORS.get(fname, "#ccc").replace("#", "rgba(") + ",0.08)",
            )
        )
    radar_fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickfont=dict(color="#6878A0", size=9),
                gridcolor="#1C2E4A", linecolor="#1C2E4A",
            ),
            angularaxis=dict(
                tickfont=dict(color="#DDE8FF", size=10),
                gridcolor="#1C2E4A", linecolor="#1C2E4A",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#6878A0"),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=-0.1,
            font=dict(color="#DDE8FF", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=40, r=40, t=20, b=60),
        height=360,
    )
    st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Matrizes de Confusão (Normalizadas)</p>', unsafe_allow_html=True)

    valid_cm = [(fn, r) for fn, r in valid.items() if "cm" in r]
    if valid_cm:
        cm_cols = st.columns(len(valid_cm), gap="small")
        for col, (fname, res) in zip(cm_cols, valid_cm):
            with col:
                cm      = res["cm"]
                classes = res["classes"]

                # Limit to top-30 classes by test frequency to keep matrix readable
                class_counts = cm.sum(axis=1)
                top_idx      = np.argsort(class_counts)[::-1][:30]
                cm_sub       = cm[np.ix_(top_idx, top_idx)]
                cls_sub      = [classes[i] for i in top_idx]

                # Normalise rows
                row_sums = cm_sub.sum(axis=1, keepdims=True).astype(float)
                cm_norm  = np.divide(cm_sub, row_sums, where=row_sums > 0)

                short_cls = [c[:14] + "…" if len(c) > 14 else c for c in cls_sub]

                fig = go.Figure(
                    go.Heatmap(
                        z=cm_norm, x=short_cls, y=short_cls,
                        colorscale=[[0, "#0F1929"], [0.5, "#006E62"], [1, "#00C2A8"]],
                        showscale=False,
                        hovertemplate="Real: %{y}<br>Previsto: %{x}<br>Score: %{z:.2f}<extra></extra>",
                    )
                )
                fig.update_layout(
                    title=dict(
                        text=res["label"],
                        font=dict(size=11, color="#DDE8FF", family="Plus Jakarta Sans"),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans", color="#6878A0", size=7),
                    xaxis=dict(showgrid=False, tickangle=45, tickfont=dict(size=7)),
                    yaxis=dict(showgrid=False, tickfont=dict(size=7), autorange="reversed"),
                    margin=dict(l=10, r=10, t=40, b=90),
                    height=420,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Per-class F1 heatmap ──────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">F1-Score por Classe (Top 25)</p>', unsafe_allow_html=True)

    valid_f1 = [(fn, r) for fn, r in valid.items() if "per_class_f1" in r]
    if valid_f1:
        classes_ref = valid_f1[0][1]["classes"]
        f1_mat      = np.column_stack([r["per_class_f1"] for _, r in valid_f1])
        model_lbls  = [r["label"] for _, r in valid_f1]

        # Sort by mean F1 descending, keep top 25
        mean_f1  = f1_mat.mean(axis=1)
        top_idx  = np.argsort(mean_f1)[::-1][:25]
        f1_top   = f1_mat[top_idx]
        cls_top  = [classes_ref[i] for i in top_idx]

        fig = go.Figure(
            go.Heatmap(
                z=f1_top.T, x=cls_top, y=model_lbls,
                colorscale=[[0, "#0F1929"], [0.4, "#1F4E78"], [1, "#00C2A8"]],
                showscale=True,
                zmin=0, zmax=1,
                colorbar=dict(
                    tickfont=dict(color="#6878A0", size=9),
                    outlinewidth=0,
                    len=0.8,
                ),
                hovertemplate="Classe: %{x}<br>Modelo: %{y}<br>F1: %{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", color="#6878A0"),
            xaxis=dict(
                showgrid=False, tickangle=45,
                tickfont=dict(size=9, color="#DDE8FF"),
            ),
            yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#DDE8FF")),
            margin=dict(l=20, r=60, t=20, b=130),
            height=max(220, len(valid_f1) * 70 + 180),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Summary comparison table ──────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Tabela Resumo</p>', unsafe_allow_html=True)

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
            f"**Erro de importação:** `{_IMPORT_ERR}`\n\n"
            "Verifique que a estrutura de pastas `src/` está correcta e que as "
            "dependências estão instaladas (`pip install -r requirements.txt`)."
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
