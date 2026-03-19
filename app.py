"""
FraudLens AI - Financial Intelligence & Investigation AI System
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import time
from datetime import datetime, timedelta
import pickle

# ─── Page config must be FIRST ───────────────────────────────────────────────
st.set_page_config(
    page_title="FraudLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Imports ─────────────────────────────────────────────────────────────────
from ml_model import train_model, load_model, predict_transaction, MODEL_PATH, ENCODERS_PATH
from ai_report import generate_investigation_report
from datastore import init_db, save_audit_decision, get_audit_log, get_live_transactions, get_live_transaction_count

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Crimson+Pro:ital,wght@0,400;1,400&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #0f1628;
    --bg-card: #131929;
    --bg-card-hover: #1a2235;
    --accent-blue: #2d6ef6;
    --accent-cyan: #00d4ff;
    --accent-green: #00e5a0;
    --accent-red: #ff3b6b;
    --accent-amber: #ffb800;
    --text-primary: #e8edf5;
    --text-secondary: #8899bb;
    --text-muted: #4a5678;
    --border: #1e2d4a;
    --border-bright: #2d4070;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1220 50%, #0a1015 100%);
}

/* ── Header ── */
.fraudlens-header {
    background: linear-gradient(135deg, #0f1628 0%, #131929 100%);
    border: 1px solid var(--border-bright);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.fraudlens-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan), var(--accent-green));
}
.header-logo {
    display: flex;
    align-items: center;
    gap: 16px;
}
.header-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
}
.header-title {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, #fff 30%, var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin-bottom: 4px;
}
.header-tagline {
    font-size: 12px;
    color: var(--text-secondary);
    letter-spacing: 2px;
    text-transform: uppercase;
}
.header-status {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,229,160,0.1);
    border: 1px solid rgba(0,229,160,0.3);
    border-radius: 20px;
    padding: 8px 16px;
    font-size: 13px;
    color: var(--accent-green);
    font-weight: 600;
}
.status-dot {
    width: 8px; height: 8px;
    background: var(--accent-green);
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.3); }
}

/* ── Metric Cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.blue::after { background: var(--accent-blue); }
.metric-card.cyan::after { background: var(--accent-cyan); }
.metric-card.red::after { background: var(--accent-red); }
.metric-card.green::after { background: var(--accent-green); }
.metric-card.amber::after { background: var(--accent-amber); }
.metric-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
.metric-value { font-size: 32px; font-weight: 700; line-height: 1; }
.metric-value.blue { color: var(--accent-blue); }
.metric-value.cyan { color: var(--accent-cyan); }
.metric-value.red { color: var(--accent-red); }
.metric-value.green { color: var(--accent-green); }
.metric-value.amber { color: var(--accent-amber); }
.metric-sub { font-size: 12px; color: var(--text-secondary); margin-top: 6px; }

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
.section-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: 0.3px;
}
.section-badge {
    background: rgba(45,110,246,0.15);
    border: 1px solid rgba(45,110,246,0.3);
    color: var(--accent-blue);
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 10px;
    letter-spacing: 0.5px;
}

/* ── Transaction rows ── */
.txn-row {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.txn-row:hover { border-color: var(--accent-blue); background: var(--bg-card-hover); }
.txn-id { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--accent-cyan); font-weight: 600; }
.txn-amount { font-size: 15px; font-weight: 700; color: var(--text-primary); }
.txn-meta { font-size: 11px; color: var(--text-secondary); }

/* ── Risk badges ── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.risk-CRITICAL { background: rgba(255,59,107,0.15); border: 1px solid rgba(255,59,107,0.4); color: #ff3b6b; }
.risk-HIGH { background: rgba(255,184,0,0.12); border: 1px solid rgba(255,184,0,0.35); color: #ffb800; }
.risk-MEDIUM { background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.3); color: #00d4ff; }
.risk-LOW { background: rgba(0,229,160,0.1); border: 1px solid rgba(0,229,160,0.3); color: #00e5a0; }

/* ── Investigation Panel ── */
.panel-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
}
.panel-card-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}
.info-row {
    display: flex;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px solid rgba(30,45,74,0.5);
    font-size: 13px;
}
.info-label { color: var(--text-muted); }
.info-value { color: var(--text-primary); font-family: 'JetBrains Mono', monospace; font-size: 12px; }
.info-value.highlight { color: var(--accent-cyan); }
.info-value.danger { color: var(--accent-red); font-weight: 700; }
.info-value.safe { color: var(--accent-green); }

/* ── Fraud meter ── */
.fraud-meter-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    margin-bottom: 16px;
}
.fraud-score-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px; }
.fraud-score-value {
    font-size: 56px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin-bottom: 8px;
}

/* ── Report ── */
.report-section {
    background: rgba(15, 22, 40, 0.8);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.report-section-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    margin-bottom: 10px;
}
.report-text {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.7;
    font-family: 'Crimson Pro', serif;
    font-size: 15px;
}
.flag-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 12px;
    background: rgba(255,59,107,0.05);
    border: 1px solid rgba(255,59,107,0.15);
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 13px;
    color: #ff8099;
}

/* ── Alert banner ── */
.alert-banner {
    background: rgba(255,59,107,0.08);
    border: 1px solid rgba(255,59,107,0.3);
    border-radius: 12px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    animation: alertPulse 2s infinite;
}
@keyframes alertPulse {
    0%, 100% { border-color: rgba(255,59,107,0.3); }
    50% { border-color: rgba(255,59,107,0.7); }
}
.alert-icon { font-size: 20px; }
.alert-text { font-size: 13px; color: #ff8099; font-weight: 600; }

/* ── Buttons ── */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
}

/* ── Audit table ── */
.audit-row {
    display: grid;
    grid-template-columns: 1.2fr 0.8fr 0.8fr 0.8fr 1fr;
    gap: 12px;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
    align-items: center;
}
.audit-row:hover { background: var(--bg-card-hover); }
.decision-CONFIRMED { color: var(--accent-red); font-weight: 700; }
.decision-CLEARED { color: var(--accent-green); font-weight: 700; }
.decision-ESCALATED { color: var(--accent-amber); font-weight: 700; }

/* ── Live feed ── */
.live-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,59,107,0.1);
    border: 1px solid rgba(255,59,107,0.3);
    border-radius: 16px;
    padding: 4px 12px;
    font-size: 11px;
    color: var(--accent-red);
    font-weight: 700;
    letter-spacing: 1px;
}
.live-dot {
    width: 6px; height: 6px;
    background: var(--accent-red);
    border-radius: 50%;
    animation: pulse 1s infinite;
}

/* ── Outcome badge ── */
.outcome-FRAUD_CONFIRMED { 
    background: rgba(255,59,107,0.15); 
    border: 1px solid var(--accent-red); 
    color: var(--accent-red);
    padding: 10px 20px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 16px;
    text-align: center;
}
.outcome-SUSPICIOUS { 
    background: rgba(255,184,0,0.12); 
    border: 1px solid var(--accent-amber); 
    color: var(--accent-amber);
    padding: 10px 20px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 16px;
    text-align: center;
}
.outcome-CLEARED { 
    background: rgba(0,229,160,0.1); 
    border: 1px solid var(--accent-green); 
    color: var(--accent-green);
    padding: 10px 20px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    customers = pd.read_csv("upi_customers.csv")
    merchants = pd.read_csv("upi_merchants.csv")
    transactions = pd.read_csv("upi_transactions.csv")
    return customers, merchants, transactions


@st.cache_resource
def get_model(customers, merchants, transactions):
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
        try:
            model, encoders = load_model()
            return model, encoders, None
        except:
            pass
    model, encoders, acc, _ = train_model(customers, merchants, transactions)
    return model, encoders, acc


def score_all_transactions(transactions, customers, merchants, model, encoders):
    """Pre-score all transactions for the dashboard."""
    results = []
    for _, row in transactions.iterrows():
        txn = row.to_dict()
        prob, meta = predict_transaction(txn, model, encoders, customers, merchants, transactions)
        if prob is not None:
            meta['transaction_uuid'] = txn['transaction_uuid']
            meta['transaction_amount'] = txn['transaction_amount']
            meta['transaction_location'] = txn['transaction_location']
            meta['transaction_timestamp'] = txn['transaction_timestamp']
            meta['customer_uuid'] = txn['customer_uuid']
            meta['merchant_uuid'] = txn['merchant_uuid']
            results.append(meta)
    return pd.DataFrame(results)


# ─── Init ─────────────────────────────────────────────────────────────────────
init_db()
customers_df, merchants_df, transactions_df = load_data()

with st.spinner("🤖 Initializing FraudLens AI engine..."):
    model, encoders, model_acc = get_model(customers_df, merchants_df, transactions_df)

# Session state
if 'scored_df' not in st.session_state:
    with st.spinner("🔍 Scoring all transactions..."):
        st.session_state.scored_df = score_all_transactions(
            transactions_df, customers_df, merchants_df, model, encoders
        )
if 'selected_txn' not in st.session_state:
    st.session_state.selected_txn = None
if 'investigation_report' not in st.session_state:
    st.session_state.investigation_report = {}
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'api_provider' not in st.session_state:
    st.session_state.api_provider = "Anthropic"

scored_df = st.session_state.scored_df

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 8px 0;'>
        <div style='font-size: 18px; font-weight: 700; color: #e8edf5; margin-bottom: 4px;'>⚙️ Configuration</div>
        <div style='font-size: 11px; color: #4a5678; letter-spacing: 1px; text-transform: uppercase;'>FraudLens AI v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.selectbox(
        "Navigation",
        ["🏠 Command Center", "🔍 Investigation Panel", "📊 Analytics Hub", "📋 Audit Trail", "📡 Live Feed"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("<div style='font-size:12px; color:#4a5678; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>AI Model Provider</div>", unsafe_allow_html=True)
    provider_options = ["Anthropic", "OpenAI", "Hugging Face"]
    st.session_state.api_provider = st.selectbox(
        "Provider", provider_options, 
        index=provider_options.index(st.session_state.api_provider),
        label_visibility="collapsed"
    )

    placeholder_map = {
        "Anthropic": "sk-ant-...",
        "OpenAI": "sk-proj-...",
        "Hugging Face": "hf_..."
    }

    st.markdown(f"<div style='font-size:12px; color:#4a5678; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; margin-top:12px;'>{st.session_state.api_provider} API Key</div>", unsafe_allow_html=True)
    api_key_input = st.text_input("API Key", type="password", value=st.session_state.api_key,
                                   placeholder=placeholder_map.get(st.session_state.api_provider, "sk-..."), label_visibility="collapsed")
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.success(f"✓ {st.session_state.api_provider} key set", icon="🔑")

    st.divider()

    # Filters
    st.markdown("<div style='font-size:12px; color:#4a5678; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>Risk Filter</div>", unsafe_allow_html=True)
    risk_filter = st.multiselect("Risk", ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                                  default=["CRITICAL", "HIGH"], label_visibility="collapsed")

    st.markdown("<div style='font-size:12px; color:#4a5678; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; margin-top:12px;'>Fraud Probability Threshold</div>", unsafe_allow_html=True)
    threshold = st.slider("Threshold", 0, 100, 50, label_visibility="collapsed")

    st.divider()

    if model_acc:
        st.markdown(f"""
        <div style='background: rgba(0,229,160,0.08); border: 1px solid rgba(0,229,160,0.2); border-radius: 10px; padding: 14px;'>
            <div style='font-size: 11px; color: #4a5678; text-transform: uppercase; letter-spacing: 1px;'>Model Performance</div>
            <div style='font-size: 24px; font-weight: 700; color: #00e5a0; margin: 6px 0;'>{model_acc*100:.1f}%</div>
            <div style='font-size: 11px; color: #8899bb;'>XGBoost Accuracy</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────
flagged = scored_df[scored_df['fraud_probability'] >= threshold]
total_flagged = len(flagged)
critical_count = len(flagged[flagged.get('risk_level', pd.Series()) == 'CRITICAL']) if 'risk_level' in flagged.columns else 0

st.markdown(f"""
<div class="fraudlens-header">
    <div class="header-logo">
        <div class="header-icon">🔍</div>
        <div>
            <div class="header-title">FraudLens AI</div>
            <div class="header-tagline">Financial Intelligence & Investigation AI System</div>
        </div>
    </div>
    <div style="display:flex; gap:16px; align-items:center;">
        <div style="text-align:right;">
            <div style="font-size:12px; color:#4a5678; text-transform:uppercase; letter-spacing:1px;">Active Alerts</div>
            <div style="font-size:20px; font-weight:700; color:#ff3b6b;">{total_flagged}</div>
        </div>
        <div class="header-status">
            <div class="status-dot"></div>
            SYSTEM ONLINE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: COMMAND CENTER
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Command Center":

    # ── KPI Row ──
    total = len(transactions_df)
    fraud_rate = (scored_df['fraud_probability'] >= 50).mean() * 100
    high_risk = len(scored_df[scored_df['fraud_probability'] >= 70])
    avg_amount = transactions_df['transaction_amount'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "Total Transactions", f"{total:,}", "Loaded from dataset", "blue"),
        (c2, "Flagged Cases", f"{total_flagged}", f"≥ {threshold}% fraud prob", "red"),
        (c3, "High Risk (≥70%)", f"{high_risk}", "Immediate review needed", "amber"),
        (c4, "Fraud Rate", f"{fraud_rate:.1f}%", "Model estimated", "cyan"),
        (c5, "Avg Amount", f"₹{avg_amount/1000:.1f}K", "Across all transactions", "green"),
    ]
    for col, label, value, sub, color in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card {color}">
                <div class="metric-label">{label}</div>
                <div class="metric-value {color}">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row ──
    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown("""
        <div class="section-header">
            <span class="section-title">📈 Fraud Probability Distribution</span>
            <span class="section-badge">ML MODEL</span>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=scored_df['fraud_probability'],
            nbinsx=40,
            marker=dict(
                color=scored_df['fraud_probability'].apply(
                    lambda x: '#ff3b6b' if x >= 70 else ('#ffb800' if x >= 50 else '#2d6ef6')
                ),
                line=dict(width=0)
            ),
            opacity=0.85,
            name="Transactions"
        ))
        fig.add_vline(x=threshold, line_dash="dash", line_color="#00d4ff",
                      annotation_text=f"Threshold: {threshold}%",
                      annotation_font_color="#00d4ff")
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#8899bb', font_family='Space Grotesk',
            xaxis=dict(title="Fraud Probability (%)", gridcolor='#1e2d4a', color='#8899bb'),
            yaxis=dict(title="Transaction Count", gridcolor='#1e2d4a', color='#8899bb'),
            showlegend=False, height=280, margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="section-header">
            <span class="section-title">🗺️ City-wise Flagged Transactions</span>
            <span class="section-badge">GEO</span>
        </div>
        """, unsafe_allow_html=True)

        flagged_city = flagged.groupby('transaction_location').size().reset_index(name='count').sort_values('count', ascending=True).tail(8)
        fig2 = go.Figure(go.Bar(
            x=flagged_city['count'],
            y=flagged_city['transaction_location'],
            orientation='h',
            marker=dict(
                color=flagged_city['count'],
                colorscale=[[0, '#1a2d5a'], [0.5, '#2d6ef6'], [1, '#ff3b6b']],
                showscale=False
            )
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#8899bb', font_family='Space Grotesk',
            xaxis=dict(gridcolor='#1e2d4a', color='#8899bb', title='Count'),
            yaxis=dict(gridcolor='#1e2d4a', color='#8899bb'),
            height=280, margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Second charts row ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div class="section-header">
            <span class="section-title">⚠️ Anomaly Flags Frequency</span>
            <span class="section-badge">RISK SIGNALS</span>
        </div>
        """, unsafe_allow_html=True)

        flag_counts = {
            'Unregistered Device': int(scored_df['new_device'].sum()) if 'new_device' in scored_df.columns else 0,
            'New IP Address': int(scored_df['new_ip'].sum()) if 'new_ip' in scored_df.columns else 0,
            'Different City': int(scored_df['is_different_city'].sum()) if 'is_different_city' in scored_df.columns else 0,
            'High Amount Ratio': int((scored_df['txn_amount_ratio'] > 1.5).sum()) if 'txn_amount_ratio' in scored_df.columns else 0,
        }
        colors = ['#ff3b6b', '#ffb800', '#00d4ff', '#2d6ef6']
        fig3 = go.Figure(go.Bar(
            x=list(flag_counts.keys()),
            y=list(flag_counts.values()),
            marker_color=colors,
            text=list(flag_counts.values()),
            textposition='outside',
            textfont=dict(color='#8899bb', size=12)
        ))
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#8899bb', font_family='Space Grotesk',
            xaxis=dict(gridcolor='#1e2d4a', color='#8899bb'),
            yaxis=dict(gridcolor='#1e2d4a', color='#8899bb', title='Count'),
            height=260, margin=dict(l=0, r=0, t=30, b=0), showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("""
        <div class="section-header">
            <span class="section-title">🎯 Risk Level Distribution</span>
            <span class="section-badge">OVERVIEW</span>
        </div>
        """, unsafe_allow_html=True)

        def get_risk(prob):
            if prob >= 80: return 'CRITICAL'
            elif prob >= 60: return 'HIGH'
            elif prob >= 40: return 'MEDIUM'
            return 'LOW'

        scored_df['risk_level'] = scored_df['fraud_probability'].apply(get_risk)
        risk_counts = scored_df['risk_level'].value_counts()

        fig4 = go.Figure(go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.65,
            marker=dict(colors=['#ff3b6b', '#ffb800', '#00d4ff', '#00e5a0'],
                        line=dict(color='#0a0e1a', width=2)),
            textfont=dict(family='Space Grotesk', size=12),
        ))
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#8899bb', font_family='Space Grotesk',
            legend=dict(bgcolor='rgba(0,0,0,0)', font_color='#8899bb'),
            height=260, margin=dict(l=0, r=0, t=10, b=0),
            annotations=[dict(text=f'{len(scored_df)}', x=0.5, y=0.5,
                              font_size=22, font_color='#e8edf5',
                              showarrow=False, font_family='JetBrains Mono')]
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Flagged transactions table ──
    st.markdown("""
    <div class="section-header" style="margin-top: 12px;">
        <span class="section-title">🚨 High-Priority Flagged Cases</span>
        <span class="section-badge">REQUIRES ACTION</span>
    </div>
    """, unsafe_allow_html=True)

    top_flagged = scored_df[scored_df['fraud_probability'] >= threshold].sort_values(
        'fraud_probability', ascending=False
    ).head(20)

    if top_flagged.empty:
        st.info(f"No transactions above {threshold}% fraud probability threshold.")
    else:
        for _, row in top_flagged.iterrows():
            risk = row.get('risk_level', 'MEDIUM')
            prob = row['fraud_probability']
            flags_count = len(row.get('flags', []))

            col_a, col_b = st.columns([5, 1])
            with col_a:
                st.markdown(f"""
                <div class="txn-row">
                    <div>
                        <div class="txn-id">{row['transaction_uuid']}</div>
                        <div class="txn-meta">👤 {row['customer_uuid']} &nbsp;|&nbsp; 🏪 {row.get('merchant_uuid','—')} &nbsp;|&nbsp; 📍 {row.get('transaction_location','—')} &nbsp;|&nbsp; 🕐 {str(row.get('transaction_timestamp',''))[:16]}</div>
                    </div>
                    <div style="display:flex; gap:16px; align-items:center;">
                        <div style="text-align:right;">
                            <div class="txn-amount">₹{float(row['transaction_amount']):,.2f}</div>
                            <div class="txn-meta">{flags_count} flag(s)</div>
                        </div>
                        <div>
                            <div class="risk-badge risk-{risk}">{risk}</div>
                            <div style="font-size:13px; font-family:'JetBrains Mono'; color:{'#ff3b6b' if prob>70 else '#ffb800'}; margin-top:4px; text-align:center;">{prob:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                if st.button("Investigate →", key=f"inv_{row['transaction_uuid']}"):
                    st.session_state.selected_txn = row['transaction_uuid']
                    st.session_state.page_redirect = "🔍 Investigation Panel"
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: INVESTIGATION PANEL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Investigation Panel":

    st.markdown("""
    <div class="section-header">
        <span class="section-title">🔍 Investigation Panel</span>
        <span class="section-badge">OFFICER WORKBENCH</span>
    </div>
    """, unsafe_allow_html=True)

    # Transaction selector
    all_txn_ids = scored_df.sort_values('fraud_probability', ascending=False)['transaction_uuid'].tolist()
    selected = st.selectbox(
        "Select Transaction to Investigate",
        all_txn_ids,
        index=all_txn_ids.index(st.session_state.selected_txn) if st.session_state.selected_txn in all_txn_ids else 0
    )
    st.session_state.selected_txn = selected

    if not selected:
        st.info("Select a transaction to begin investigation.")
        st.stop()

    # Load data
    txn_row = transactions_df[transactions_df['transaction_uuid'] == selected]
    if txn_row.empty:
        st.error("Transaction not found.")
        st.stop()

    txn = txn_row.iloc[0].to_dict()
    cust = customers_df[customers_df['customer_uuid'] == txn['customer_uuid']]
    merch = merchants_df[merchants_df['merchant_uuid'] == txn['merchant_uuid']]
    scored_row = scored_df[scored_df['transaction_uuid'] == selected]

    if cust.empty:
        st.error("Customer record not found.")
        st.stop()

    cust = cust.iloc[0].to_dict()
    merch = merch.iloc[0].to_dict() if not merch.empty else {}
    ml_result = scored_row.iloc[0].to_dict() if not scored_row.empty else {}

    fraud_prob = ml_result.get('fraud_probability', 0)
    flags = ml_result.get('flags', [])
    risk_level = ml_result.get('risk_level', 'LOW')

    # Alert banner for high-risk
    if fraud_prob >= 70:
        st.markdown(f"""
        <div class="alert-banner">
            <div class="alert-icon">🚨</div>
            <div class="alert-text">HIGH FRAUD PROBABILITY DETECTED — Immediate review required for {selected}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Left: Data Panels | Right: ML Results ──
    left, right = st.columns([1.4, 1])

    with left:
        # Transaction details
        st.markdown("""<div class="panel-card">
        <div class="panel-card-title">💳 Transaction Details</div>""", unsafe_allow_html=True)
        details = [
            ("Transaction ID", txn.get('transaction_uuid'), 'highlight'),
            ("Amount", f"₹{float(txn.get('transaction_amount',0)):,.2f}", 'highlight'),
            ("Timestamp", str(txn.get('transaction_timestamp',''))[:19], ''),
            ("Location", txn.get('transaction_location',''), ''),
            ("Device Used", txn.get('customer_device_id',''), 'danger' if ml_result.get('new_device') else ''),
            ("IP Address", txn.get('customer_ip_address',''), 'danger' if ml_result.get('new_ip') else ''),
        ]
        for label, value, cls in details:
            st.markdown(f"""
            <div class="info-row">
                <span class="info-label">{label}</span>
                <span class="info-value {cls}">{value}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Customer profile
        st.markdown("""<div class="panel-card">
        <div class="panel-card-title">👤 Customer Profile</div>""", unsafe_allow_html=True)
        cust_details = [
            ("Name", cust.get('full_name',''), ''),
            ("Customer ID", cust.get('customer_uuid',''), 'highlight'),
            ("Age", str(cust.get('age','')), ''),
            ("Bank", cust.get('bank_name',''), ''),
            ("Home Branch", cust.get('home_branch',''), ''),
            ("Account Balance", f"₹{float(cust.get('account_balance',0)):,.2f}", 'safe'),
            ("Registered Device", cust.get('registered_device_id',''), 'danger' if ml_result.get('new_device') else 'safe'),
            ("Registered IP", cust.get('registered_ip_address',''), 'danger' if ml_result.get('new_ip') else 'safe'),
            ("Total Transactions", str(cust.get('total_transactions_count','')), ''),
            ("Last Txn Amount", f"₹{float(cust.get('last_transaction_amount',0)):,.2f}", ''),
        ]
        for label, value, cls in cust_details:
            st.markdown(f"""
            <div class="info-row">
                <span class="info-label">{label}</span>
                <span class="info-value {cls}">{value}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Merchant info
        if merch:
            st.markdown("""<div class="panel-card">
            <div class="panel-card-title">🏪 Merchant Information</div>""", unsafe_allow_html=True)
            merch_details = [
                ("Merchant Name", merch.get('merchant_name',''), ''),
                ("Merchant ID", merch.get('merchant_uuid',''), 'highlight'),
                ("UPI ID", merch.get('merchant_upi_id',''), ''),
                ("Bank", merch.get('merchant_bank_name',''), ''),
                ("Branch", merch.get('merchant_bank_branch',''), ''),
                ("Account Since", str(merch.get('merchant_account_open_date',''))[:10], ''),
            ]
            for label, value, cls in merch_details:
                st.markdown(f"""
                <div class="info-row">
                    <span class="info-label">{label}</span>
                    <span class="info-value {cls}">{value}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Transaction history chart
        cust_history = transactions_df[transactions_df['customer_uuid'] == txn['customer_uuid']].copy()
        if len(cust_history) > 1:
            cust_history['transaction_timestamp'] = pd.to_datetime(cust_history['transaction_timestamp'])
            cust_history = cust_history.sort_values('transaction_timestamp').tail(20)

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=cust_history['transaction_timestamp'],
                y=cust_history['transaction_amount'],
                mode='lines+markers',
                line=dict(color='#2d6ef6', width=2),
                marker=dict(size=6, color='#2d6ef6'),
                fill='tozeroy',
                fillcolor='rgba(45,110,246,0.08)',
                name='Amount'
            ))
            # Mark current transaction
            fig_hist.add_trace(go.Scatter(
                x=[pd.to_datetime(txn['transaction_timestamp'])],
                y=[float(txn['transaction_amount'])],
                mode='markers',
                marker=dict(size=12, color='#ff3b6b', symbol='star'),
                name='This Transaction'
            ))
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#8899bb', font_family='Space Grotesk',
                xaxis=dict(gridcolor='#1e2d4a', color='#8899bb'),
                yaxis=dict(gridcolor='#1e2d4a', color='#8899bb', title='Amount (₹)'),
                height=220, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(bgcolor='rgba(0,0,0,0)', font_color='#8899bb'),
                title=dict(text="Customer Transaction History (Last 20)", font_color='#8899bb', font_size=13)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    with right:
        # ── Fraud Score Meter ──
        color_map = {'CRITICAL': '#ff3b6b', 'HIGH': '#ffb800', 'MEDIUM': '#00d4ff', 'LOW': '#00e5a0'}
        score_color = color_map.get(risk_level, '#2d6ef6')

        st.markdown(f"""
        <div class="fraud-meter-wrap">
            <div class="fraud-score-label">Fraud Probability Score</div>
            <div class="fraud-score-value" style="color:{score_color};">{fraud_prob:.1f}%</div>
            <div class="risk-badge risk-{risk_level}" style="margin: 8px auto; display:inline-flex;">{risk_level} RISK</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob,
            number=dict(suffix="%", font=dict(color=score_color, size=28, family='JetBrains Mono')),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor='#4a5678', tickfont=dict(color='#4a5678')),
                bar=dict(color=score_color, thickness=0.3),
                bgcolor='rgba(0,0,0,0)',
                bordercolor='#1e2d4a',
                steps=[
                    dict(range=[0, 40], color='rgba(0,229,160,0.1)'),
                    dict(range=[40, 60], color='rgba(0,212,255,0.1)'),
                    dict(range=[60, 80], color='rgba(255,184,0,0.1)'),
                    dict(range=[80, 100], color='rgba(255,59,107,0.1)'),
                ],
                threshold=dict(line=dict(color="#fff", width=2), thickness=0.75, value=threshold)
            )
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', font_color='#8899bb',
            height=200, margin=dict(l=20, r=20, t=20, b=0)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Anomaly Flags
        st.markdown("""<div class="panel-card">
        <div class="panel-card-title">⚠️ Anomaly Flags</div>""", unsafe_allow_html=True)
        if flags:
            for flag in flags:
                st.markdown(f'<div class="flag-item">🔴 {flag}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#00e5a0; font-size:13px;">✅ No anomaly flags detected</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Amount analysis
        avg_amount = ml_result.get('avg_amount', 0)
        ratio = ml_result.get('txn_amount_ratio', 1)
        st.markdown(f"""<div class="panel-card">
        <div class="panel-card-title">💰 Amount Analysis</div>
        <div class="info-row">
            <span class="info-label">Current Transaction</span>
            <span class="info-value highlight">₹{float(txn.get('transaction_amount',0)):,.2f}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Customer Average</span>
            <span class="info-value">₹{avg_amount:,.2f}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Amount Ratio</span>
            <span class="info-value {'danger' if ratio > 1.5 else 'safe'}">{ratio:.2f}x</span>
        </div>
        </div>""", unsafe_allow_html=True)

        # ── Generate AI Report ──
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🤖 Generate AI Investigation Report", use_container_width=True, type="primary"):
            with st.spinner("🧠 AI analyzing transaction data..."):
                report = generate_investigation_report(
                    txn, cust, merch, ml_result,
                    st.session_state.api_key,
                    st.session_state.api_provider
                )
                st.session_state.investigation_report[selected] = report

    # ── AI Report Section ──
    if selected in st.session_state.investigation_report:
        report = st.session_state.investigation_report[selected]
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
            <span class="section-title">🤖 AI Investigation Report</span>
            <span class="section-badge">GENAI POWERED</span>
        </div>
        """, unsafe_allow_html=True)

        outcome = report.get('investigation_outcome', 'CLEARED')
        confidence = report.get('confidence_score', fraud_prob)

        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            st.markdown(f'<div class="outcome-{outcome}">📋 {outcome.replace("_"," ")}</div>', unsafe_allow_html=True)
        with oc2:
            risk_r = report.get('risk_level', risk_level)
            st.markdown(f'<div style="background:rgba(0,0,0,0.3); border: 1px solid #1e2d4a; border-radius:10px; padding:10px 20px; text-align:center;"><span class="risk-badge risk-{risk_r}">{risk_r} RISK</span></div>', unsafe_allow_html=True)
        with oc3:
            st.markdown(f'<div style="background:rgba(0,0,0,0.3); border: 1px solid #1e2d4a; border-radius:10px; padding:10px 20px; text-align:center; font-family:JetBrains Mono; color:#00d4ff;">Confidence: {confidence:.0f}%</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Executive Summary
        st.markdown(f"""
        <div class="report-section">
            <div class="report-section-title">📄 Executive Summary</div>
            <div class="report-text">{report.get('executive_summary','')}</div>
        </div>
        """, unsafe_allow_html=True)

        rep_col1, rep_col2 = st.columns(2)

        with rep_col1:
            # Detected Inconsistencies
            inconsistencies = report.get('detected_inconsistencies', [])
            incon_html = ''.join([f'<div class="flag-item">⚡ {i}</div>' for i in inconsistencies]) if inconsistencies else '<div style="color:#00e5a0;">No critical inconsistencies detected.</div>'
            st.markdown(f"""
            <div class="report-section">
                <div class="report-section-title">🔎 Detected Inconsistencies</div>
                {incon_html}
            </div>
            """, unsafe_allow_html=True)

            # Supporting Evidence
            evidence = report.get('supporting_evidence', [])
            ev_html = ''.join([f'<div style="padding:6px 0; border-bottom:1px solid #1e2d4a; font-size:13px; color:#8899bb;">✓ {e}</div>' for e in evidence])
            st.markdown(f"""
            <div class="report-section">
                <div class="report-section-title">📌 Supporting Evidence</div>
                {ev_html}
            </div>
            """, unsafe_allow_html=True)

        with rep_col2:
            # Data Analyzed
            data_analyzed = report.get('data_analyzed', {})
            st.markdown(f"""
            <div class="report-section">
                <div class="report-section-title">📊 Data Analyzed</div>
                <div style="font-size:12px; color:#4a5678; margin-bottom:6px; text-transform:uppercase;">Customer Risk Factors</div>
                {''.join([f'<div style="font-size:13px; color:#8899bb; padding:3px 0;">• {f}</div>' for f in data_analyzed.get('customer_risk_factors',[])])}
                <div style="font-size:12px; color:#4a5678; margin: 10px 0 6px; text-transform:uppercase;">Transaction Anomalies</div>
                {''.join([f'<div style="font-size:13px; color:#8899bb; padding:3px 0;">• {f}</div>' for f in data_analyzed.get('transaction_anomalies',[])])}
                <div style="font-size:12px; color:#4a5678; margin: 10px 0 6px; text-transform:uppercase;">Merchant Risk</div>
                {''.join([f'<div style="font-size:13px; color:#8899bb; padding:3px 0;">• {f}</div>' for f in data_analyzed.get('merchant_risk_factors',[])])}
            </div>
            """, unsafe_allow_html=True)

            # Mitigating factors
            mit = report.get('mitigating_factors', [])
            if mit:
                mit_html = ''.join([f'<div style="padding:6px 0; border-bottom:1px solid #1e2d4a; font-size:13px; color:#00e5a0;">✓ {m}</div>' for m in mit])
                st.markdown(f"""
                <div class="report-section">
                    <div class="report-section-title">🟢 Mitigating Factors</div>
                    {mit_html}
                </div>
                """, unsafe_allow_html=True)

        # Reasoning
        st.markdown(f"""
        <div class="report-section">
            <div class="report-section-title">🧠 AI Reasoning</div>
            <div class="report-text">{report.get('reasoning','')}</div>
        </div>
        """, unsafe_allow_html=True)

        # Recommended Action
        st.markdown(f"""
        <div style="background: rgba(45,110,246,0.08); border: 1px solid rgba(45,110,246,0.3); border-radius: 10px; padding: 16px 20px; margin-bottom: 16px;">
            <div style="font-size:11px; color:#4a5678; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Recommended Action</div>
            <div style="font-size:15px; color:#2d6ef6; font-weight:600;">→ {report.get('recommended_action','')}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Officer Decision ──
        st.markdown("""
        <div class="section-header" style="margin-top:20px;">
            <span class="section-title">⚖️ Officer Decision</span>
            <span class="section-badge">FINALIZE CASE</span>
        </div>
        """, unsafe_allow_html=True)

        notes = st.text_area("Investigation Notes (optional)", placeholder="Add case notes, observations, or justification...", height=80)

        d1, d2, d3 = st.columns(3)
        with d1:
            if st.button("🚫 CONFIRM FRAUD", use_container_width=True,
                          help="Mark transaction as confirmed fraud and block"):
                save_audit_decision(selected, "CONFIRMED", outcome, risk_level, fraud_prob, notes)
                st.success(f"✅ Case {selected} marked as FRAUD CONFIRMED and logged to audit trail.")

        with d2:
            if st.button("✅ CLEAR TRANSACTION", use_container_width=True,
                          help="Mark as legitimate and clear"):
                save_audit_decision(selected, "CLEARED", outcome, risk_level, fraud_prob, notes)
                st.success(f"✅ Transaction {selected} CLEARED and logged to audit trail.")

        with d3:
            if st.button("📤 ESCALATE TO SENIOR", use_container_width=True,
                          help="Escalate to senior officer for review"):
                save_audit_decision(selected, "ESCALATED", outcome, risk_level, fraud_prob, notes)
                st.warning(f"⚠️ Case {selected} ESCALATED to senior officer.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS HUB
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics Hub":

    st.markdown("""
    <div class="section-header">
        <span class="section-title">📊 Analytics Hub</span>
        <span class="section-badge">DEEP INSIGHTS</span>
    </div>
    """, unsafe_allow_html=True)

    merged = transactions_df.merge(customers_df, on='customer_uuid', how='left')
    merged = merged.merge(merchants_df, on='merchant_uuid', how='left')
    if 'risk_level' not in scored_df.columns:
        scored_df['risk_level'] = scored_df['fraud_probability'].apply(
            lambda x: 'CRITICAL' if x >= 80 else ('HIGH' if x >= 60 else ('MEDIUM' if x >= 40 else 'LOW'))
        )

    c1, c2 = st.columns(2)

    with c1:
        # Amount by location
        st.markdown("**Transaction Amount by City**")
        loc_data = merged.groupby('transaction_location')['transaction_amount'].agg(['mean','count']).reset_index()
        loc_data.columns = ['City', 'Avg Amount', 'Count']
        loc_data = loc_data.sort_values('Avg Amount', ascending=False).head(12)
        fig = px.bar(loc_data, x='City', y='Avg Amount', color='Count',
                     color_continuous_scale=[[0,'#1a2d5a'],[1,'#2d6ef6']])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#8899bb', height=300, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Bank distribution
        st.markdown("**Customers by Bank**")
        bank_data = customers_df['bank_name'].value_counts().head(8)
        fig2 = px.pie(values=bank_data.values, names=bank_data.index, hole=0.5,
                      color_discrete_sequence=['#2d6ef6','#00d4ff','#00e5a0','#ffb800','#ff3b6b','#8b5cf6','#ec4899','#06b6d4'])
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#8899bb', height=300, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Fraud prob by age group
        st.markdown("**Fraud Risk by Customer Age Group**")
        merged_scored = merged.merge(scored_df[['transaction_uuid','fraud_probability']], on='transaction_uuid', how='left')
        merged_scored['age_group'] = pd.cut(merged_scored['age'], bins=[0,25,35,45,55,100],
                                             labels=['<25','25-35','35-45','45-55','55+'])
        age_fraud = merged_scored.groupby('age_group')['fraud_probability'].mean().reset_index()
        fig3 = px.bar(age_fraud, x='age_group', y='fraud_probability',
                      color='fraud_probability', color_continuous_scale=[[0,'#00e5a0'],[0.5,'#ffb800'],[1,'#ff3b6b']])
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#8899bb', height=280, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # Top fraudulent merchants
        st.markdown("**Top Merchants by Fraud Exposure**")
        merch_fraud = scored_df.merge(merchants_df[['merchant_uuid','merchant_name']], on='merchant_uuid', how='left')
        top_merch = merch_fraud.groupby('merchant_name')['fraud_probability'].mean().reset_index()
        top_merch = top_merch.sort_values('fraud_probability', ascending=False).head(10)
        fig4 = go.Figure(go.Bar(
            x=top_merch['fraud_probability'],
            y=top_merch['merchant_name'],
            orientation='h',
            marker=dict(color=top_merch['fraud_probability'],
                        colorscale=[[0,'#1a2d5a'],[0.5,'#ffb800'],[1,'#ff3b6b']])
        ))
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#8899bb', height=280, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig4, use_container_width=True)

    # Scatter: amount vs fraud probability
    st.markdown("**Amount vs. Fraud Probability Scatter**")
    fig5 = px.scatter(
        scored_df, x='transaction_amount', y='fraud_probability',
        color='risk_level',
        color_discrete_map={'CRITICAL':'#ff3b6b','HIGH':'#ffb800','MEDIUM':'#00d4ff','LOW':'#00e5a0'},
        opacity=0.7, size_max=8
    )
    fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       font_color='#8899bb', height=350, margin=dict(l=0,r=0,t=10,b=0),
                       legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AUDIT TRAIL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Audit Trail":

    st.markdown("""
    <div class="section-header">
        <span class="section-title">📋 Audit Trail</span>
        <span class="section-badge">COMPLIANCE LOG</span>
    </div>
    """, unsafe_allow_html=True)

    audit_df = get_audit_log()

    if audit_df.empty:
        st.markdown("""
        <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; padding: 48px; text-align: center;">
            <div style="font-size: 48px; margin-bottom: 16px;">📋</div>
            <div style="font-size: 18px; color: var(--text-secondary); margin-bottom: 8px;">No decisions recorded yet</div>
            <div style="font-size: 13px; color: var(--text-muted);">Investigate transactions and make decisions to populate the audit trail.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Summary stats
        c1, c2, c3, c4 = st.columns(4)
        confirmed = len(audit_df[audit_df['officer_decision'] == 'CONFIRMED'])
        cleared = len(audit_df[audit_df['officer_decision'] == 'CLEARED'])
        escalated = len(audit_df[audit_df['officer_decision'] == 'ESCALATED'])

        for col, label, val, color in [(c1,'Total Decisions',len(audit_df),'blue'),
                                        (c2,'Fraud Confirmed',confirmed,'red'),
                                        (c3,'Cleared',cleared,'green'),
                                        (c4,'Escalated',escalated,'amber')]:
            with col:
                st.markdown(f"""
                <div class="metric-card {color}">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {color}">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Table header
        st.markdown("""
        <div class="audit-row" style="background:rgba(30,45,74,0.5); border-radius:8px 8px 0 0; font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:1px; color:#4a5678;">
            <span>Transaction ID</span>
            <span>Decision</span>
            <span>Risk Level</span>
            <span>Fraud Prob</span>
            <span>Decided At</span>
        </div>
        """, unsafe_allow_html=True)

        for _, row in audit_df.iterrows():
            decision_cls = f"decision-{row['officer_decision']}"
            prob = row.get('fraud_probability', 0)
            risk = row.get('risk_level', '')
            st.markdown(f"""
            <div class="audit-row">
                <span style="font-family:'JetBrains Mono'; font-size:12px; color:#00d4ff;">{row['transaction_id']}</span>
                <span class="{decision_cls}">{row['officer_decision']}</span>
                <span><div class="risk-badge risk-{risk}" style="font-size:11px;">{risk}</div></span>
                <span style="font-family:'JetBrains Mono'; color:{'#ff3b6b' if prob > 70 else '#ffb800' if prob > 40 else '#00e5a0'};">{prob:.1f}%</span>
                <span style="font-size:12px; color:#4a5678;">{str(row.get('decided_at',''))[:16]}</span>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE FEED
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Live Feed":

    st.markdown("""
    <div class="section-header">
        <span class="section-title">📡 Real-Time Transaction Feed</span>
        <span class="section-badge">LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(45,110,246,0.08); border: 1px solid rgba(45,110,246,0.2); border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
        <div style="font-size: 14px; color: #8899bb; line-height: 1.8;">
            🚀 <strong style="color:#e8edf5;">Start the Transaction Emitter</strong> to see live data appear here.<br>
            Run in a separate terminal: <code style="background: rgba(0,0,0,0.4); padding: 2px 8px; border-radius: 4px; color: #00d4ff; font-family: JetBrains Mono;">python emitter.py</code><br>
            New transactions will stream in every 5 seconds and be auto-analyzed for fraud.
        </div>
    </div>
    """, unsafe_allow_html=True)

    live_count = get_live_transaction_count()
    auto_refresh = st.checkbox("🔄 Auto-refresh (every 5s)", value=False)

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
        <div class="live-indicator"><div class="live-dot"></div> LIVE</div>
        <span style="font-size:13px; color:#8899bb;">{live_count} transactions received</span>
    </div>
    """, unsafe_allow_html=True)

    live_df = get_live_transactions()

    if live_df.empty:
        st.markdown("""
        <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; padding: 48px; text-align: center;">
            <div style="font-size: 48px; margin-bottom: 16px;">📡</div>
            <div style="font-size: 18px; color: var(--text-secondary);">Waiting for live transactions...</div>
            <div style="font-size: 13px; color: var(--text-muted); margin-top: 8px;">Run <code>python emitter.py</code> to start streaming</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Score live transactions
        for _, live_row in live_df.iterrows():
            txn_dict = live_row.to_dict()
            prob, meta = predict_transaction(txn_dict, model, encoders, customers_df, merchants_df, transactions_df)
            if prob is None:
                continue

            fraud_prob = meta.get('fraud_probability', 0)
            flags = meta.get('flags', [])
            risk = 'CRITICAL' if fraud_prob >= 80 else ('HIGH' if fraud_prob >= 60 else ('MEDIUM' if fraud_prob >= 40 else 'LOW'))
            color_m = {'CRITICAL': '#ff3b6b', 'HIGH': '#ffb800', 'MEDIUM': '#00d4ff', 'LOW': '#00e5a0'}[risk]

            if fraud_prob >= 60:
                st.markdown(f"""
                <div class="alert-banner">
                    <div class="alert-icon">🚨</div>
                    <div class="alert-text">FRAUD ALERT: {live_row['transaction_uuid']} — {fraud_prob:.1f}% fraud probability — {', '.join(flags[:2]) if flags else 'Multiple flags'}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="txn-row">
                <div>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <div class="live-indicator" style="font-size:10px; padding:2px 8px;"><div class="live-dot"></div> NEW</div>
                        <div class="txn-id">{live_row['transaction_uuid']}</div>
                    </div>
                    <div class="txn-meta" style="margin-top:4px;">
                        👤 {live_row.get('customer_uuid','')} &nbsp;|&nbsp; 
                        📍 {live_row.get('transaction_location','')} &nbsp;|&nbsp; 
                        🕐 {str(live_row.get('inserted_at',''))[:19]} &nbsp;|&nbsp;
                        ⚠️ {len(flags)} flags
                    </div>
                </div>
                <div style="display:flex; gap:16px; align-items:center;">
                    <div>
                        <div class="txn-amount">₹{float(live_row.get('transaction_amount',0)):,.2f}</div>
                    </div>
                    <div>
                        <div class="risk-badge risk-{risk}">{risk}</div>
                        <div style="font-size:13px; font-family:'JetBrains Mono'; color:{color_m}; text-align:center; margin-top:4px;">{fraud_prob:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if auto_refresh:
        time.sleep(5)
        st.rerun()
