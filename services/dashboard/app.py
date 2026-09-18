"""
Sensex Quant Lab — Institutional Quant Dashboard
Institutional-grade analytics UI: dense, dark, data-first.
Works standalone (local artifacts) or via Prediction API (Render).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared"))

st.set_page_config(
    page_title="Sensex Quant Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API = os.getenv("PREDICTION_API", "").rstrip("/")
REPORT_DIR = ROOT / "artifacts" / "reports"
MODEL_DIR = ROOT / "artifacts" / "models"


# ═══════════════════════════════════════════════════════════════════
#  INSTITUTIONAL THEME  —  deep navy, muted amber / teal, monospace
# ═══════════════════════════════════════════════════════════════════
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

    <style>
    :root {
        /* Palette — institutional, muted, high-contrast */
        --bg-0:      #0b0f17;
        --bg-1:      #101623;
        --bg-2:      #161e2e;
        --panel:     #121a28;
        --panel-hi:  #1a2436;
        --border:    #1f2a3d;
        --border-hi: #2d3b52;

        --text-0:    #e8edf5;   /* primary */
        --text-1:    #a8b2c4;   /* secondary */
        --text-2:    #6a7690;   /* tertiary */

        --amber:     #f0b429;   /* primary accent — Bloomberg-style */
        --amber-dim: #b8860b;
        --teal:      #2dd4bf;   /* secondary accent */
        --green:     #34d399;   /* long / up */
        --red:       #f87171;   /* short / down */
        --blue:      #60a5fa;
        --violet:    #a78bfa;
        --slate:     #64748b;
    }

    /* ─── Shell ─── */
    .stApp {
        background: var(--bg-0);
        background-image:
            linear-gradient(180deg, #0b0f17 0%, #0a0d15 100%);
        color: var(--text-0);
    }
    #MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }

    html, body, [class*="css"], .stApp, .stMarkdown, p, span, div, label, li {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
        font-feature-settings: 'tnum' 1, 'cv11' 1;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* ─── Top bar / Hero ─── */
    .quant-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        padding: 0.9rem 1.2rem;
        background: var(--bg-1);
        border: 1px solid var(--border);
        border-radius: 4px;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    .quant-header-left {
        display: flex; align-items: center; gap: 1rem;
    }
    .quant-logo {
        width: 40px; height: 40px;
        display: grid; place-items: center;
        background: linear-gradient(135deg, var(--amber), var(--amber-dim));
        border-radius: 4px;
        color: #0b0f17;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: -0.05em;
    }
    .quant-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-0);
        letter-spacing: -0.01em;
        margin: 0;
    }
    .quant-subtitle {
        font-size: 0.72rem;
        color: var(--text-2);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 500;
        margin-top: 2px;
    }
    .quant-header-right {
        display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;
    }
    .tag {
        display: inline-flex; align-items:center; gap:0.35rem;
        padding: 0.25rem 0.6rem;
        font-size: 0.68rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: var(--bg-2);
        color: var(--text-1);
        border: 1px solid var(--border);
        border-radius: 3px;
    }
    .tag.live {
        color: var(--green);
        border-color: rgba(52,211,153,0.35);
        background: rgba(52,211,153,0.08);
    }
    .tag.local {
        color: var(--amber);
        border-color: rgba(240,180,41,0.35);
        background: rgba(240,180,41,0.08);
    }
    .tag .dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: currentColor;
        box-shadow: 0 0 6px currentColor;
        animation: statusBlink 2s ease-in-out infinite;
    }
    @keyframes statusBlink {
        0%,100% { opacity: 1; }
        50%     { opacity: 0.3; }
    }

    /* ─── KPI Strip ─── */
    .kpi-strip {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.6rem;
        margin-bottom: 1rem;
    }
    @media (max-width: 1100px) {
        .kpi-strip { grid-template-columns: repeat(2, 1fr); }
    }
    .kpi {
        position: relative;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 0.7rem 0.9rem 0.75rem 0.9rem;
        overflow: hidden;
        transition: border-color 0.15s ease, background 0.15s ease;
    }
    .kpi:hover {
        border-color: var(--border-hi);
        background: var(--panel-hi);
    }
    .kpi::before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: var(--slate);
    }
    .kpi.up::before     { background: var(--green); }
    .kpi.down::before   { background: var(--red); }
    .kpi.accent::before { background: var(--amber); }
    .kpi-label {
        color: var(--text-2);
        font-size: 0.66rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }
    .kpi-value {
        color: var(--text-0);
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.35rem;
        font-weight: 600;
        margin-top: 0.35rem;
        letter-spacing: -0.01em;
        font-variant-numeric: tabular-nums;
    }
    .kpi-value.sm { font-size: 1rem; }
    .kpi-delta {
        display: inline-flex; align-items: center; gap: 0.25rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        margin-top: 0.3rem;
        font-variant-numeric: tabular-nums;
    }
    .kpi-delta.up   { color: var(--green); }
    .kpi-delta.down { color: var(--red); }
    .kpi-delta.neutral { color: var(--text-2); }

    /* ─── Section titles ─── */
    .section-hdr {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.5rem 0;
        margin: 0.6rem 0 0.55rem 0;
        border-bottom: 1px solid var(--border);
    }
    .section-hdr .bar {
        width: 3px; height: 14px;
        background: var(--amber);
        border-radius: 1px;
    }
    .section-hdr .title {
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--text-0);
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .section-hdr .meta {
        margin-left: auto;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: var(--text-2);
        letter-spacing: 0.05em;
    }

    /* ─── Tabs ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        padding: 0;
        border-radius: 0;
        border-bottom: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        border-radius: 0;
        padding: 0 1rem;
        color: var(--text-2);
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        transition: color 0.15s ease, border-color 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-0);
        background: rgba(255,255,255,0.02);
    }
    .stTabs [aria-selected="true"] {
        color: var(--amber) !important;
        background: transparent !important;
        border-bottom: 2px solid var(--amber) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: var(--bg-1);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text-0);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: var(--amber);
        border: 2px solid var(--bg-0);
        box-shadow: 0 0 0 1px var(--amber);
    }
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div > div {
        background: var(--amber);
    }

    /* ─── Dataframe ─── */
    [data-testid="stDataFrame"] {
        border-radius: 4px;
        overflow: hidden;
        border: 1px solid var(--border);
        background: var(--panel);
    }

    /* ─── Expander ─── */
    details[data-testid="stExpander"] {
        background: var(--panel);
        border-radius: 4px;
        border: 1px solid var(--border);
    }
    details[data-testid="stExpander"] summary {
        color: var(--text-1);
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* ─── Mini stat ─── */
    .mini-stat {
        padding: 0.55rem 0.7rem;
        background: var(--bg-2);
        border: 1px solid var(--border);
        border-radius: 4px;
        text-align: left;
    }
    .mini-stat .k {
        color: var(--text-2);
        font-size: 0.62rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .mini-stat .v {
        color: var(--text-0);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 0.25rem;
        font-variant-numeric: tabular-nums;
    }

    /* ─── Model block ─── */
    .model-block {
        background: var(--panel);
        border: 1px solid var(--border);
        border-left: 3px solid var(--amber);
        border-radius: 4px;
        padding: 0.7rem 1rem;
        margin: 0.7rem 0 0.55rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .model-block .name {
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--text-0);
        letter-spacing: 0.02em;
    }
    .model-block .badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--amber);
        background: rgba(240,180,41,0.1);
        border: 1px solid rgba(240,180,41,0.3);
        padding: 0.2rem 0.55rem;
        border-radius: 3px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ─── Plotly containers ─── */
    [data-testid="stPlotlyChart"] {
        border: 1px solid var(--border);
        border-radius: 4px;
        overflow: hidden;
        background: var(--panel);
    }

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--bg-0); }
    ::-webkit-scrollbar-thumb {
        background: var(--border-hi);
        border-radius: 4px;
        border: 2px solid var(--bg-0);
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--slate); }

    /* ─── Streamlit default widget cosmetics ─── */
    .stAlert { border-radius: 4px; }
    button[kind="secondary"], button[kind="primary"] {
        border-radius: 3px !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════
#  DATA HELPERS  (unchanged logic)
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_history_local(limit: int = 1500) -> pd.DataFrame:
    path = REPORT_DIR / "history.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df.tail(limit)
    raw_path = ROOT / "data" / "raw" / "sensex_raw.csv"
    if not raw_path.exists():
        return pd.DataFrame()
    try:
        from sensex_ml.data import load_ohlcv
        from sensex_ml.features import make_features
        raw = load_ohlcv(raw_path)
        feat = make_features(raw)
        cols = [c for c in ["Open","High","Low","Close","RSI_14","Return_std_20","ATR_pct","Return"] if c in feat.columns]
        return feat[cols].dropna().tail(limit)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_leaderboard_local() -> pd.DataFrame:
    path = REPORT_DIR / "leaderboard.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=120)
def fetch_api_predict():
    if not API:
        return None
    import requests
    r = requests.get(f"{API}/predict", timeout=15)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=120)
def fetch_api_model():
    if not API:
        return None
    import requests
    r = requests.get(f"{API}/model", timeout=15)
    r.raise_for_status()
    return r.json()


def local_predict():
    import joblib
    state_path = MODEL_DIR / "latest_state.joblib"
    if not state_path.exists():
        return None
    state = joblib.load(state_path)
    X = state["features"]
    last = float(state["close"])
    out = {"as_of": state["date"], "last_close": last, "models": {}}
    for t, key in [("open_return","open_return"),("close_return","close_return")]:
        p = MODEL_DIR / f"best_{t}.joblib"
        if p.exists():
            m = joblib.load(p)
            r = float(m.predict(X)[0])
            out[key] = r
            out[f"predicted_{'open' if 'open' in t else 'close'}"] = float(last * np.exp(r))
        else:
            out[key] = 0.0
            out[f"predicted_{'open' if 'open' in t else 'close'}"] = last
    return out


# ═══════════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#a8b2c4", size=11),
    margin=dict(l=50, r=25, t=35, b=30),
    legend=dict(
        orientation="h", y=1.08,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=10, color="#a8b2c4"),
    ),
    xaxis=dict(
        gridcolor="rgba(45,59,82,0.35)",
        zerolinecolor="rgba(45,59,82,0.55)",
        linecolor="#1f2a3d",
        tickfont=dict(size=10, color="#6a7690"),
        title_font=dict(size=10, color="#6a7690"),
    ),
    yaxis=dict(
        gridcolor="rgba(45,59,82,0.35)",
        zerolinecolor="rgba(45,59,82,0.55)",
        linecolor="#1f2a3d",
        tickfont=dict(size=10, color="#6a7690"),
        title_font=dict(size=10, color="#6a7690"),
    ),
)


def _delta_kind(v):
    if v > 0: return "up", "▲"
    if v < 0: return "down", "▼"
    return "neutral", "—"


def _fmt_num(v, pct=False):
    if v is None: return "—"
    if isinstance(v, bool): return "✓" if v else "✗"
    if isinstance(v, (int, float)):
        if pct: return f"{v * 100:.2f}%"
        if abs(v) < 1e-4 and v != 0: return f"{v:.2e}"
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    return str(v)


def kpi(label, value, delta=None, kind="neutral", accent=None, small=False):
    """accent: 'up' | 'down' | 'accent' | None"""
    cls = f"kpi {accent}" if accent else "kpi"
    value_cls = "kpi-value sm" if small else "kpi-value"
    delta_html = ""
    if delta is not None:
        delta_html = f'<div class="kpi-delta {kind}">{delta}</div>'
    return f"""
    <div class="{cls}">
        <div class="kpi-label">{label}</div>
        <div class="{value_cls}">{value}</div>
        {delta_html}
    </div>
    """


def section_hdr(title, meta=""):
    meta_html = f'<span class="meta">{meta}</span>' if meta else ""
    return f"""
    <div class="section-hdr">
        <span class="bar"></span>
        <span class="title">{title}</span>
        {meta_html}
    </div>
    """


def mini_stat(label, value):
    return f"""
    <div class="mini-stat">
        <div class="k">{label}</div>
        <div class="v">{value}</div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════
use_api_placeholder = ""  # populated after sidebar toggle

st.markdown(
    """
    <div class="quant-header">
        <div class="quant-header-left">
            <div class="quant-logo">SQ</div>
            <div>
                <div class="quant-title">Sensex Quant Lab</div>
                <div class="quant-subtitle">Systematic Forecasting · Open &amp; Close · Time-Series CV</div>
            </div>
        </div>
        <div class="quant-header-right">
            <span class="tag">NSE · SENSEX</span>
            <span class="tag">1D HORIZON</span>
            <span class="tag">LOG-RETURN TARGET</span>
            <span class="tag">TIME-SERIES CV</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Control Deck")
    use_api = st.toggle("Use Live API", value=bool(API), help="If off, uses local artifacts.")
    lookback = st.slider("Chart lookback (days)", 60, 1500, 400, 20)

    st.markdown("---")
    st.markdown("### Deployment")
    if use_api and API:
        st.markdown('<span class="tag live"><span class="dot"></span>Live API</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="tag local"><span class="dot"></span>Local Artifacts</span>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="margin-top:1rem; color:#6a7690; font-size:0.76rem; line-height:1.75;">
            <div style="color:#a8b2c4; font-weight:700; text-transform:uppercase;
                        letter-spacing:0.08em; font-size:0.68rem; margin-bottom:0.4rem;">
                Stack
            </div>
            <div>· Backend — FastAPI</div>
            <div>· Frontend — Streamlit</div>
            <div>· Training — GitHub Actions</div>
            <div>· Artifacts — committed daily</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
#  PREDICTION
# ═══════════════════════════════════════════════════════════════════
pred = None
model_meta = None
try:
    if use_api and API:
        pred = fetch_api_predict()
        model_meta = fetch_api_model()
    else:
        pred = local_predict()
except Exception as e:
    st.warning(f"API unavailable ({e}). Falling back to local artifacts.")
    pred = local_predict()

if pred is None:
    st.error("No prediction available. Train models first: `python services/training_service/main.py --mode smoke`")
    st.stop()

or_kind, or_arrow = _delta_kind(pred.get("open_return", 0.0))
cr_kind, cr_arrow = _delta_kind(pred.get("close_return", 0.0))
models_txt = pred.get("models") or {}

st.markdown(
    f"""
    <div class="kpi-strip">
        {kpi("As Of", str(pred.get("as_of", "—")), small=True, accent="accent")}
        {kpi("Last Close", f"{pred['last_close']:,.2f}", accent="accent")}
        {kpi("Predicted Open", f"{pred.get('predicted_open', 0):,.2f}",
             f"{or_arrow} {pred.get('open_return', 0) * 100:+.3f}%", or_kind, accent=or_kind)}
        {kpi("Predicted Close", f"{pred.get('predicted_close', 0):,.2f}",
             f"{cr_arrow} {pred.get('close_return', 0) * 100:+.3f}%", cr_kind, accent=cr_kind)}
        {kpi("Models", f"{models_txt.get('open_return','?')} · {models_txt.get('close_return','?')}",
             small=True)}
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════
tab_price, tab_3d, tab_vol, tab_lb, tab_math = st.tabs(
    ["Price & Forecast", "3D Feature Space", "Volatility Regime", "Model Leaderboard", "Methodology"]
)

hist = load_history_local(lookback + 50)
if hist.empty:
    st.warning("No history.csv — run training to generate reports.")
else:
    plot_df = hist.tail(lookback).copy()

    # ─────────────────────── PRICE ───────────────────────
    with tab_price:
        st.markdown(
            section_hdr("Sensex OHLC + Next-Session Forecast",
                        f"{len(plot_df)} sessions · log-scale return target"),
            unsafe_allow_html=True,
        )

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.045,
            row_heights=[0.55, 0.25, 0.20],
            subplot_titles=("OHLC + Forecast", "RSI (14)", "Rolling Volatility"),
        )

        if {"Open","High","Low","Close"}.issubset(plot_df.columns):
            fig.add_trace(
                go.Candlestick(
                    x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
                    low=plot_df["Low"], close=plot_df["Close"], name="OHLC",
                    increasing_line_color="#34d399", decreasing_line_color="#f87171",
                    increasing_fillcolor="#34d399", decreasing_fillcolor="#f87171",
                    line=dict(width=0.6),
                ),
                row=1, col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=plot_df["Close"],
                           line=dict(color="#2dd4bf", width=1.6), name="Close"),
                row=1, col=1,
            )

        try:
            last_dt = plot_df.index[-1]
            next_dt = last_dt + pd.Timedelta(days=1)
            while next_dt.weekday() >= 5:
                next_dt += pd.Timedelta(days=1)

            fig.add_trace(
                go.Scatter(
                    x=[last_dt, next_dt],
                    y=[pred["last_close"], pred.get("predicted_open", pred["last_close"])],
                    mode="lines+markers",
                    line=dict(color="#f0b429", dash="dot", width=1.8),
                    marker=dict(size=8, symbol="diamond",
                                color="#f0b429", line=dict(color="#0b0f17", width=1)),
                    name="Pred Open",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[next_dt], y=[pred.get("predicted_close", pred["last_close"])],
                    mode="markers",
                    marker=dict(size=11, color="#f87171", symbol="x",
                                line=dict(color="#0b0f17", width=1.2)),
                    name="Pred Close",
                ),
                row=1, col=1,
            )
        except Exception:
            pass

        if "RSI_14" in plot_df.columns:
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=plot_df["RSI_14"],
                           line=dict(color="#f0b429", width=1.4), name="RSI(14)"),
                row=2, col=1,
            )
            fig.add_hline(y=70, line_dash="dash", line_color="#f87171",
                          row=2, col=1, opacity=0.35, line_width=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#34d399",
                          row=2, col=1, opacity=0.35, line_width=1)

        vol_col = "Return_std_20" if "Return_std_20" in plot_df.columns else ("ATR_pct" if "ATR_pct" in plot_df.columns else None)
        if vol_col:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index, y=plot_df[vol_col],
                    fill="tozeroy", fillcolor="rgba(96,165,250,0.14)",
                    line=dict(color="#60a5fa", width=1.2),
                    name=vol_col,
                ),
                row=3, col=1,
            )

        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_layout(height=740, xaxis_rangeslider_visible=False)
        for i in (1,2,3):
            fig.update_xaxes(gridcolor="rgba(45,59,82,0.35)", linecolor="#1f2a3d", row=i, col=1)
            fig.update_yaxes(gridcolor="rgba(45,59,82,0.35)", linecolor="#1f2a3d", row=i, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Return distribution
        if "Return" in plot_df.columns or "Close" in plot_df.columns:
            rets = plot_df["Return"] if "Return" in plot_df.columns else np.log(plot_df["Close"]).diff()
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(
                x=rets.dropna(), nbinsx=70, name="Daily log-return",
                marker=dict(color="#60a5fa",
                            line=dict(color="rgba(11,15,23,0.6)", width=0.4)),
            ))
            fig_h.add_vline(x=pred.get("open_return", 0), line_color="#f0b429",
                            line_width=1.6, line_dash="dot",
                            annotation_text="Open",
                            annotation_font=dict(color="#f0b429", size=10))
            fig_h.add_vline(x=pred.get("close_return", 0), line_color="#f87171",
                            line_width=1.6, line_dash="dot",
                            annotation_text="Close",
                            annotation_font=dict(color="#f87171", size=10))
            fig_h.update_layout(**PLOTLY_LAYOUT)
            fig_h.update_layout(
                height=320,
                title=dict(text="Return distribution vs forecast",
                           font=dict(size=12, color="#e8edf5")),
                showlegend=False,
            )
            st.plotly_chart(fig_h, use_container_width=True)

    # ─────────────────────── 3D ───────────────────────
    with tab_3d:
        st.markdown(
            section_hdr("3D Regime Cloud — Momentum × Volatility × RSI",
                        "causal features only"),
            unsafe_allow_html=True,
        )
        df3 = plot_df.copy()
        if "Close" in df3.columns:
            df3["mom_20"] = df3["Close"].pct_change(20)
        if "Return_std_20" in df3.columns:
            df3["vol"] = df3["Return_std_20"]
        elif "Close" in df3.columns:
            df3["vol"] = np.log(df3["Close"]).diff().rolling(20).std()
        df3["rsi"] = df3["RSI_14"] if "RSI_14" in df3.columns else 50.0

        df3 = df3.dropna(subset=[c for c in ["mom_20","vol","rsi"] if c in df3.columns])

        if len(df3) > 30 and {"mom_20","vol","rsi"}.issubset(df3.columns):
            color = df3["Close"].pct_change().shift(-1) if "Close" in df3.columns else df3["mom_20"]

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter3d(
                x=df3["mom_20"], y=df3["vol"], z=df3["rsi"],
                mode="markers",
                marker=dict(
                    size=3,
                    color=color,
                    colorscale=[[0,"#f87171"],[0.5,"#f0b429"],[1,"#34d399"]],
                    opacity=0.75,
                    colorbar=dict(
                        title=dict(
                            text="Next ret",
                            font=dict(color="#a8b2c4", size=11),
                        ),
                        tickfont=dict(color="#6a7690", size=10),
                        thickness=12,
                        len=0.7,
                        outlinewidth=0,
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    line=dict(width=0),
                ),
                text=df3.index.astype(str),
                name="Regime",
                hovertemplate="Mom: %{x:.4f}<br>Vol: %{y:.4f}<br>RSI: %{z:.2f}<extra></extra>",
            ))
            tail = df3.tail(60)
            fig3.add_trace(go.Scatter3d(
                x=tail["mom_20"], y=tail["vol"], z=tail["rsi"],
                mode="lines",
                line=dict(color="#2dd4bf", width=3),
                opacity=0.55, name="Recent path", hoverinfo="skip",
            ))
            last = df3.iloc[-1]
            fig3.add_trace(go.Scatter3d(
                x=[last["mom_20"]], y=[last["vol"]], z=[last["rsi"]],
                mode="markers",
                marker=dict(size=10, color="#f0b429", symbol="diamond",
                            line=dict(color="#0b0f17", width=1.5)),
                name="Latest",
            ))
            fig3.update_layout(
                height=620,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#a8b2c4", size=11),
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)",
                            font=dict(size=10, color="#a8b2c4")),
                scene=dict(
                    xaxis_title="Momentum 20d",
                    yaxis_title="Volatility",
                    zaxis_title="RSI(14)",
                    bgcolor="rgba(11,15,23,0.85)",
                    xaxis=dict(gridcolor="rgba(45,59,82,0.5)", zerolinecolor="rgba(45,59,82,0.8)",
                               tickfont=dict(color="#6a7690", size=9),
                               title_font=dict(color="#6a7690", size=10)),
                    yaxis=dict(gridcolor="rgba(45,59,82,0.5)", zerolinecolor="rgba(45,59,82,0.8)",
                               tickfont=dict(color="#6a7690", size=9),
                               title_font=dict(color="#6a7690", size=10)),
                    zaxis=dict(gridcolor="rgba(45,59,82,0.5)", zerolinecolor="rgba(45,59,82,0.8)",
                               tickfont=dict(color="#6a7690", size=9),
                               title_font=dict(color="#6a7690", size=10)),
                    camera=dict(eye=dict(x=1.6, y=1.4, z=1.0)),
                ),
            )
            st.plotly_chart(fig3, use_container_width=True)

            # 3D surface — FIXED colorbar API
            try:
                pivot = df3.copy()
                pivot["mom_bin"] = pd.qcut(pivot["mom_20"], 12, duplicates="drop")
                pivot["vol_bin"] = pd.qcut(pivot["vol"], 12, duplicates="drop")
                grid = pivot.groupby(["mom_bin","vol_bin"], observed=True)["rsi"].mean().unstack()
                if grid.shape[0] > 2 and grid.shape[1] > 2:
                    fig_s = go.Figure(data=[go.Surface(
                        z=grid.values,
                        colorscale=[[0,"#1e3a8a"],[0.4,"#2dd4bf"],[0.7,"#f0b429"],[1,"#f87171"]],
                        showscale=True,
                        colorbar=dict(
                            title=dict(
                                text="Mean RSI",
                                font=dict(color="#a8b2c4", size=11),
                            ),
                            tickfont=dict(color="#6a7690", size=10),
                            thickness=12,
                            len=0.7,
                            outlinewidth=0,
                            bgcolor="rgba(0,0,0,0)",
                        ),
                        contours=dict(
                            z=dict(show=True, usecolormap=True,
                                   highlightcolor="#f0b429",
                                   project=dict(z=True))
                        ),
                        lighting=dict(ambient=0.6, diffuse=0.85, specular=0.8,
                                      roughness=0.4, fresnel=0.25),
                    )])
                    fig_s.update_layout(
                        height=520,
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", color="#a8b2c4", size=11),
                        title=dict(
                            text="Mean RSI surface over (momentum × volatility) bins",
                            font=dict(size=12, color="#e8edf5"),
                        ),
                        margin=dict(l=0, r=0, t=40, b=0),
                        scene=dict(
                            xaxis_title="Vol bin", yaxis_title="Mom bin", zaxis_title="RSI",
                            bgcolor="rgba(11,15,23,0.85)",
                            xaxis=dict(gridcolor="rgba(45,59,82,0.5)",
                                       tickfont=dict(color="#6a7690", size=9),
                                       title_font=dict(color="#6a7690", size=10)),
                            yaxis=dict(gridcolor="rgba(45,59,82,0.5)",
                                       tickfont=dict(color="#6a7690", size=9),
                                       title_font=dict(color="#6a7690", size=10)),
                            zaxis=dict(gridcolor="rgba(45,59,82,0.5)",
                                       tickfont=dict(color="#6a7690", size=9),
                                       title_font=dict(color="#6a7690", size=10)),
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                        ),
                    )
                    st.plotly_chart(fig_s, use_container_width=True)
            except Exception as e:
                st.info(f"Surface skipped: {e}")
        else:
            st.info("Not enough columns for 3D view. Re-run training to generate richer history.")

    # ─────────────────────── VOL ───────────────────────
    with tab_vol:
        st.markdown(section_hdr("Volatility Regime Monitor"), unsafe_allow_html=True)
        if "Close" in plot_df.columns:
            logp = np.log(plot_df["Close"])
            ret = logp.diff()
            vol20 = ret.rolling(20).std() * np.sqrt(252)
            vol60 = ret.rolling(60).std() * np.sqrt(252)

            fig_v = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  subplot_titles=("Annualized Volatility",
                                                  "Vol-of-Vol (20d rolling σ)"))
            fig_v.add_trace(
                go.Scatter(x=plot_df.index, y=vol20,
                           line=dict(color="#f0b429", width=1.4), name="Vol 20d"),
                row=1, col=1,
            )
            fig_v.add_trace(
                go.Scatter(x=plot_df.index, y=vol60,
                           line=dict(color="#a78bfa", width=1.4), name="Vol 60d"),
                row=1, col=1,
            )
            fig_v.add_trace(
                go.Scatter(x=plot_df.index, y=vol20.rolling(20).std(),
                           fill="tozeroy", fillcolor="rgba(45,212,191,0.14)",
                           line=dict(color="#2dd4bf", width=1.2),
                           name="Vol-of-vol"),
                row=2, col=1,
            )
            fig_v.update_layout(**PLOTLY_LAYOUT)
            fig_v.update_layout(height=520)
            for i in (1,2):
                fig_v.update_xaxes(gridcolor="rgba(45,59,82,0.35)", linecolor="#1f2a3d", row=i, col=1)
                fig_v.update_yaxes(gridcolor="rgba(45,59,82,0.35)", linecolor="#1f2a3d", row=i, col=1)
            st.plotly_chart(fig_v, use_container_width=True)

            if "RSI_14" in plot_df.columns:
                rc = plot_df["RSI_14"].rolling(60).corr(vol20)
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(
                    x=plot_df.index, y=rc,
                    line=dict(color="#60a5fa", width=1.4),
                    name="RSI–Vol corr (60d)",
                ))
                fig_c.add_hline(y=0, line_dash="dash",
                                line_color="#6a7690", opacity=0.4, line_width=1)
                fig_c.update_layout(**PLOTLY_LAYOUT)
                fig_c.update_layout(
                    height=320,
                    title=dict(text="Rolling RSI–Volatility correlation (60d)",
                               font=dict(size=12, color="#e8edf5")),
                    showlegend=False,
                )
                st.plotly_chart(fig_c, use_container_width=True)

    # ─────────────────────── LEADERBOARD ───────────────────────
    with tab_lb:
        lb = load_leaderboard_local()

        if model_meta and model_meta.get("test_metrics"):
            st.markdown(section_hdr("Production Model Metrics", "from API manifest"),
                        unsafe_allow_html=True)
            metrics = model_meta.get("test_metrics") or {}

            for target_key, block in metrics.items():
                if not isinstance(block, dict):
                    continue
                model_name = block.get("model", "—")
                st.markdown(
                    f"""
                    <div class="model-block">
                        <span class="name">{target_key.replace('_',' ').title()}</span>
                        <span class="badge">{model_name}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                da = block.get("DirectionalAccuracy")
                da_kind = "up" if (isinstance(da,(int,float)) and da > 0.5) else "down"
                da_arrow = "▲" if da_kind == "up" else "▼"

                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.markdown(kpi("MAE", _fmt_num(block.get("MAE"))), unsafe_allow_html=True)
                with k2:
                    st.markdown(kpi("RMSE", _fmt_num(block.get("RMSE"))), unsafe_allow_html=True)
                with k3:
                    r2 = block.get("R2")
                    r2_kind = "up" if (isinstance(r2,(int,float)) and r2 > 0) else "down"
                    st.markdown(kpi("R²", _fmt_num(r2), accent=r2_kind), unsafe_allow_html=True)
                with k4:
                    st.markdown(
                        kpi("Directional Accuracy", _fmt_num(da, pct=True),
                            f"{da_arrow} vs 50% baseline" if da is not None else None,
                            da_kind, accent=da_kind),
                        unsafe_allow_html=True,
                    )

                details = {
                    "CV Best MAE": _fmt_num(block.get("cv_best_MAE")),
                    "CV Fits": _fmt_num(block.get("cv_fits")),
                    "Search Time": f"{block['search_seconds']:.2f}s" if isinstance(block.get("search_seconds"),(int,float)) else "—",
                    "Rank by MAE": _fmt_num(block.get("rank_by_MAE")),
                }
                dcols = st.columns(len(details))
                for col, (label, value) in zip(dcols, details.items()):
                    with col:
                        st.markdown(mini_stat(label, value), unsafe_allow_html=True)

                params = block.get("best_params") or {}
                if params:
                    with st.expander("Best hyperparameters"):
                        rows = "".join(
                            f"""
                            <div style="display:flex; justify-content:space-between;
                                        padding:0.35rem 0;
                                        border-bottom:1px solid rgba(45,59,82,0.5);">
                                <span style="color:#6a7690; font-size:0.78rem;">{k}</span>
                                <span style="color:#e8edf5; font-size:0.78rem; font-weight:600;
                                             font-family:'JetBrains Mono', monospace;">{v}</span>
                            </div>
                            """ for k, v in params.items()
                        )
                        st.markdown(f'<div style="padding:0.2rem 0;">{rows}</div>',
                                    unsafe_allow_html=True)
                st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

        if not lb.empty:
            st.markdown(section_hdr("Holdout Leaderboard", "sorted by MAE"),
                        unsafe_allow_html=True)
            st.dataframe(
                lb.sort_values(["target","MAE"])[
                    [c for c in ["target","model","MAE","RMSE","R2",
                                 "DirectionalAccuracy","cv_fits"] if c in lb.columns]
                ],
                use_container_width=True,
                hide_index=True,
            )
            for target in lb["target"].unique():
                sub = lb[lb["target"] == target].nsmallest(8, "MAE")
                fig_b = px.bar(
                    sub, x="MAE", y="model", orientation="h",
                    color="DirectionalAccuracy",
                    color_continuous_scale=[[0,"#1e3a8a"],[0.5,"#2dd4bf"],[1,"#f0b429"]],
                    title=f"{target} — MAE (lower is better)",
                )
                fig_b.update_traces(
                    marker=dict(line=dict(color="rgba(11,15,23,0.8)", width=1)),
                )
                fig_b.update_layout(**PLOTLY_LAYOUT)
                fig_b.update_layout(
                    height=340,
                    title=dict(font=dict(size=12, color="#e8edf5")),
                    coloraxis_colorbar=dict(
                        title=dict(text="Dir. Acc.",
                                   font=dict(color="#a8b2c4", size=10)),
                        tickfont=dict(color="#6a7690", size=10),
                        thickness=10,
                        len=0.7,
                        outlinewidth=0,
                    ),
                )
                st.plotly_chart(fig_b, use_container_width=True)
        elif not (model_meta and model_meta.get("test_metrics")):
            st.info("No leaderboard.csv yet. Run full/daily training.")

    # ─────────────────────── METHODOLOGY ───────────────────────
    with tab_math:
        st.markdown(section_hdr("Methodology"), unsafe_allow_html=True)
        st.markdown(
            r"""
### Forecasting Setup
We predict **one-step log-returns**:
$$
r^O_{t+1} = \log\frac{O_{t+1}}{C_t},\qquad r^C_{t+1} = \log\frac{C_{t+1}}{C_t}
$$

Levels are recovered via the exponential map \( \hat P = C_t\, e^{\hat r} \).

### Causality Guarantee
Every feature is \(\mathcal{F}_t\)-measurable — lags, rolling windows ending at \(t\), calendar Fourier terms.
Targets are strictly \(t+1\). Evaluation uses **TimeSeriesSplit** with a gap plus a chronological holdout.

### Metrics That Matter
| Metric | Interpretation |
|---|---|
| **MAE** of log-return | ≈ typical percentage error |
| **Directional accuracy** | \(\mathbb{P}(\mathrm{sign}(\hat r)=\mathrm{sign}(r))\) |
| **R²** | Often ≤ 0 for daily equity returns (near-martingale); expected, not a bug |

### Stack
- **Train** — `services/training_service` (model zoo + GridSearch inside Pipeline)
- **API** — FastAPI on Render (`/predict`, `/model`, `/leaderboard`)
- **UI** — this Streamlit app (Streamlit Community Cloud)
- **Daily** — GitHub Actions after market close → retrain → commit artifacts → redeploy
            """
        )

st.markdown("---")
st.caption("Research / statistical forecast only — not investment advice. Sensex Quant Lab v3")