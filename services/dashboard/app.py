"""
Sensex Quant Lab — Advanced Streamlit Dashboard
3D surfaces, regime views, feature space, forecast panel, leaderboard.
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
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared"))

st.set_page_config(
    page_title="Sensex Quant Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API = os.getenv("PREDICTION_API", "").rstrip("/")
REPORT_DIR = ROOT / "artifacts" / "reports"
MODEL_DIR = ROOT / "artifacts" / "models"


# ═══════════════════════════════════════════════════════════════════
#  ██  CINEMATIC FX LAYER  ██
# ═══════════════════════════════════════════════════════════════════

# ── 1. Fonts + master CSS ─────────────────────────────────────────
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;800&family=Orbitron:wght@500;700;900&display=swap" rel="stylesheet">

    <style>
    /* ═══════════ ROOT VARS ═══════════ */
    :root {
        --neon-cyan:   #00e5ff;
        --neon-violet: #7c3aed;
        --neon-pink:   #db2777;
        --neon-green:  #22c55e;
        --neon-red:    #ef4444;
        --neon-amber:  #fbbf24;
        --bg-0: #05070d;
        --bg-1: #0a0e17;
        --bg-2: #0b1120;
        --glass: rgba(15,23,42,0.55);
        --stroke: rgba(99,102,241,0.25);
        --text-dim: #94a3b8;
        --text-hi:  #f8fafc;
    }

    /* ═══════════ APP SHELL ═══════════ */
    .stApp {
        background: var(--bg-0);
        background-image:
            radial-gradient(1400px 700px at 8% -8%,  rgba(124,58,237,0.22), transparent 60%),
            radial-gradient(1200px 600px at 100% 0%, rgba(0,229,255,0.15), transparent 55%),
            radial-gradient(900px  500px at 50% 110%, rgba(219,39,119,0.14), transparent 60%),
            linear-gradient(180deg, #05070d 0%, #070b14 55%, #05070d 100%);
        background-attachment: fixed;
        animation: ambient 22s ease-in-out infinite alternate;
    }
    @keyframes ambient {
        0%   { background-position: 0% 0%, 100% 0%, 50% 100%, 0% 0%; }
        100% { background-position: 4% 2%, 96% 4%, 52% 98%, 0% 0%; }
    }

    /* subtle grid overlay */
    .stApp::before {
        content:"";
        position: fixed; inset: 0;
        background-image:
            linear-gradient(rgba(99,102,241,0.045) 1px, transparent 1px),
            linear-gradient(90deg, rgba(99,102,241,0.045) 1px, transparent 1px);
        background-size: 48px 48px, 48px 48px;
        pointer-events: none;
        z-index: 0;
        mask-image: radial-gradient(circle at 50% 30%, black 0%, transparent 78%);
        -webkit-mask-image: radial-gradient(circle at 50% 30%, black 0%, transparent 78%);
    }

    /* vignette */
    .stApp::after {
        content:"";
        position: fixed; inset: 0;
        background: radial-gradient(ellipse at center,
            transparent 45%, rgba(0,0,0,0.55) 100%);
        pointer-events: none;
        z-index: 0;
    }

    #MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }

    html, body, [class*="css"], .stApp, .stMarkdown, p, span, div, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 3rem;
        max-width: 1440px;
        position: relative;
        z-index: 2;
    }

    /* ═══════════ SCANLINE ═══════════ */
    .scanline {
        position: fixed; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(0,229,255,0.55), transparent);
        filter: blur(1px);
        z-index: 3;
        pointer-events: none;
        animation: scan 7.5s linear infinite;
        opacity: 0.5;
    }
    @keyframes scan {
        0%   { top: -10px; opacity: 0; }
        10%  { opacity: 0.6; }
        90%  { opacity: 0.6; }
        100% { top: 100vh; opacity: 0; }
    }

    /* ═══════════ HERO ═══════════ */
    .sql-hero {
        position: relative;
        padding: 2rem 2.2rem 1.9rem 2.2rem;
        border-radius: 22px;
        overflow: hidden;
        isolation: isolate;
        background:
            linear-gradient(135deg,
                rgba(30,58,138,0.55) 0%,
                rgba(124,58,237,0.55) 45%,
                rgba(219,39,119,0.45) 100%);
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow:
            0 20px 60px rgba(124,58,237,0.35),
            0 4px 12px rgba(0,0,0,0.5),
            inset 0 1px 0 rgba(255,255,255,0.16),
            inset 0 -1px 0 rgba(0,0,0,0.3);
        margin-bottom: 1.4rem;
        backdrop-filter: blur(14px);
    }
    /* animated conic sheen */
    .sql-hero::before {
        content:"";
        position:absolute; inset:-60%;
        background: conic-gradient(from 0deg,
            transparent 0deg,
            rgba(0,229,255,0.30) 40deg,
            transparent 90deg,
            transparent 180deg,
            rgba(219,39,119,0.28) 220deg,
            transparent 270deg,
            transparent 360deg);
        animation: heroSpin 14s linear infinite;
        z-index: -1;
        opacity: 0.75;
    }
    @keyframes heroSpin { to { transform: rotate(360deg); } }

    /* glow orbs */
    .sql-hero::after {
        content:"";
        position:absolute; inset:0;
        background:
            radial-gradient(500px 200px at 12% 130%, rgba(0,229,255,0.45), transparent 65%),
            radial-gradient(420px 180px at 92% -20%, rgba(255,82,82,0.35), transparent 70%);
        pointer-events:none;
        z-index: -1;
    }
    .sql-hero h1 {
        margin: 0;
        font-family: 'Orbitron', 'Inter', sans-serif;
        font-size: 2.35rem;
        font-weight: 900;
        letter-spacing: -0.015em;
        color: #ffffff;
        text-shadow:
            0 0 18px rgba(0,229,255,0.55),
            0 0 40px rgba(124,58,237,0.55),
            0 2px 4px rgba(0,0,0,0.6);
        position: relative;
    }
    .sql-hero p {
        margin: 0.5rem 0 0 0;
        color: rgba(226,232,240,0.92);
        font-size: 0.98rem;
        font-weight: 400;
        letter-spacing: 0.01em;
        position: relative;
    }
    .sql-hero .badges {
        margin-top: 1rem;
        display: flex; gap: 0.55rem; flex-wrap: wrap;
        position: relative;
    }

    /* ═══════════ BADGES ═══════════ */
    .sql-badge {
        display:inline-flex; align-items:center; gap:0.35rem;
        padding: 0.32rem 0.78rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #e0e7ff;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.20);
        backdrop-filter: blur(8px);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.10);
        transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    }
    .sql-badge:hover {
        transform: translateY(-1px);
        border-color: rgba(0,229,255,0.55);
        box-shadow: 0 0 18px rgba(0,229,255,0.35), inset 0 1px 0 rgba(255,255,255,0.15);
    }
    .sql-badge::before {
        content:"";
        width:6px; height:6px; border-radius:50%;
        background: var(--neon-cyan);
        box-shadow: 0 0 10px var(--neon-cyan);
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%,100% { opacity: 1; transform: scale(1); }
        50%     { opacity: .35; transform: scale(.7); }
    }

    /* ═══════════ LIVE PILL ═══════════ */
    .live-pill {
        display:inline-flex; align-items:center; gap:0.45rem;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(34,197,94,0.05));
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.4);
        box-shadow: 0 0 20px rgba(34,197,94,0.25);
    }
    .live-pill .dot {
        width:8px; height:8px; border-radius:50%;
        background: #4ade80;
        box-shadow: 0 0 12px #4ade80;
        animation: pulse 1.4s ease-in-out infinite;
    }
    .offline-pill {
        display:inline-flex; align-items:center; gap:0.45rem;
        padding: 0.35rem 0.85rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase;
        background: linear-gradient(135deg, rgba(251,191,36,0.18), rgba(251,191,36,0.05));
        color: #fcd34d;
        border: 1px solid rgba(251,191,36,0.4);
        box-shadow: 0 0 20px rgba(251,191,36,0.22);
    }

    /* ═══════════ METRIC CARDS ═══════════ */
    .metric-card {
        position: relative;
        background:
            linear-gradient(160deg, rgba(30,41,59,0.72) 0%, rgba(15,23,42,0.92) 100%);
        border: 1px solid var(--stroke);
        border-radius: 16px;
        padding: 1rem 1.1rem 0.95rem 1.1rem;
        box-shadow:
            0 10px 32px rgba(0,0,0,0.45),
            inset 0 1px 0 rgba(255,255,255,0.06);
        height: 100%;
        overflow: hidden;
        transition: transform .22s cubic-bezier(.2,.7,.3,1), box-shadow .22s ease, border-color .22s ease;
        backdrop-filter: blur(10px);
    }
    .metric-card::before {
        content:"";
        position:absolute; inset:0;
        background: linear-gradient(120deg, transparent 30%, rgba(0,229,255,0.10) 50%, transparent 70%);
        transform: translateX(-100%);
        transition: transform .8s ease;
        pointer-events: none;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(129,140,248,0.6);
        box-shadow:
            0 18px 48px rgba(124,58,237,0.35),
            0 0 0 1px rgba(129,140,248,0.35),
            inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .metric-card:hover::before { transform: translateX(100%); }

    .metric-card .corner {
        position: absolute; top:0; right:0;
        width: 60px; height: 60px;
        background: radial-gradient(circle at top right, rgba(0,229,255,0.35), transparent 70%);
        pointer-events: none;
    }
    .metric-label {
        color: var(--text-dim);
        font-size: 0.68rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.14em;
    }
    .metric-value {
        color: var(--text-hi);
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 0.4rem;
        letter-spacing: -0.02em;
        text-shadow: 0 0 22px rgba(0,229,255,0.18);
    }
    .metric-delta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        margin-top: 0.45rem;
        display: inline-flex;
        align-items: center;
        gap: 0.28rem;
        padding: 0.16rem 0.55rem;
        border-radius: 8px;
        letter-spacing: -0.01em;
    }
    .metric-delta.up      {
        color:#22c55e;
        background: rgba(34,197,94,0.14);
        box-shadow: inset 0 0 0 1px rgba(34,197,94,0.35), 0 0 18px rgba(34,197,94,0.20);
    }
    .metric-delta.down    {
        color:#ef4444;
        background: rgba(239,68,68,0.14);
        box-shadow: inset 0 0 0 1px rgba(239,68,68,0.35), 0 0 18px rgba(239,68,68,0.20);
    }
    .metric-delta.neutral {
        color:#94a3b8;
        background: rgba(148,163,184,0.12);
        box-shadow: inset 0 0 0 1px rgba(148,163,184,0.28);
    }

    /* ═══════════ SECTION TITLE ═══════════ */
    .section-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0.4rem 0 0.85rem 0;
        display:flex; align-items:center; gap:0.6rem;
        letter-spacing: -0.005em;
    }
    .section-title .dot {
        width: 10px; height:10px; border-radius:50%;
        background: linear-gradient(135deg,#7c3aed,#06b6d4);
        box-shadow: 0 0 14px rgba(124,58,237,0.95), 0 0 28px rgba(6,182,212,0.55);
        animation: pulse 2.2s ease-in-out infinite;
    }

    /* ═══════════ TABS ═══════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: rgba(10,15,26,0.7);
        padding: 0.4rem;
        border-radius: 14px;
        border: 1px solid rgba(99,102,241,0.22);
        backdrop-filter: blur(8px);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 10px;
        padding: 0 1.1rem;
        color: #cbd5e1;
        font-weight: 600;
        font-size: 0.86rem;
        background: transparent;
        border: none;
        transition: all .22s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background: rgba(99,102,241,0.12);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(124,58,237,0.95), rgba(219,39,119,0.85)) !important;
        color: #ffffff !important;
        box-shadow:
            0 8px 24px rgba(124,58,237,0.45),
            inset 0 1px 0 rgba(255,255,255,0.18);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: transparent !important;
    }

    /* ═══════════ SIDEBAR ═══════════ */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(11,17,32,0.98) 0%, rgba(5,7,13,0.98) 100%);
        border-right: 1px solid rgba(99,102,241,0.18);
        box-shadow: 12px 0 40px rgba(0,0,0,0.4);
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #c7d2fe;
        letter-spacing: -0.005em;
    }
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: linear-gradient(135deg,#7c3aed,#06b6d4);
        box-shadow: 0 0 14px rgba(124,58,237,0.75);
        border: none;
    }

    /* ═══════════ INPUTS ═══════════ */
    .stToggle [data-baseweb="checkbox"] div {
        box-shadow: 0 0 12px rgba(124,58,237,0.55);
    }

    /* ═══════════ DATAFRAME ═══════════ */
    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(99,102,241,0.20);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }

    /* ═══════════ EXPANDER ═══════════ */
    details[data-testid="stExpander"] {
        background: rgba(15,23,42,0.55);
        border-radius: 12px;
        border: 1px solid rgba(99,102,241,0.22);
        overflow: hidden;
    }
    details[data-testid="stExpander"] summary { color:#c7d2fe; }

    /* ═══════════ METRIC CARD INNER ═══════════ */
    .mini-stat {
        text-align:center;
        padding: 0.6rem 0.4rem;
        background: rgba(15,23,42,0.55);
        border-radius: 12px;
        border: 1px solid rgba(99,102,241,0.15);
        transition: all .2s ease;
    }
    .mini-stat:hover {
        border-color: rgba(0,229,255,0.4);
        box-shadow: 0 0 20px rgba(0,229,255,0.2);
    }
    .mini-stat .k {
        color:#94a3b8; font-size:0.66rem; font-weight:800;
        text-transform:uppercase; letter-spacing:0.1em;
    }
    .mini-stat .v {
        color:#e2e8f0; font-family:'JetBrains Mono', monospace;
        font-size:1rem; font-weight:700; margin-top:0.3rem;
    }

    /* ═══════════ CUSTOM SCROLLBAR ═══════════ */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #05070d; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #7c3aed, #06b6d4);
        border-radius: 10px;
        border: 2px solid #05070d;
    }
    ::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #8b5cf6, #22d3ee); }

    /* ═══════════ PLOTLY CONTAINER GLOW ═══════════ */
    [data-testid="stPlotlyChart"] {
        border-radius: 16px;
        border: 1px solid rgba(99,102,241,0.18);
        box-shadow: 0 12px 40px rgba(0,0,0,0.45), inset 0 0 60px rgba(124,58,237,0.06);
        overflow: hidden;
        background: rgba(10,15,26,0.45);
    }

    /* ═══════════ SPINNER ═══════════ */
    .stSpinner > div > div { border-top-color: #00e5ff !important; }

    /* ═══════════ ALERTS ═══════════ */
    [data-testid="stAlert"] { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 2. Particle canvas + scanline injected into DOM ───────────────
components.html(
    """
    <script>
    (function(){
        const w = window.parent, d = w.document;
        // Scanline
        if (!d.getElementById('__sql_scanline')) {
            const s = d.createElement('div');
            s.id = '__sql_scanline';
            s.className = 'scanline';
            d.body.appendChild(s);
        }
        // Particle canvas
        if (d.getElementById('__sql_particles')) return;
        const cv = d.createElement('canvas');
        cv.id = '__sql_particles';
        Object.assign(cv.style, {
            position:'fixed', inset:'0', width:'100vw', height:'100vh',
            pointerEvents:'none', zIndex:'1', opacity:'0.55'
        });
        d.body.appendChild(cv);
        const ctx = cv.getContext('2d');
        let W, H, parts = [];
        const COLORS = ['#00e5ff','#7c3aed','#db2777','#22c55e'];
        function size(){ W = cv.width = w.innerWidth; H = cv.height = w.innerHeight; }
        function spawn(){
            return {
                x: Math.random()*W, y: Math.random()*H,
                vx: (Math.random()-0.5)*0.35,
                vy: (Math.random()-0.5)*0.35,
                r: Math.random()*1.6 + 0.4,
                c: COLORS[(Math.random()*COLORS.length)|0],
                a: Math.random()*0.6 + 0.2
            };
        }
        size();
        const N = Math.min(90, Math.floor(W*H/22000));
        for (let i=0;i<N;i++) parts.push(spawn());
        w.addEventListener('resize', size);
        function tick(){
            ctx.clearRect(0,0,W,H);
            for (let i=0;i<parts.length;i++){
                const p = parts[i];
                p.x += p.vx; p.y += p.vy;
                if (p.x < -20) p.x = W+20; if (p.x > W+20) p.x = -20;
                if (p.y < -20) p.y = H+20; if (p.y > H+20) p.y = -20;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
                ctx.fillStyle = p.c;
                ctx.globalAlpha = p.a;
                ctx.shadowBlur = 12;
                ctx.shadowColor = p.c;
                ctx.fill();
            }
            // connective lines
            ctx.shadowBlur = 0;
            for (let i=0;i<parts.length;i++){
                for (let j=i+1;j<parts.length;j++){
                    const a = parts[i], b = parts[j];
                    const dx=a.x-b.x, dy=a.y-b.y;
                    const dist = dx*dx + dy*dy;
                    if (dist < 14000) {
                        ctx.globalAlpha = (1 - dist/14000) * 0.18;
                        ctx.strokeStyle = '#7c3aed';
                        ctx.lineWidth = 0.6;
                        ctx.beginPath();
                        ctx.moveTo(a.x,a.y);
                        ctx.lineTo(b.x,b.y);
                        ctx.stroke();
                    }
                }
            }
            ctx.globalAlpha = 1;
            requestAnimationFrame(tick);
        }
        tick();
    })();
    </script>
    """,
    height=0,
)


# ═══════════════════════════════════════════════════════════════════
#  ██  DATA HELPERS (unchanged logic)  ██
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
        cols = [c for c in ["Open", "High", "Low", "Close", "RSI_14", "Return_std_20", "ATR_pct", "Return"] if c in feat.columns]
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
    for t, key in [("open_return", "open_return"), ("close_return", "close_return")]:
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
#  ██  UI HELPERS  ██
# ═══════════════════════════════════════════════════════════════════
def _delta_kind(v: float):
    if v > 0: return "up", "▲"
    if v < 0: return "down", "▼"
    return "neutral", "•"


def _fmt_num(v, pct=False):
    if v is None: return "—"
    if isinstance(v, bool): return "✅" if v else "❌"
    if isinstance(v, (int, float)):
        if pct: return f"{v * 100:.2f}%"
        if abs(v) < 1e-4 and v != 0: return f"{v:.2e}"
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    return str(v)


def metric_card(label: str, value: str, delta: str | None = None, kind: str = "neutral") -> str:
    delta_html = ""
    if delta is not None:
        delta_html = f'<div class="metric-delta {kind}">{delta}</div>'
    return f"""
    <div class="metric-card">
        <div class="corner"></div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def section_title(text: str) -> str:
    return f'<div class="section-title"><span class="dot"></span>{text}</div>'


def glow_scatter(x, y, name, color="#00e5ff", width=2.2, dash=None, fill=None, showlegend=True):
    """Return a list of 2 traces: soft glow halo + crisp line."""
    return [
        go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=color, width=width * 4, shape="spline"),
            opacity=0.16, hoverinfo="skip", showlegend=False,
            fill=fill,
        ),
        go.Scatter(
            x=x, y=y, mode="lines", name=name,
            line=dict(color=color, width=width, dash=dash, shape="spline"),
            showlegend=showlegend,
            fill=fill,
        ),
    ]


# ═══════════════════════════════════════════════════════════════════
#  ██  HERO  ██
# ═══════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="sql-hero">
        <h1>◈ SENSEX QUANT LAB</h1>
        <p>Next-session Open &amp; Close forecasting · causal features · time-series CV · production artifacts</p>
        <div class="badges">
            <span class="sql-badge">FastAPI Backend</span>
            <span class="sql-badge">Streamlit Frontend</span>
            <span class="sql-badge">TimeSeriesSplit</span>
            <span class="sql-badge">Daily Retrain</span>
            <span class="sql-badge">3D Regime Engine</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════
#  ██  SIDEBAR  ██
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<h2 style='margin-bottom:0.2rem;'>⚙️ Control Deck</h2>"
        "<div style='color:#64748b; font-size:0.78rem; margin-bottom:1rem;'>"
        "Live configuration &amp; deployment status</div>",
        unsafe_allow_html=True,
    )
    use_api = st.toggle("Use Live API", value=bool(API), help="If off, uses local artifacts.")
    lookback = st.slider("Chart lookback (days)", 60, 1500, 400, 20)
    st.markdown("---")
    st.subheader("Deployment")
    mode_label = (
        '<span class="live-pill"><span class="dot"></span>Live API</span>'
        if (use_api and API) else
        '<span class="offline-pill">💾 Local Artifacts</span>'
    )
    st.markdown(mode_label, unsafe_allow_html=True)
    st.markdown(
        """
        <div style="color:#94a3b8; font-size:0.82rem; line-height:1.7; margin-top:1rem;">
            <b style="color:#c7d2fe;">Stack</b><br>
            • Backend: FastAPI<br>
            • Frontend: Streamlit<br>
            • Daily train: GitHub Actions → commit artifacts
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
#  ██  PREDICTION PANEL  ██
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

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(metric_card("As of", str(pred.get("as_of", "—"))), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Last Close", f"{pred['last_close']:,.2f}"), unsafe_allow_html=True)
with c3:
    st.markdown(
        metric_card("Predicted Open", f"{pred.get('predicted_open', 0):,.2f}",
                    f"{or_arrow} {pred.get('open_return', 0) * 100:.3f}%", or_kind),
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        metric_card("Predicted Close", f"{pred.get('predicted_close', 0):,.2f}",
                    f"{cr_arrow} {pred.get('close_return', 0) * 100:.3f}%", cr_kind),
        unsafe_allow_html=True,
    )
with c5:
    st.markdown(
        metric_card("Models", f"{models_txt.get('open_return','?')} / {models_txt.get('close_return','?')}"),
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  ██  TABS  ██
# ═══════════════════════════════════════════════════════════════════
tab_price, tab_3d, tab_vol, tab_lb, tab_math = st.tabs(
    ["Price & Forecast", "3D Feature Space", "Volatility Regime", "Model Leaderboard", "Math Notes"]
)

hist = load_history_local(lookback + 50)
if hist.empty:
    st.warning("No history.csv — run training to generate reports.")
else:
    plot_df = hist.tail(lookback).copy()

    # ─── PRICE ───
    with tab_price:
        st.markdown(section_title("Sensex OHLC + Forecast Overlay"), unsafe_allow_html=True)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.045,
            row_heights=[0.55, 0.25, 0.20],
            subplot_titles=("Sensex OHLC + Next-Day Forecast", "RSI(14)", "Rolling Volatility"),
        )
        if {"Open", "High", "Low", "Close"}.issubset(plot_df.columns):
            fig.add_trace(
                go.Candlestick(
                    x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
                    low=plot_df["Low"], close=plot_df["Close"], name="OHLC",
                    increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
                    increasing_fillcolor="rgba(34,197,94,0.85)",
                    decreasing_fillcolor="rgba(239,68,68,0.85)",
                    line=dict(width=1),
                ),
                row=1, col=1,
            )
        else:
            for tr in glow_scatter(plot_df.index, plot_df["Close"], "Close", "#00e5ff"):
                fig.add_trace(tr, row=1, col=1)

        try:
            last_dt = plot_df.index[-1]
            next_dt = last_dt + pd.Timedelta(days=1)
            while next_dt.weekday() >= 5:
                next_dt += pd.Timedelta(days=1)
            # glow forecast line
            fig.add_trace(
                go.Scatter(
                    x=[last_dt, next_dt],
                    y=[pred["last_close"], pred.get("predicted_open", pred["last_close"])],
                    mode="lines+markers",
                    line=dict(color="#00e5ff", dash="dot", width=6),
                    opacity=0.18, hoverinfo="skip", showlegend=False,
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[last_dt, next_dt],
                    y=[pred["last_close"], pred.get("predicted_open", pred["last_close"])],
                    mode="lines+markers",
                    line=dict(color="#00e5ff", dash="dot", width=2),
                    marker=dict(size=11, symbol="diamond",
                                color="#00e5ff",
                                line=dict(color="#ffffff", width=1)),
                    name="→ Pred Open",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[next_dt],
                    y=[pred.get("predicted_close", pred["last_close"])],
                    mode="markers",
                    marker=dict(size=16, color="#ff5252", symbol="x",
                                line=dict(color="#ffffff", width=1.5)),
                    name="Pred Close",
                ),
                row=1, col=1,
            )
        except Exception:
            pass

        if "RSI_14" in plot_df.columns:
            for tr in glow_scatter(plot_df.index, plot_df["RSI_14"], "RSI", "#fbbf24", 2.0):
                fig.add_trace(tr, row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=2, col=1, opacity=0.55)
            fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", row=2, col=1, opacity=0.55)

        vol_col = "Return_std_20" if "Return_std_20" in plot_df.columns else ("ATR_pct" if "ATR_pct" in plot_df.columns else None)
        if vol_col:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index, y=plot_df[vol_col],
                    fill="tozeroy",
                    fillcolor="rgba(124,58,237,0.18)",
                    line=dict(color="#a78bfa", width=1.6, shape="spline"),
                    name=vol_col,
                ),
                row=3, col=1,
            )

        fig.update_layout(
            height=740, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,15,26,0.35)",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=40, r=20, t=50, b=20),
            font=dict(family="Inter, sans-serif", color="#cbd5e1"),
        )
        fig.update_xaxes(gridcolor="rgba(99,102,241,0.08)", zeroline=False)
        fig.update_yaxes(gridcolor="rgba(99,102,241,0.08)", zeroline=False)
        st.plotly_chart(fig, use_container_width=True)

        # Return distribution with glow
        if "Return" in plot_df.columns or "Close" in plot_df.columns:
            rets = plot_df["Return"] if "Return" in plot_df.columns else np.log(plot_df["Close"]).diff()
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(
                x=rets.dropna(), nbinsx=60, name="Daily log-return",
                marker=dict(
                    color=rets.dropna(),
                    colorscale=[[0,"#7c3aed"],[0.5,"#00e5ff"],[1,"#22c55e"]],
                    line=dict(color="rgba(255,255,255,0.15)", width=0.5),
                ),
            ))
            fig_h.add_vline(x=pred.get("open_return", 0), line_color="#00e5ff",
                            line_width=2, annotation_text="Pred Open ret",
                            annotation_font_color="#00e5ff")
            fig_h.add_vline(x=pred.get("close_return", 0), line_color="#ff5252",
                            line_width=2, annotation_text="Pred Close ret",
                            annotation_font_color="#ff5252")
            fig_h.update_layout(
                height=340, template="plotly_dark",
                title="Return distribution vs forecast",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,15,26,0.35)",
                font=dict(family="Inter, sans-serif", color="#cbd5e1"),
            )
            fig_h.update_xaxes(gridcolor="rgba(99,102,241,0.08)")
            fig_h.update_yaxes(gridcolor="rgba(99,102,241,0.08)")
            st.plotly_chart(fig_h, use_container_width=True)

    # ─── 3D FEATURE SPACE ───
    with tab_3d:
        st.markdown(section_title("3D Regime Cloud — Momentum × Volatility × RSI"), unsafe_allow_html=True)
        df3 = plot_df.copy()
        if "Close" in df3.columns:
            df3["mom_20"] = df3["Close"].pct_change(20)
        if "Return_std_20" in df3.columns:
            df3["vol"] = df3["Return_std_20"]
        elif "Close" in df3.columns:
            df3["vol"] = np.log(df3["Close"]).diff().rolling(20).std()
        if "RSI_14" in df3.columns:
            df3["rsi"] = df3["RSI_14"]
        else:
            df3["rsi"] = 50.0

        df3 = df3.dropna(subset=[c for c in ["mom_20","vol","rsi"] if c in df3.columns])
        if len(df3) > 30 and {"mom_20","vol","rsi"}.issubset(df3.columns):
            color = df3["Close"].pct_change().shift(-1) if "Close" in df3.columns else df3["mom_20"]

            fig3 = go.Figure()
            # main cloud
            fig3.add_trace(go.Scatter3d(
                x=df3["mom_20"], y=df3["vol"], z=df3["rsi"],
                mode="markers",
                marker=dict(
                    size=3.5,
                    color=color,
                    colorscale=[[0,"#ef4444"],[0.5,"#fbbf24"],[1,"#22c55e"]],
                    opacity=0.82,
                    colorbar=dict(
                        title="Next ret (vis)",
                        tickfont=dict(color="#cbd5e1"),
                        titlefont=dict(color="#cbd5e1"),
                    ),
                    line=dict(width=0),
                ),
                text=df3.index.astype(str),
                name="Regime cloud",
                hovertemplate="Mom: %{x:.4f}<br>Vol: %{y:.4f}<br>RSI: %{z:.2f}<extra></extra>",
            ))
            # halo trail (past 60)
            tail = df3.tail(60)
            fig3.add_trace(go.Scatter3d(
                x=tail["mom_20"], y=tail["vol"], z=tail["rsi"],
                mode="lines",
                line=dict(color="#00e5ff", width=3),
                opacity=0.55, name="Recent path", hoverinfo="skip",
            ))
            # latest
            last = df3.iloc[-1]
            fig3.add_trace(go.Scatter3d(
                x=[last["mom_20"]], y=[last["vol"]], z=[last["rsi"]],
                mode="markers",
                marker=dict(size=13, color="#00e5ff", symbol="diamond",
                            line=dict(color="#ffffff", width=2)),
                name="Latest",
            ))
            fig3.update_layout(
                height=620, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(
                    xaxis_title="Momentum 20d",
                    yaxis_title="Volatility",
                    zaxis_title="RSI(14)",
                    bgcolor="rgba(5,7,13,0.9)",
                    xaxis=dict(gridcolor="rgba(99,102,241,0.12)", zerolinecolor="rgba(99,102,241,0.25)"),
                    yaxis=dict(gridcolor="rgba(99,102,241,0.12)", zerolinecolor="rgba(99,102,241,0.25)"),
                    zaxis=dict(gridcolor="rgba(99,102,241,0.12)", zerolinecolor="rgba(99,102,241,0.25)"),
                    camera=dict(eye=dict(x=1.6, y=1.4, z=1.0)),
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                font=dict(family="Inter, sans-serif", color="#cbd5e1"),
            )
            st.plotly_chart(fig3, use_container_width=True)

            # 3D surface
            try:
                pivot = df3.copy()
                pivot["mom_bin"] = pd.qcut(pivot["mom_20"], 12, duplicates="drop")
                pivot["vol_bin"] = pd.qcut(pivot["vol"], 12, duplicates="drop")
                grid = pivot.groupby(["mom_bin","vol_bin"], observed=True)["rsi"].mean().unstack()
                if grid.shape[0] > 2 and grid.shape[1] > 2:
                    fig_s = go.Figure(data=[go.Surface(
                        z=grid.values,
                        colorscale=[[0,"#7c3aed"],[0.5,"#06b6d4"],[1,"#fbbf24"]],
                        showscale=True,
                        contours=dict(
                            z=dict(show=True, usecolormap=True, highlightcolor="#00e5ff", project=dict(z=True))
                        ),
                        lighting=dict(ambient=0.55, diffuse=0.9, specular=1.1, roughness=0.35, fresnel=0.3),
                    )])
                    fig_s.update_layout(
                        height=520, template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        title="Mean RSI surface over (momentum × volatility) bins",
                        scene=dict(
                            xaxis_title="Vol bin", yaxis_title="Mom bin", zaxis_title="RSI",
                            bgcolor="rgba(5,7,13,0.9)",
                            xaxis=dict(gridcolor="rgba(99,102,241,0.12)"),
                            yaxis=dict(gridcolor="rgba(99,102,241,0.12)"),
                            zaxis=dict(gridcolor="rgba(99,102,241,0.12)"),
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                        ),
                        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
                    )
                    st.plotly_chart(fig_s, use_container_width=True)
            except Exception as e:
                st.info(f"Surface skipped: {e}")
        else:
            st.info("Not enough columns for 3D view. Re-run training to generate richer history.")

    # ─── VOLATILITY ───
    with tab_vol:
        st.markdown(section_title("Volatility Regime Monitor"), unsafe_allow_html=True)
        if "Close" in plot_df.columns:
            logp = np.log(plot_df["Close"])
            ret = logp.diff()
            vol20 = ret.rolling(20).std() * np.sqrt(252)
            vol60 = ret.rolling(60).std() * np.sqrt(252)

            fig_v = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  subplot_titles=("Annualized vol", "Vol-of-vol"))
            for tr in glow_scatter(plot_df.index, vol20, "Vol 20d", "#00e5ff", 2.0):
                fig_v.add_trace(tr, row=1, col=1)
            for tr in glow_scatter(plot_df.index, vol60, "Vol 60d", "#db2777", 2.0):
                fig_v.add_trace(tr, row=1, col=1)
            fig_v.add_trace(go.Scatter(
                x=plot_df.index, y=vol20.rolling(20).std(),
                fill="tozeroy", fillcolor="rgba(251,191,36,0.18)",
                line=dict(color="#fbbf24", width=1.6, shape="spline"),
                name="Vol-of-vol",
            ), row=2, col=1)
            fig_v.update_layout(
                height=520, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,15,26,0.35)",
                font=dict(family="Inter, sans-serif", color="#cbd5e1"),
            )
            fig_v.update_xaxes(gridcolor="rgba(99,102,241,0.08)")
            fig_v.update_yaxes(gridcolor="rgba(99,102,241,0.08)")
            st.plotly_chart(fig_v, use_container_width=True)

            if "RSI_14" in plot_df.columns:
                rc = plot_df["RSI_14"].rolling(60).corr(vol20)
                fig_c = go.Figure()
                for tr in glow_scatter(plot_df.index, rc, "RSI–Vol corr (60d)", "#a78bfa", 2.0):
                    fig_c.add_trace(tr)
                fig_c.update_layout(
                    height=320, template="plotly_dark",
                    title="Regime correlation",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,15,26,0.35)",
                    font=dict(family="Inter, sans-serif", color="#cbd5e1"),
                )
                fig_c.update_xaxes(gridcolor="rgba(99,102,241,0.08)")
                fig_c.update_yaxes(gridcolor="rgba(99,102,241,0.08)")
                st.plotly_chart(fig_c, use_container_width=True)

    # ─── LEADERBOARD ───
    with tab_lb:
        lb = load_leaderboard_local()

        if model_meta and model_meta.get("test_metrics"):
            st.markdown(section_title("Production Model Metrics"), unsafe_allow_html=True)
            metrics = model_meta.get("test_metrics") or {}
            for target_key, block in metrics.items():
                if not isinstance(block, dict):
                    continue
                model_name = block.get("model", "—")
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(160deg, rgba(30,41,59,0.72), rgba(15,23,42,0.9));
                        border: 1px solid rgba(99,102,241,0.28);
                        border-radius: 14px;
                        padding: 1rem 1.2rem 0.6rem 1.2rem;
                        margin-bottom: 0.8rem;
                        box-shadow: 0 8px 30px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);">
                        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.5rem;">
                            <div style="font-size:1.05rem; font-weight:700; color:#e2e8f0;">
                                {target_key.replace('_',' ').title()}
                            </div>
                            <span class="sql-badge">{model_name}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                k1, k2, k3, k4 = st.columns(4)
                da = block.get("DirectionalAccuracy")
                da_kind = "up" if (isinstance(da,(int,float)) and da > 0.5) else "down"
                da_arrow = "▲" if da_kind == "up" else "▼"
                with k1:
                    st.markdown(metric_card("MAE", _fmt_num(block.get("MAE"))), unsafe_allow_html=True)
                with k2:
                    st.markdown(metric_card("RMSE", _fmt_num(block.get("RMSE"))), unsafe_allow_html=True)
                with k3:
                    st.markdown(metric_card("R²", _fmt_num(block.get("R2"))), unsafe_allow_html=True)
                with k4:
                    st.markdown(
                        metric_card("Directional Acc.", _fmt_num(da, pct=True),
                                    f"{da_arrow} vs 50% baseline" if da is not None else None, da_kind),
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
                        st.markdown(
                            f'<div class="mini-stat"><div class="k">{label}</div>'
                            f'<div class="v">{value}</div></div>',
                            unsafe_allow_html=True,
                        )
                params = block.get("best_params") or {}
                if params:
                    with st.expander("Best hyperparameters", expanded=False):
                        rows = "".join(
                            f"""<div style="display:flex; justify-content:space-between;
                                            padding:0.4rem 0; border-bottom:1px solid rgba(99,102,241,0.1);">
                                <span style="color:#94a3b8; font-size:0.82rem;">{k}</span>
                                <span style="color:#f8fafc; font-size:0.82rem; font-weight:600;
                                             font-family:'JetBrains Mono', monospace;">{v}</span>
                            </div>""" for k,v in params.items()
                        )
                        st.markdown(f'<div style="padding:0.3rem 0.2rem;">{rows}</div>', unsafe_allow_html=True)
                st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

        if not lb.empty:
            st.markdown(section_title("Holdout Leaderboard"), unsafe_allow_html=True)
            st.dataframe(
                lb.sort_values(["target","MAE"])[
                    [c for c in ["target","model","MAE","RMSE","R2","DirectionalAccuracy","cv_fits"] if c in lb.columns]
                ],
                use_container_width=True,
            )
            for target in lb["target"].unique():
                sub = lb[lb["target"] == target].nsmallest(8, "MAE")
                fig_b = px.bar(
                    sub, x="MAE", y="model", orientation="h",
                    color="DirectionalAccuracy",
                    color_continuous_scale=[[0,"#7c3aed"],[0.5,"#00e5ff"],[1,"#22c55e"]],
                    title=f"{target} — MAE (lower is better)",
                )
                fig_b.update_traces(
                    marker=dict(line=dict(color="rgba(255,255,255,0.15)", width=1)),
                    textposition="outside",
                )
                fig_b.update_layout(
                    height=340, template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,15,26,0.35)",
                    font=dict(family="Inter, sans-serif", color="#cbd5e1"),
                )
                fig_b.update_xaxes(gridcolor="rgba(99,102,241,0.08)")
                fig_b.update_yaxes(gridcolor="rgba(99,102,241,0.08)")
                st.plotly_chart(fig_b, use_container_width=True)
        elif not (model_meta and model_meta.get("test_metrics")):
            st.info("No leaderboard.csv yet. Run full/daily training.")

    # ─── MATH NOTES ───
    with tab_math:
        st.markdown(section_title("Modeling Notes & The Math"), unsafe_allow_html=True)
        st.markdown(
            r"""
### Forecasting setup
We predict **one-step log-returns**:
$$
r^O_{t+1} = \log\frac{O_{t+1}}{C_t},\qquad r^C_{t+1} = \log\frac{C_{t+1}}{C_t}
$$
Levels are recovered by the exponential map \( \hat P = C_t\, e^{\hat r} \).

### Causality
Every feature is \(\mathcal{F}_t\)-measurable (lags, rolling windows ending at \(t\), calendar Fourier terms).  
Targets are strictly \(t+1\). Evaluation uses **TimeSeriesSplit** with a gap and a chronological holdout.

### Metrics that matter
- **MAE** of log-return ≈ typical percentage error  
- **Directional accuracy** = \(\mathbb{P}(\mathrm{sign}(\hat r)=\mathrm{sign}(r))\)  
- \(R^2\) is often ≤ 0 for daily equity returns (near-martingale); that is expected, not a bug.

### Stack
- **Train**: `services/training_service` (model zoo + GridSearch inside Pipeline)  
- **API**: FastAPI on Render (`/predict`, `/model`, `/leaderboard`)  
- **UI**: this Streamlit app (Streamlit Community Cloud)  
- **Daily**: GitHub Actions after market close → retrain → commit artifacts → redeploy  
            """
        )

st.markdown("---")
st.caption("Research / statistical forecast only — not investment advice. Sensex Quant Lab v3")