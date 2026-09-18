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


# ───────────────────────── data helpers ─────────────────────────
@st.cache_data(ttl=300)
def load_history_local(limit: int = 1500) -> pd.DataFrame:
    path = REPORT_DIR / "history.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df.tail(limit)
    # fallback: rebuild minimal history from raw
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
    """Predict from local joblib artifacts."""
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


# ───────────────────────── UI ─────────────────────────
st.title("📊 Sensex Quant Lab")
st.caption("Next-session Open & Close forecasting · causal features · time-series CV · production artifacts")

with st.sidebar:
    st.header("Controls")
    use_api = st.toggle("Use Render API", value=bool(API), help="If off, uses local artifacts/")
    lookback = st.slider("Chart lookback (days)", 60, 1500, 400, 20)
    st.markdown("---")
    st.subheader("Deployment")
    st.code(f"API = {API or '(local artifacts)'}", language="text")
    st.markdown(
        """
        **Stack**
        - Backend: FastAPI on Render  
        - Frontend: this Streamlit app  
        - Daily train: GitHub Actions → commit artifacts  
        """
    )

# Prediction panel
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

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("As of", pred.get("as_of", "—"))
c2.metric("Last Close", f"{pred['last_close']:,.2f}")
c3.metric(
    "Predicted Open",
    f"{pred.get('predicted_open', 0):,.2f}",
    delta=f"{pred.get('open_return', 0)*100:.3f}%",
)
c4.metric(
    "Predicted Close",
    f"{pred.get('predicted_close', 0):,.2f}",
    delta=f"{pred.get('close_return', 0)*100:.3f}%",
)
models_txt = pred.get("models") or {}
c5.metric("Models", f"{models_txt.get('open_return', '?')} / {models_txt.get('close_return', '?')}")

# Tabs
tab_price, tab_3d, tab_vol, tab_lb, tab_math = st.tabs(
    ["Price & Forecast", "3D Feature Space", "Volatility Regime", "Model Leaderboard", "Math Notes"]
)

hist = load_history_local(lookback + 50)
if hist.empty:
    st.warning("No history.csv — run training to generate reports.")
else:
    plot_df = hist.tail(lookback).copy()

    # ─── Price tab ───
    with tab_price:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            row_heights=[0.55, 0.25, 0.20],
            subplot_titles=("Sensex OHLC + Next-Day Forecast", "RSI(14)", "Rolling Volatility"),
        )
        if {"Open", "High", "Low", "Close"}.issubset(plot_df.columns):
            fig.add_trace(
                go.Candlestick(
                    x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
                    low=plot_df["Low"], close=plot_df["Close"], name="OHLC",
                ),
                row=1, col=1,
            )
        else:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Close"], name="Close"), row=1, col=1)

        # forecast markers
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
                    line=dict(color="#00e5ff", dash="dot", width=2),
                    marker=dict(size=11, symbol="diamond"),
                    name="→ Pred Open",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[next_dt],
                    y=[pred.get("predicted_close", pred["last_close"])],
                    mode="markers",
                    marker=dict(size=14, color="#ff5252", symbol="x"),
                    name="Pred Close",
                ),
                row=1, col=1,
            )
        except Exception:
            pass

        if "RSI_14" in plot_df.columns:
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=plot_df["RSI_14"], line=dict(color="#ffd54f"), name="RSI"),
                row=2, col=1,
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)

        vol_col = "Return_std_20" if "Return_std_20" in plot_df.columns else ("ATR_pct" if "ATR_pct" in plot_df.columns else None)
        if vol_col:
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=plot_df[vol_col], fill="tozeroy", line=dict(color="#7e57c2"), name=vol_col),
                row=3, col=1,
            )

        fig.update_layout(
            height=720, template="plotly_dark", xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", y=1.08), margin=dict(l=40, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # return distribution
        if "Return" in plot_df.columns or "Close" in plot_df.columns:
            rets = plot_df["Return"] if "Return" in plot_df.columns else np.log(plot_df["Close"]).diff()
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(x=rets.dropna(), nbinsx=60, name="Daily log-return", marker_color="#26c6da"))
            fig_h.add_vline(x=pred.get("open_return", 0), line_color="#00e5ff", annotation_text="Pred Open ret")
            fig_h.add_vline(x=pred.get("close_return", 0), line_color="#ff5252", annotation_text="Pred Close ret")
            fig_h.update_layout(height=320, template="plotly_dark", title="Return distribution vs forecast")
            st.plotly_chart(fig_h, use_container_width=True)

    # ─── 3D Feature Space ───
    with tab_3d:
        st.markdown("**3D regime view** — axes are short-term momentum, volatility, and RSI (causal features only).")
        need = []
        # build proxy features from available columns
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

        df3 = df3.dropna(subset=[c for c in ["mom_20", "vol", "rsi"] if c in df3.columns])
        if len(df3) > 30 and {"mom_20", "vol", "rsi"}.issubset(df3.columns):
            # color by forward-ish proxy: next-day style using lag of return for visualization only on past
            color = df3["Close"].pct_change().shift(-1) if "Close" in df3.columns else df3["mom_20"]
            fig3 = go.Figure(
                data=[
                    go.Scatter3d(
                        x=df3["mom_20"],
                        y=df3["vol"],
                        z=df3["rsi"],
                        mode="markers",
                        marker=dict(
                            size=3,
                            color=color,
                            colorscale="RdYlGn",
                            opacity=0.75,
                            colorbar=dict(title="Next ret (vis)"),
                        ),
                        text=df3.index.astype(str),
                        name="Regime cloud",
                    )
                ]
            )
            # current point
            last = df3.iloc[-1]
            fig3.add_trace(
                go.Scatter3d(
                    x=[last["mom_20"]], y=[last["vol"]], z=[last["rsi"]],
                    mode="markers",
                    marker=dict(size=10, color="#00e5ff", symbol="diamond"),
                    name="Latest",
                )
            )
            fig3.update_layout(
                height=560,
                template="plotly_dark",
                scene=dict(
                    xaxis_title="Momentum 20d",
                    yaxis_title="Volatility",
                    zaxis_title="RSI(14)",
                    bgcolor="#0e1117",
                ),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig3, use_container_width=True)

            # 3D surface: vol × mom grid of mean RSI (smooth view)
            try:
                pivot = df3.copy()
                pivot["mom_bin"] = pd.qcut(pivot["mom_20"], 12, duplicates="drop")
                pivot["vol_bin"] = pd.qcut(pivot["vol"], 12, duplicates="drop")
                grid = pivot.groupby(["mom_bin", "vol_bin"], observed=True)["rsi"].mean().unstack()
                if grid.shape[0] > 2 and grid.shape[1] > 2:
                    fig_s = go.Figure(
                        data=[go.Surface(z=grid.values, colorscale="Viridis", showscale=True)]
                    )
                    fig_s.update_layout(
                        height=480,
                        template="plotly_dark",
                        title="Mean RSI surface over (momentum × volatility) bins",
                        scene=dict(xaxis_title="Vol bin", yaxis_title="Mom bin", zaxis_title="RSI"),
                    )
                    st.plotly_chart(fig_s, use_container_width=True)
            except Exception as e:
                st.info(f"Surface skipped: {e}")
        else:
            st.info("Not enough columns for 3D view. Re-run training to generate richer history.")

    # ─── Volatility regime ───
    with tab_vol:
        if "Close" in plot_df.columns:
            logp = np.log(plot_df["Close"])
            ret = logp.diff()
            vol20 = ret.rolling(20).std() * np.sqrt(252)
            vol60 = ret.rolling(60).std() * np.sqrt(252)
            fig_v = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Annualized vol", "Vol-of-vol"))
            fig_v.add_trace(go.Scatter(x=plot_df.index, y=vol20, name="Vol 20d"), row=1, col=1)
            fig_v.add_trace(go.Scatter(x=plot_df.index, y=vol60, name="Vol 60d"), row=1, col=1)
            fig_v.add_trace(go.Scatter(x=plot_df.index, y=vol20.rolling(20).std(), name="Vol-of-vol", fill="tozeroy"), row=2, col=1)
            fig_v.update_layout(height=500, template="plotly_dark")
            st.plotly_chart(fig_v, use_container_width=True)

            # rolling correlation proxy: mom vs vol
            if "RSI_14" in plot_df.columns:
                rc = plot_df["RSI_14"].rolling(60).corr(vol20)
                fig_c = go.Figure(go.Scatter(x=plot_df.index, y=rc, name="RSI–Vol corr (60d)"))
                fig_c.update_layout(height=300, template="plotly_dark", title="Regime correlation")
                st.plotly_chart(fig_c, use_container_width=True)

    # ─── Leaderboard ───
    with tab_lb:
        lb = load_leaderboard_local()
        if model_meta and model_meta.get("test_metrics"):
            st.subheader("Production model metrics (from API/manifest)")
            st.json(model_meta.get("test_metrics"))
        if not lb.empty:
            st.subheader("Holdout leaderboard")
            st.dataframe(
                lb.sort_values(["target", "MAE"])[
                    [c for c in ["target", "model", "MAE", "RMSE", "R2", "DirectionalAccuracy", "cv_fits"] if c in lb.columns]
                ],
                use_container_width=True,
            )
            for target in lb["target"].unique():
                sub = lb[lb["target"] == target].nsmallest(8, "MAE")
                fig_b = px.bar(
                    sub, x="MAE", y="model", orientation="h", color="DirectionalAccuracy",
                    color_continuous_scale="Tealgrn", title=f"{target} — MAE (lower better)",
                )
                fig_b.update_layout(height=320, template="plotly_dark")
                st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.info("No leaderboard.csv yet. Run full/daily training.")

    # ─── Math ───
    with tab_math:
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
