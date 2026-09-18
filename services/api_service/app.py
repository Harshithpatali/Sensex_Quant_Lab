"""
Sensex Prediction API — production FastAPI service.
Deploy on Render (or any ASGI host).
"""
from __future__ import annotations

import math

def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


import json
import sys
from importlib import util
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared"))

_CONFIG_PATH = ROOT / "shared" / "sensex_ml" / "config.py"
_config_spec = util.spec_from_file_location("sensex_ml.config", _CONFIG_PATH)
if _config_spec is None or _config_spec.loader is None:
    raise ImportError(f"Unable to load configuration from {_CONFIG_PATH}")
config = util.module_from_spec(_config_spec)
sys.modules["sensex_ml.config"] = config
_config_spec.loader.exec_module(config)
MODEL_DIR = config.MODEL_DIR
REPORT_DIR = config.REPORT_DIR
APP_ROOT = config.ROOT

app = FastAPI(
    title="Sensex Quant Prediction API",
    version="3.0.0",
    description="Next-session Open/Close return forecasts for BSE Sensex",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_artifacts():
    mf = REPORT_DIR / "manifest.json"
    if not mf.exists():
        raise HTTPException(503, "No trained model. Run training first.")
    manifest = json.loads(mf.read_text())
    models = {}
    for t in ["open_return", "close_return"]:
        rel = manifest.get("best_models", {}).get(t, {}).get("path")
        if rel:
            path = APP_ROOT / rel
            if path.exists():
                models[t] = joblib.load(path)
            else:
                # fallback: direct model dir
                alt = MODEL_DIR / f"best_{t}.joblib"
                models[t] = joblib.load(alt) if alt.exists() else None
        else:
            alt = MODEL_DIR / f"best_{t}.joblib"
            models[t] = joblib.load(alt) if alt.exists() else None
    state_path = MODEL_DIR / "latest_state.joblib"
    if not state_path.exists():
        raise HTTPException(503, "Missing latest_state.joblib")
    state = joblib.load(state_path)
    return manifest, models, state


@app.get("/health")
def health():
    return {"status": "ok", "service": "sensex-prediction-api"}


@app.get("/model")
def model_info():
    m, models, s = _load_artifacts()
    payload = {
        "created_at": m.get("created_at"),
        "mode": m.get("mode"),
        "feature_count": m.get("feature_count"),
        "as_of": s.get("date"),
        "models": {
            k: m.get("best_models", {}).get(k, {}).get("name", "unknown")
            for k in ["open_return", "close_return"]
        },
        "test_metrics": {
            k: m.get("best_models", {}).get(k, {}).get("test_metrics")
            for k in ["open_return", "close_return"]
        },
    }
    return _clean(payload)


@app.get("/predict")
def predict():
    m, models, s = _load_artifacts()
    X = s["features"]
    last = float(s["close"])

    ro = float(models["open_return"].predict(X)[0]) if models.get("open_return") is not None else 0.0
    rc = float(models["close_return"].predict(X)[0]) if models.get("close_return") is not None else 0.0

    return {
        "as_of": s["date"],
        "last_close": last,
        "open_return": ro,
        "close_return": rc,
        "predicted_open": float(last * np.exp(ro)),
        "predicted_close": float(last * np.exp(rc)),
        "models": {
            k: m.get("best_models", {}).get(k, {}).get("name", "unknown")
            for k in ["open_return", "close_return"]
        },
    }


@app.get("/leaderboard")
def leaderboard():
    path = REPORT_DIR / "leaderboard.csv"
    if not path.exists():
        raise HTTPException(404, "No leaderboard")
    import pandas as pd
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


@app.get("/history")
def history(limit: int = 500):
    path = REPORT_DIR / "history.csv"
    if not path.exists():
        raise HTTPException(404, "No history")
    import pandas as pd
    df = pd.read_csv(path, index_col=0, parse_dates=True).tail(limit)
    df = df.reset_index()
    df.columns = [str(c) for c in df.columns]
    # first col is date
    if df.columns[0] != "Date":
        df = df.rename(columns={df.columns[0]: "Date"})
    return json.loads(df.to_json(orient="records", date_format="iso"))
