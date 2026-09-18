# Sensex Quant Lab — Production Platform

Advanced quantitative forecasting system for **BSE Sensex** next-session **Open** and **Close**.

| Layer | Technology | Host |
|-------|------------|------|
| **Frontend** | Streamlit (3D visuals, regimes, leaderboard) | Streamlit Community Cloud |
| **Backend API** | FastAPI (`/predict`, `/model`, `/leaderboard`) | **Render** |
| **Training** | Model zoo + GridSearch + TimeSeriesSplit | GitHub Actions (daily) |
| **Artifacts** | `joblib` models + reports | Git repo (committed by CI) |

---

## Quick start (local)

```powershell
cd sensex_quant_prod
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:PYTHONPATH = "$PWD\shared"
$env:N_JOBS = "2"

# Train (smoke = fast check, daily = practical, full = research)
python services\training_service\main.py --mode smoke --splits 3
python services\training_service\main.py --mode daily --splits 3

# API
uvicorn services.api_service.app:app --host 0.0.0.0 --port 8000

# Dashboard (another terminal)
streamlit run services\dashboard\app.py
```

Dashboard works **without** API (reads `artifacts/` locally). Set `PREDICTION_API` to use Render.

---

## Deploy

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Sensex Quant Lab v3"
git branch -M main
git remote add origin https://github.com/YOUR_USER/sensex-quant-lab.git
git push -u origin main
```

### 2. Backend on Render

1. [render.com](https://render.com) → New → Blueprint → connect repo  
   **or** New Web Service → root directory of repo  
2. Build: `pip install -r requirements.txt`  
3. Start: `uvicorn services.api_service.app:app --host 0.0.0.0 --port $PORT`  
4. Env: `PYTHONPATH=shared`  
5. Copy the public URL, e.g. `https://sensex-prediction-api.onrender.com`

### 3. Frontend on Streamlit Cloud

1. [share.streamlit.io](https://share.streamlit.io) → New app  
2. Repo + branch `main`  
3. Main file: `services/dashboard/app.py`  
4. **Advanced settings → Secrets** (or env):

```toml
PREDICTION_API = "https://YOUR-RENDER-URL"
```

Or in Streamlit Cloud environment variables: `PREDICTION_API=https://...`

### 4. Daily automation (already in repo)

`.github/workflows/daily_training.yml` runs **Mon–Fri 16:15 IST**:

1. Optional data refresh from Yahoo  
2. `daily` training profile  
3. Commits `artifacts/` + `data/`  
4. Streamlit / Render pick up new commit on next restart / auto-deploy  

Manual full search: Actions → **Full Model Search**.

---

## Project layout

```
sensex_quant_prod/
├── services/
│   ├── api_service/app.py          # FastAPI
│   ├── dashboard/app.py            # Streamlit Quant Lab (3D, regimes, …)
│   └── training_service/           # train + main
├── shared/sensex_ml/               # config, data, features, model_zoo
├── artifacts/models|reports/       # production binaries
├── data/raw/sensex_raw.csv
├── scripts/update_data.py
├── .github/workflows/              # daily + full search
├── render.yaml
└── requirements.txt
```

---

## Metrics philosophy

Daily equity returns are noisy. Report **MAE** and **Directional Accuracy**; negative \(R^2\) is common and not treated as failure. Open-return direction often lands near **~60%** on chronological holdouts; close is closer to **~52–55%**.

---

## Disclaimer

Statistical research tool only. **Not investment advice.**
