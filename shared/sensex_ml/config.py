from pathlib import Path
import os
ROOT = Path(os.getenv('APP_ROOT', Path(__file__).resolve().parents[2])).resolve()
RAW_PATH = ROOT/'data/raw/sensex_raw.csv'
ARTIFACT_DIR = ROOT/'artifacts'; MODEL_DIR=ARTIFACT_DIR/'models'; REPORT_DIR=ARTIFACT_DIR/'reports'; REGISTRY_DIR=ARTIFACT_DIR/'registry'
PROCESSED_DIR=ROOT/'data/processed'; LOG_DIR=ROOT/'logs'
RANDOM_STATE=42; N_JOBS=int(os.getenv('N_JOBS','-1')); N_SPLITS=int(os.getenv('CV_SPLITS','3')); TEST_FRACTION=float(os.getenv('TEST_FRACTION','0.15'))
for p in [MODEL_DIR,REPORT_DIR,REGISTRY_DIR,PROCESSED_DIR,LOG_DIR]: p.mkdir(parents=True,exist_ok=True)
