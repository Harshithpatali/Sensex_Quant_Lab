#!/usr/bin/env python
"""Refresh Sensex OHLCV from Yahoo Finance."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from sensex_ml.config import RAW_PATH
from sensex_ml.data import load_ohlcv

def main():
    try:
        import yfinance as yf
        print("Downloading ^BSESN ...")
        df = yf.download("^BSESN", start="1997-05-16", progress=False, auto_adjust=True)
        if hasattr(df.columns, "levels"):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_PATH)
        print(f"Saved {len(df)} rows → {RAW_PATH}")
    except Exception as e:
        print(f"Download failed ({e}); keeping existing cache if any.")
        if RAW_PATH.exists():
            print(f"Existing: {RAW_PATH}")
        else:
            raise

if __name__ == "__main__":
    main()
