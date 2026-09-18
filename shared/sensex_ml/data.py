from pathlib import Path
import pandas as pd

def load_ohlcv(path: Path) -> pd.DataFrame:
    # Handles Yahoo Finance's two-row CSV export and ordinary flat CSVs.
    try:
        df=pd.read_csv(path,header=[0,1],index_col=0,parse_dates=True)
        if not isinstance(df.columns,pd.MultiIndex): raise ValueError
        df.columns=[str(c[0]) for c in df.columns]
    except Exception:
        df=pd.read_csv(path,index_col=0,parse_dates=True)
    rename={}
    for c in df.columns:
        s=str(c).lower()
        for key in ['open','high','low','close','volume']:
            if key in s: rename[c]=key.title(); break
    df=df.rename(columns=rename)
    needed=['Open','High','Low','Close']
    missing=[c for c in needed if c not in df]
    if missing: raise ValueError(f'Missing columns: {missing}')
    if 'Volume' not in df: df['Volume']=0.0
    df=df[needed+['Volume']].apply(pd.to_numeric,errors='coerce')
    df=df[~df.index.duplicated(keep='last')].sort_index().dropna(subset=needed)
    if len(df)<500: raise ValueError(f'Only {len(df)} valid OHLC rows; need >=500')
    return df
