import numpy as np, pandas as pd

def _ema(s,w): return s.ewm(span=w,adjust=False,min_periods=w).mean()
def _rsi(close,w=14):
    d=close.diff(); up=d.clip(lower=0); down=-d.clip(upper=0)
    au=up.ewm(alpha=1/w,adjust=False,min_periods=w).mean(); ad=down.ewm(alpha=1/w,adjust=False,min_periods=w).mean()
    rs=au/(ad+1e-12); return 100-100/(1+rs)
def _atr(h,l,c,w=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/w,adjust=False,min_periods=w).mean()
def _macd(c):
    m=_ema(c,12)-_ema(c,26); sig=_ema(m,9); return m,sig,m-sig

def make_features(df):
    x=df.copy().sort_index()
    c,h,l,o,v=x.Close,x.High,x.Low,x.Open,x.Volume
    x['Return']=np.log(c).diff(); x['Open_Return']=np.log(o/c.shift(1)); x['HL_Range']=(h-l)/c; x['OC_Range']=(c-o)/o
    for w in [3,5,10,20,50,100,200]:
        sma=c.rolling(w).mean(); ema=_ema(c,w)
        x[f'Close_SMA_{w}_ratio']=c/sma-1; x[f'Close_EMA_{w}_ratio']=c/ema-1
        x[f'Return_mean_{w}']=x.Return.rolling(w).mean(); x[f'Return_std_{w}']=x.Return.rolling(w).std(); x[f'Return_skew_{w}']=x.Return.rolling(w).skew()
    x['SMA_20_slope']=c.rolling(20).mean().pct_change(5); x['SMA_50_slope']=c.rolling(50).mean().pct_change(5); x['SMA_200_slope']=c.rolling(200).mean().pct_change(10)
    x['RSI_14']=_rsi(c); x['ROC_10']=c.pct_change(10)*100
    low14=l.rolling(14).min(); high14=h.rolling(14).max(); k=100*(c-low14)/(high14-low14+1e-12); x['Stoch_K']=k; x['Stoch_D']=k.rolling(3).mean()
    macd,sig,hist=_macd(c); x['MACD']=macd; x['MACD_Signal']=sig; x['MACD_Hist']=hist
    tr=_atr(h,l,c); x['ATR_pct']=tr/c
    x['ADX_proxy']=((c.diff().abs()).rolling(14).mean()/(tr+1e-12)*100).clip(0,100)
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); x['BB_Width']=4*sd/(mid+1e-12); x['BB_PctB']=(c-(mid-2*sd))/(4*sd+1e-12)
    kc_mid=_ema(c,20); kc_rng=tr.rolling(20).mean(); x['KC_Width']=4*kc_rng/(kc_mid+1e-12)
    hl=np.log((h/l).clip(lower=1)); x['Parkinson_vol']=np.sqrt(hl.pow(2)/(4*np.log(2))).rolling(20).mean()
    if v.sum()>0:
        x['Log_Volume']=np.log1p(v); x['Volume_ratio']=v/(v.rolling(20).mean()+1e-12); direction=np.sign(c.diff()).fillna(0); x['OBV']=(direction*v).cumsum(); x['OBV_change']=x.OBV.pct_change(5).replace([np.inf,-np.inf],np.nan)
    else: x['Log_Volume']=0.; x['Volume_ratio']=1.; x['OBV_change']=0.
    for lag in range(1,21): x[f'Return_lag_{lag}']=x.Return.shift(lag); x[f'Open_Return_lag_{lag}']=x.Open_Return.shift(lag)
    dow=x.index.dayofweek; month=x.index.month
    x['dow_sin']=np.sin(2*np.pi*dow/5); x['dow_cos']=np.cos(2*np.pi*dow/5); x['month_sin']=np.sin(2*np.pi*month/12); x['month_cos']=np.cos(2*np.pi*month/12)
    x['RSI_x_Vol']=x.RSI_14*x.Return_std_20; x['Trend_x_Vol']=x.SMA_50_slope*x.Return_std_20; x['MACD_x_ADX']=x.MACD_Hist*x.ADX_proxy
    x['High_252_dist']=c/c.rolling(252).max()-1; x['Low_252_dist']=c/c.rolling(252).min()-1
    x['Target_Open_Return']=np.log(o.shift(-1)/c); x['Target_Close_Return']=np.log(c.shift(-1)/c)
    return x

def feature_columns(df):
    raw={'Open','High','Low','Close','Volume','Target_Open_Return','Target_Close_Return'}
    return [c for c in df.columns if c not in raw and not c.startswith('Target_')]
