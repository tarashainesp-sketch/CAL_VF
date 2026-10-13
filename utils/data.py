from __future__ import annotations
import numpy as np
import pandas as pd

def _read_yfinance_multiheader(path):
    df=pd.read_csv(path,header=[0,1],index_col=0)
    df.index=pd.to_datetime(df.index,errors="coerce")
    df=df[~df.index.isna()].copy(); df.index.name="date"
    return df

def build_weekly_ml_from_yfinance(path, task_symbols, price_field="Adj Close", n_rv_lags=4):
    raw=_read_yfinance_multiheader(path); rows=[]
    tops=set(raw.columns.get_level_values(0))
    for task_id,symbol in task_symbols.items():
        if symbol not in tops: raise ValueError(f"Ticker {symbol!r} not found.")
        sub=raw[symbol]
        field=price_field if price_field in sub.columns else "Close"
        s=sub[field].astype(float).dropna().sort_index()
        r=np.log(s).diff().dropna()
        weekly=pd.DataFrame(index=r.resample("W-FRI").size().index)
        weekly["rv"]=np.sqrt(r.pow(2).resample("W-FRI").sum())
        weekly["weekly_return"]=r.resample("W-FRI").sum()
        weekly["abs_weekly_return"]=weekly["weekly_return"].abs()
        for lag in range(1,n_rv_lags+1):
            weekly[f"rv_lag{lag}"]=weekly["rv"].shift(lag-1)
        weekly["target"]=weekly["rv"].shift(-1)
        weekly["task_id"]=int(task_id); weekly["date"]=weekly.index
        rows.append(weekly.dropna().reset_index(drop=True))
    out=pd.concat(rows,ignore_index=True).sort_values(["date","task_id"]).reset_index(drop=True)
    feats=["weekly_return","abs_weekly_return",*[f"rv_lag{i}" for i in range(1,n_rv_lags+1)]]
    return out,feats

def chronological_split(df, train_frac=.60, val_frac=.20):
    dates=np.array(sorted(pd.to_datetime(df["date"]).unique())); n=len(dates)
    a=int(n*train_frac); b=a+int(n*val_frac)
    tr=set(dates[:a]); va=set(dates[a:b]); te=set(dates[b:])
    return df[df.date.isin(tr)].copy(),df[df.date.isin(va)].copy(),df[df.date.isin(te)].copy()

def to_xy(df, feature_cols, target_col="target"):
    X=df[feature_cols].to_numpy(float); t=df["task_id"].to_numpy(float).reshape(-1,1)
    return np.hstack([X,t]),df[target_col].to_numpy(float)
