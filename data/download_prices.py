import yfinance as yf
import pandas as pd
from pathlib import Path

tickers = [
    "SPY",
    "QQQ",
    "IWM",
    "EEM",
    "TLT",
    "IEF",
    "LQD",
    "HYG",
    "GLD",
    "SLV",
    "USO",
    "DBA",
    "BTC-USD",
    "ETH-USD",
]

output_path = Path("data/cross_asset_daily_data.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

data = yf.download(
    tickers=tickers,
    start="2015-12-07",
    end="2026-10-02",
    interval="1d",
    auto_adjust=False,
    group_by="ticker",
    threads=True,
    progress=True,
)

data = data.sort_index()

data.to_csv(output_path)
