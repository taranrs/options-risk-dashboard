import pandas as pd
import yfinance as yf
from math import log
from datetime import datetime, timezone

def get_spot(ticker: str) -> float:
    df = yf.Ticker(ticker).history(period="1d")
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    return float(df["Close"].iloc[-1])

def get_hist(ticker: str, period: str = "2y") -> pd.Series:
    df = yf.Ticker(ticker).history(period=period)
    return df["Close"].rename("close")

def get_risk_free(T_years: float, manual_rate_pct: float | None = None) -> float:
    """
    Return continuous‑compounding risk‑free rate (decimal).
    If manual_rate_pct is provided, use it; otherwise infer:
      T<=0.5y → ^IRX (13w bill); else → ^TNX (10y note) in %.
    """
    if manual_rate_pct is not None:
        return log(1.0 + manual_rate_pct / 100.0)

    try:
        if T_years <= 0.5:
            r_pct = yf.Ticker("^IRX").history(period="5d")["Close"].dropna().iloc[-1]
        else:
            r_pct = yf.Ticker("^TNX").history(period="5d")["Close"].dropna().iloc[-1]
    except Exception:
        r_pct = 4.5
    return log(1.0 + float(r_pct) / 100.0)

# ---------- options helpers (yfinance) ----------
def get_expirations(ticker: str) -> list[str]:
    return yf.Ticker(ticker).options or []

def get_option_chain(ticker: str, expiry: str) -> pd.DataFrame:
    """
    Return combined calls+puts for a single expiry with a 'type' column.
    Columns of interest: ['contractSymbol','lastPrice','bid','ask','strike']
    """
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)
    calls = chain.calls.copy()
    calls["type"] = "call"
    puts = chain.puts.copy()
    puts["type"] = "put"
    df = pd.concat([calls, puts], ignore_index=True)
    df["expiry"] = pd.to_datetime(expiry).tz_localize("UTC")
    return df

def days_to_expiry(expiry_ts) -> float:
    """calendar days between now and expiry"""
    now = datetime.now(timezone.utc)
    return (expiry_ts - now).total_seconds() / 86400.0