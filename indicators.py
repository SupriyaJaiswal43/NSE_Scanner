"""
Technical Indicators Calculator
EMA 50, EMA 200, RSI, VWAP, ATR, Volume MA
"""

import pandas as pd
import numpy as np


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low  = df["Low"]
    close_prev = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low  - close_prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday VWAP — resets each day automatically since we pass 1-day data."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_vol = typical_price * df["Volume"]
    vwap = tp_vol.cumsum() / df["Volume"].cumsum()
    return vwap


def calc_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["Volume"].rolling(window=period).mean()


def ema_slope(ema_series: pd.Series, lookback: int = 3) -> float:
    """
    Returns slope of EMA over last `lookback` candles.
    Positive → upward sloping.
    """
    if len(ema_series) < lookback:
        return 0.0
    vals = ema_series.iloc[-lookback:].values
    slope = (vals[-1] - vals[0]) / lookback
    return slope


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators to OHLCV DataFrame. Returns enriched DataFrame."""
    df = df.copy()
    df["EMA50"]     = calc_ema(df["Close"], 50)
    df["EMA200"]    = calc_ema(df["Close"], 200)
    df["RSI"]       = calc_rsi(df["Close"], 14)
    df["ATR"]       = calc_atr(df, 14)
    df["VWAP"]      = calc_vwap(df)
    df["VolMA20"]   = calc_volume_ma(df, 20)
    df["EMA50_slope"]  = df["EMA50"].diff(3)
    df["EMA200_slope"] = df["EMA200"].diff(3)
    return df
