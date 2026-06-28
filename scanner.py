"""
NSE EMA Breakout Scanner
Core logic: buy signal detection + scoring/ranking
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pytz

from indicators import add_all_indicators

IST = pytz.timezone("Asia/Kolkata")


# ─── FILTER CONFIG (toggle in app.py) ────────────────────────────────────────
DEFAULT_FILTERS = {
    "volume_filter":      True,   # Volume > 20-period MA
    "ema50_slope":        True,   # EMA50 sloping upward
    "ema200_flat_rising": True,   # EMA200 flat or rising
    "green_candle":       True,   # Current candle must be green (Close > Open)
    "min_body_pct":       0.3,    # Candle body ≥ 0.3% of price
    "rsi_filter":         True,   # RSI between 45–75
    "vwap_filter":        True,   # Price above VWAP
    "atr_filter":         True,   # Candle range ≥ 0.5× ATR
}


def check_buy_signal(df: pd.DataFrame, filters: dict) -> dict | None:
    """
    Evaluate buy conditions on a prepared (indicators-added) DataFrame.
    Returns signal dict if triggered, else None.
    """
    if len(df) < 5:
        return None

    # ── Grab last 2 candles ──────────────────────────────────────────────────
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # ── CORE CONDITIONS ───────────────────────────────────────────────────────
    # 1. Price > EMA50
    cond1 = curr["Close"] > curr["EMA50"]

    # 2. EMA200 crossover (prev close ≤ EMA200, curr close > EMA200)
    cond2 = (prev["Close"] <= prev["EMA200"]) and (curr["Close"] > curr["EMA200"])

    if not (cond1 and cond2):
        return None

    # ── OPTIONAL FILTERS ──────────────────────────────────────────────────────
    fails = []

    if filters.get("volume_filter") and not pd.isna(curr["VolMA20"]):
        if curr["Volume"] <= curr["VolMA20"]:
            fails.append("vol")

    if filters.get("ema50_slope"):
        if curr["EMA50_slope"] <= 0:
            fails.append("ema50_slope")

    if filters.get("ema200_flat_rising"):
        if curr["EMA200_slope"] < -0.05:          # allow tiny dip
            fails.append("ema200_slope")

    if filters.get("green_candle"):
        if curr["Close"] <= curr["Open"]:
            fails.append("green")

    if filters.get("min_body_pct", 0) > 0:
        body_pct = abs(curr["Close"] - curr["Open"]) / curr["Open"] * 100
        if body_pct < filters["min_body_pct"]:
            fails.append("body")

    if filters.get("rsi_filter") and not pd.isna(curr["RSI"]):
        if not (45 <= curr["RSI"] <= 75):
            fails.append("rsi")

    if filters.get("vwap_filter") and not pd.isna(curr["VWAP"]):
        if curr["Close"] <= curr["VWAP"]:
            fails.append("vwap")

    if filters.get("atr_filter") and not pd.isna(curr["ATR"]):
        candle_range = curr["High"] - curr["Low"]
        if candle_range < 0.5 * curr["ATR"]:
            fails.append("atr")

    if fails:
        return None

    # ── BUILD SIGNAL ──────────────────────────────────────────────────────────
    signal_time = df.index[-1].strftime("%H:%M")

    signal = {
        "Time":       signal_time,
        "EMA50":      round(curr["EMA50"],  2),
        "EMA200":     round(curr["EMA200"], 2),
        "Price":      round(curr["Close"],  2),
        "Volume":     int(curr["Volume"]),
        "VolMA20":    int(curr["VolMA20"]) if not pd.isna(curr["VolMA20"]) else 0,
        "RSI":        round(curr["RSI"],    1) if not pd.isna(curr["RSI"]) else 0,
        "VWAP":       round(curr["VWAP"],   2) if not pd.isna(curr["VWAP"]) else 0,
        "ATR":        round(curr["ATR"],    2) if not pd.isna(curr["ATR"]) else 0,
        "Signal":     "BUY 🟢",
        "Score":      0,           # filled below
    }

    signal["Score"] = compute_score(signal, curr)
    return signal


def compute_score(signal: dict, curr: pd.Series) -> int:
    """
    Score 0–100 for ranking multiple simultaneous signals.
    Components:
      Volume Spike     (0–30)
      Distance>EMA200  (0–20)
      RSI strength     (0–20)
      Candle body      (0–15)
      VWAP margin      (0–15)
    """
    score = 0

    # Volume spike
    if signal["VolMA20"] > 0:
        vol_ratio = signal["Volume"] / signal["VolMA20"]
        score += min(30, int((vol_ratio - 1) * 15))   # 2× vol → 15 pts, 3× → 30 pts

    # Distance above EMA200
    dist_pct = (signal["Price"] - signal["EMA200"]) / signal["EMA200"] * 100
    score += min(20, int(dist_pct * 4))

    # RSI (ideal: 55–65)
    rsi = signal["RSI"]
    if 55 <= rsi <= 65:
        score += 20
    elif 45 <= rsi < 55 or 65 < rsi <= 75:
        score += 10

    # Candle body
    if not pd.isna(curr["Open"]):
        body_pct = abs(curr["Close"] - curr["Open"]) / curr["Open"] * 100
        score += min(15, int(body_pct * 5))

    # VWAP margin
    if signal["VWAP"] > 0:
        vwap_margin = (signal["Price"] - signal["VWAP"]) / signal["VWAP"] * 100
        score += min(15, int(vwap_margin * 3))

    return max(0, score)


def run_scanner(data: dict, triggered_today: set, filters: dict) -> list:
    """
    data          : {symbol: raw_ohlcv_df}
    triggered_today: set of symbols already alerted today
    filters       : filter config dict

    Returns list of signal dicts (already sorted by Score desc), max 50.
    """
    signals = []

    for symbol, raw_df in data.items():
        if symbol in triggered_today:
            continue

        df = add_all_indicators(raw_df)
        sig = check_buy_signal(df, filters)

        if sig:
            sig["Stock"] = symbol
            signals.append(sig)

    # Sort by score descending
    signals.sort(key=lambda x: x["Score"], reverse=True)
    return signals[:50]
