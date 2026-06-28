"""
NSE Data Fetcher — uses yfinance (free, no API key needed)
Fetches 2-minute OHLCV data for NSE stocks (symbol.NS format)
Supports: live data (market open) + previous session data (market closed)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time

# ─── TOP 50 NSE STOCKS (Nifty 50) ───────────────────────────────────────────
NIFTY50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
    "SUNPHARMA", "ULTRACEMCO", "BAJFINANCE", "WIPRO", "HCLTECH",
    "POWERGRID", "NESTLEIND", "NTPC", "TATAMOTORS", "TATASTEEL",
    "JSWSTEEL", "M&M", "TECHM", "ADANIENT", "ADANIPORTS",
    "ONGC", "COALINDIA", "BAJAJFINSV", "HDFCLIFE", "SBILIFE",
    "DIVISLAB", "CIPLA", "DRREDDY", "APOLLOHOSP", "BRITANNIA",
    "EICHERMOT", "HEROMOTOCO", "BPCL", "GRASIM", "HINDALCO",
    "INDUSINDBK", "SHRIRAMFIN", "TATACONSUM", "UPL", "VEDL"
]

NIFTY100_EXTRA = [
    "BAJAJ-AUTO", "BERGEPAINT", "BIOCON", "BOSCHLTD", "CHOLAFIN",
    "COLPAL", "DABUR", "DLF", "ESCORTS", "FEDERALBNK",
    "GAIL", "GODREJCP", "HAVELLS", "IDFCFIRSTB", "INDHOTEL",
    "INDUSTOWER", "IRCTC", "JUBLFOOD", "LICHSGFIN", "LUPIN",
    "MARICO", "MOTHERSON", "MUTHOOTFIN", "NAUKRI", "PAGEIND",
    "PERSISTENT", "PIIND", "PNB", "RECLTD", "SAIL",
    "SIEMENS", "SRF", "TORNTPHARM", "TRENT", "TVSMOTOR",
    "VOLTAS", "WHIRLPOOL", "ZYDUSLIFE", "PIDILITIND", "ALKEM"
]

IST = pytz.timezone("Asia/Kolkata")


def get_symbols(use_nifty100: bool = False) -> list:
    symbols = NIFTY50_SYMBOLS.copy()
    if use_nifty100:
        symbols += NIFTY100_EXTRA
    return symbols


def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def get_last_trading_day():
    """Returns last trading weekday as a date object."""
    now = datetime.now(IST)
    day = now.date()
    offset = 1
    while True:
        candidate = day - timedelta(days=offset)
        if candidate.weekday() < 5:
            return candidate
        offset += 1


def get_prev_session_period():
    """Returns (start_str, end_str, label) for the last trading day."""
    last_day = get_last_trading_day()
    start = last_day.strftime("%Y-%m-%d")
    end   = (last_day + timedelta(days=1)).strftime("%Y-%m-%d")
    label = last_day.strftime("%A, %d %b %Y")
    return start, end, label


def _safe_fetch(symbol: str, period: str = "5d", interval: str = "2m",
                retries: int = 3, delay: float = 2.0) -> pd.DataFrame:
    """
    Robust single-symbol fetch with retries and proper error propagation.
    Raises RuntimeError with clear message if all retries fail.
    """
    last_err = None
    for attempt in range(retries):
        try:
            tk = yf.Ticker(f"{symbol}.NS")
            df = tk.history(period=period, interval=interval,
                            auto_adjust=True, raise_errors=False)
            if df is not None and not df.empty:
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                df.index = df.index.tz_convert(IST)
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.dropna(inplace=True)
                return df
            # Empty but no exception — might be a holiday / delisted
            return pd.DataFrame()
        except Exception as e:
            last_err = str(e)
            if attempt < retries - 1:
                time.sleep(delay)

    raise RuntimeError(f"yfinance fetch failed for {symbol}: {last_err}")


def fetch_batch_live(symbols: list) -> dict:
    """
    Fetch TODAY's 2-minute data (period='1d') for all symbols.
    Returns {symbol: df}. Failed symbols are silently skipped.
    """
    results = {}
    test_error = None

    for sym in symbols:
        try:
            df = _safe_fetch(sym, period="1d", interval="2m", retries=2, delay=1.0)
            if not df.empty and len(df) >= 5:
                results[sym] = df
        except RuntimeError as e:
            if test_error is None:
                test_error = str(e)
            continue

    # If NOTHING came back, surface the error so caller can show it
    if not results and test_error:
        raise ConnectionError(
            f"Could not fetch live data from Yahoo Finance.\n\n"
            f"Possible reasons:\n"
            f"• No internet connection\n"
            f"• Yahoo Finance is temporarily down\n"
            f"• yfinance rate limit hit (wait 2–3 minutes)\n\n"
            f"Technical detail: {test_error}"
        )
    return results


def fetch_batch_prev_session(symbols: list, progress_callback=None) -> dict:
    """
    Fetch PREVIOUS SESSION 2-min data.
    Strategy: period='5d' → filter to last trading day's candles.
    progress_callback(i, total, symbol) called after each symbol if provided.
    Returns {symbol: df}. Raises ConnectionError if nothing fetched at all.
    """
    last_day = get_last_trading_day()
    results  = {}
    test_error = None
    total = len(symbols)

    for i, sym in enumerate(symbols):
        try:
            df = _safe_fetch(sym, period="5d", interval="2m", retries=3, delay=2.0)

            if not df.empty:
                # Filter to last trading day + market hours only
                df = df[df.index.date == last_day]
                df = df.between_time("09:15", "15:30")
                df.dropna(inplace=True)
                if len(df) >= 10:
                    results[sym] = df
        except RuntimeError as e:
            if test_error is None:
                test_error = str(e)
        except Exception as e:
            if test_error is None:
                test_error = str(e)

        if progress_callback:
            progress_callback(i + 1, total, sym)

    if not results:
        hint = ""
        if test_error and "403" in test_error:
            hint = "Yahoo Finance returned a 403 error — this usually means a temporary block. Wait 2–3 minutes and try again."
        elif test_error and "NoneType" in test_error:
            hint = "Yahoo Finance returned empty data. This can happen on weekends or holidays when 2-min data is unavailable."
        else:
            hint = f"Technical detail: {test_error or 'Unknown error'}"

        raise ConnectionError(
            f"Could not load data for **{last_day.strftime('%A, %d %b %Y')}**.\n\n"
            f"{hint}\n\n"
            f"**What to do:**\n"
            f"- Wait 2–3 minutes, then click **Force Scan Now**\n"
            f"- Try turning off some filters to reduce signal threshold\n"
            f"- Check your internet connection"
        )

    return results


# Legacy wrapper
def fetch_batch(symbols: list, period: str = "1d", interval: str = "2m",
                start: str = None, end: str = None) -> dict:
    if start and end:
        return fetch_batch_prev_session(symbols)
    return fetch_batch_live(symbols)


def get_current_ist_time() -> str:
    return datetime.now(IST).strftime("%H:%M:%S")


def get_today_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")
