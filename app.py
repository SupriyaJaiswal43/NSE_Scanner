"""
NSE Intraday EMA Breakout Scanner
Streamlit App — auto-refreshes every 2 minutes during market hours
Supports: Live mode (market open) + Previous Session mode (market closed)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import os
import time
from io import BytesIO

from data_fetcher import (
    get_symbols, fetch_batch,
    is_market_open, get_current_ist_time, get_today_ist,
    get_last_trading_day, get_prev_session_period
)
from scanner import run_scanner, DEFAULT_FILTERS
from alerts import trigger_alerts

IST = pytz.timezone("Asia/Kolkata")
HISTORY_FILE = "signals_history.csv"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE EMA Breakout Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 5px 0 0; opacity: 0.8; font-size: 0.95rem; }

    .metric-card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px 20px;
        text-align: center;
        color: white;
    }
    .metric-val  { font-size: 2rem; font-weight: bold; color: #00ff88; }
    .metric-label{ font-size: 0.8rem; opacity: 0.7; text-transform: uppercase; }

    .prev-session-banner {
        background: linear-gradient(90deg, #2d1b69, #11998e);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        margin-bottom: 16px;
        font-size: 0.95rem;
    }
    .prev-session-banner strong { font-size: 1.05rem; }

    .status-open   { color: #00ff88; font-weight: bold; }
    .status-closed { color: #ff4444; font-weight: bold; }
    .status-prev   { color: #f0a500; font-weight: bold; }

    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }

    footer { display: none; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    today = get_today_ist()

    if "last_scan_date" not in st.session_state:
        st.session_state.last_scan_date = today

    # Reset daily on new market day
    if st.session_state.last_scan_date != today:
        st.session_state.triggered_today    = set()
        st.session_state.active_signals     = []
        st.session_state.prev_signals       = []
        st.session_state.last_scan_date     = today
        st.session_state.prev_session_done  = False

    defaults = {
        "triggered_today":   set(),
        "active_signals":    [],
        "prev_signals":      [],        # signals from previous trading session
        "scan_count":        0,
        "last_scan_time":    "—",
        "prev_session_done": False,     # True once prev-session scan is complete
        "prev_session_label": "",       # e.g. "Friday, 27 Jun 2025"
        "history":           load_history(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=["Date","Session","Time","Stock","Price","EMA50","EMA200","RSI","Score","Signal"])


def save_to_history(signals: list, session_label: str = "Live"):
    if not signals:
        return
    today = get_today_ist()
    rows = []
    for s in signals:
        rows.append({
            "Date":    today,
            "Session": session_label,
            "Time":    s["Time"],
            "Stock":   s["Stock"],
            "Price":   s["Price"],
            "EMA50":   s["EMA50"],
            "EMA200":  s["EMA200"],
            "RSI":     s["RSI"],
            "Score":   s["Score"],
            "Signal":  "BUY"
        })
    new_df   = pd.DataFrame(rows)
    existing = st.session_state.history
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.drop_duplicates(subset=["Date", "Session", "Stock"], keep="first", inplace=True)
    combined.to_csv(HISTORY_FILE, index=False)
    st.session_state.history = combined


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(market_open: bool):
    st.sidebar.markdown("## ⚙️ Scanner Settings")

    use_100 = st.sidebar.toggle("Use Nifty 100 stocks", value=False)
    symbols = get_symbols(use_nifty100=use_100)
    st.sidebar.caption(f"Scanning **{len(symbols)}** stocks")

    # ── Previous Session Option (shown when market is closed) ────────────────
    show_prev = False
    if not market_open:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📅 Previous Session")
        _, _, prev_label = get_prev_session_period()
        show_prev = st.sidebar.toggle(
            f"Show last session data\n({prev_label})",
            value=True,
            key="show_prev_toggle"
        )
        if show_prev:
            st.sidebar.info(
                "📊 Scanner will run on the **last trading session's** "
                "2-minute candles and show all EMA200 crossover signals from that day."
            )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Optional Filters")

    filters = {}
    filters["volume_filter"]      = st.sidebar.toggle("Volume > 20-period MA",   value=True)
    filters["ema50_slope"]        = st.sidebar.toggle("EMA50 sloping upward",     value=True)
    filters["ema200_flat_rising"] = st.sidebar.toggle("EMA200 flat or rising",    value=True)
    filters["green_candle"]       = st.sidebar.toggle("Green breakout candle",    value=True)
    filters["rsi_filter"]         = st.sidebar.toggle("RSI between 45–75",        value=True)
    filters["vwap_filter"]        = st.sidebar.toggle("Price above VWAP",         value=True)
    filters["atr_filter"]         = st.sidebar.toggle("Candle range ≥ 0.5× ATR", value=True)
    filters["min_body_pct"]       = st.sidebar.slider("Min candle body (%)", 0.0, 1.0, 0.3, 0.05)

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Force Scan Now"):
        st.session_state["force_scan"] = True

    if st.sidebar.button("🗑️ Reset All Signals"):
        st.session_state.triggered_today   = set()
        st.session_state.active_signals    = []
        st.session_state.prev_signals      = []
        st.session_state.prev_session_done = False
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Market Hours:** 9:15 AM – 3:30 PM IST")
    if market_open:
        st.sidebar.markdown("**Auto-refresh:** Every 2 minutes")

    return symbols, filters, show_prev


# ─────────────────────────────────────────────────────────────────────────────
# SCAN FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def do_live_scan(symbols: list, filters: dict) -> int:
    """Scan live (today's) 2-min data."""
    from data_fetcher import fetch_batch_live
    try:
        with st.spinner(f"📡 Scanning {len(symbols)} stocks (live data)…"):
            data = fetch_batch_live(symbols)
    except ConnectionError as ce:
        st.error(f"⚠️ Live data fetch failed: {ce}")
        return 0

    new_signals = run_scanner(data, st.session_state.triggered_today, filters)

    if new_signals:
        trigger_alerts(new_signals)
        for sig in new_signals:
            st.session_state.triggered_today.add(sig["Stock"])
        st.session_state.active_signals = new_signals + st.session_state.active_signals
        save_to_history(new_signals, session_label="Live")

    st.session_state.scan_count    += 1
    st.session_state.last_scan_time = get_current_ist_time()
    return len(new_signals)


def do_prev_session_scan(symbols: list, filters: dict):
    """
    Scan the FULL previous trading session candle-by-candle.
    Uses period='5d' then filters to last trading day — more reliable than start/end params.
    """
    from data_fetcher import fetch_batch_prev_session
    from indicators import add_all_indicators
    from scanner import check_buy_signal

    start, end, label = get_prev_session_period()
    st.session_state.prev_session_label = label

    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0, text=f"⏳ Connecting to Yahoo Finance…")

    data = {}
    try:
        def on_progress(i, total, sym):
            pct = int(i / total * 50)
            progress_bar.progress(pct, text=f"📥 Fetching {sym}… ({i}/{total})")

        data = fetch_batch_prev_session(symbols, progress_callback=on_progress)

    except ConnectionError as ce:
        progress_placeholder.empty()
        st.session_state.prev_session_done = False
        # Show nicely formatted error with retry button
        with st.container(border=True):
            st.error(f"⚠️ **Data Fetch Failed — {label}**")
            st.markdown(str(ce))
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🔄 Retry Now", type="primary"):
                    st.session_state["force_scan"] = True
                    st.rerun()
        return []

    progress_bar.progress(50, text="🔍 Scanning candles for EMA200 crossovers…")

    found = []
    seen  = set()
    total = len(data)

    for idx, (symbol, raw_df) in enumerate(data.items()):
        if symbol in seen:
            continue
        try:
            df = add_all_indicators(raw_df)
            # Slide through candles (need at least 5 to evaluate)
            for i in range(5, len(df)):
                window = df.iloc[:i+1]
                sig = check_buy_signal(window, filters)
                if sig:
                    sig["Stock"]   = symbol
                    sig["Session"] = f"Prev ({label})"
                    found.append(sig)
                    seen.add(symbol)
                    break
        except Exception:
            continue

        # Update progress
        pct = int(50 + (idx + 1) / total * 50)
        progress_bar.progress(pct, text=f"🔍 Scanning {symbol}… ({idx+1}/{total})")

    progress_placeholder.empty()

    found.sort(key=lambda x: x["Score"], reverse=True)
    found = found[:50]

    st.session_state.prev_signals      = found
    st.session_state.prev_session_done = True
    st.session_state.last_scan_time    = get_current_ist_time()
    st.session_state.scan_count       += 1

    if found:
        save_to_history(found, session_label=f"Prev ({label})")

    return found


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def to_excel(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Signals")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# RENDER TABLES
# ─────────────────────────────────────────────────────────────────────────────
def render_signals_table(signals: list, label: str = ""):
    if not signals:
        st.info(f"No BUY signals found{' for ' + label if label else ''}. 👀")
        return

    cols = ["Time", "Stock", "Price", "EMA50", "EMA200", "RSI", "Score", "Signal"]
    # Add Session col if present
    if "Session" in signals[0]:
        cols = ["Session"] + cols

    df = pd.DataFrame(signals)
    df = df[[c for c in cols if c in df.columns]]

    st.dataframe(
        df.style
          .background_gradient(subset=["Score"], cmap="Greens")
          .format({"Price": "₹{:.2f}", "EMA50": "₹{:.2f}", "EMA200": "₹{:.2f}", "RSI": "{:.1f}"}),
        use_container_width=True,
        height=min(600, 50 + len(df) * 40)
    )

    excel_data = to_excel(df)
    st.download_button(
        label="📥 Export to Excel",
        data=excel_data,
        file_name=f"nse_signals_{get_today_ist()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def render_history():
    hist = st.session_state.history
    if hist.empty:
        st.info("No signal history yet.")
        return
    st.dataframe(hist.sort_values(["Date","Time"], ascending=False),
                 use_container_width=True, height=400)
    excel_data = to_excel(hist)
    st.download_button(
        "📥 Download Full History",
        data=excel_data,
        file_name="nse_signal_history.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    init_state()

    market_open = is_market_open()

    # Auto-refresh only when market is open
    if market_open:
        st.markdown('<meta http-equiv="refresh" content="120">', unsafe_allow_html=True)

    # Track timed refresh
    now_ts = time.time()
    if "last_refresh_ts" not in st.session_state:
        st.session_state.last_refresh_ts = now_ts
        refresh_count = 0
    else:
        elapsed = now_ts - st.session_state.last_refresh_ts
        refresh_count = 1 if elapsed >= 110 else 0
        if refresh_count:
            st.session_state.last_refresh_ts = now_ts

    # ── Header ───────────────────────────────────────────────────────────────
    if market_open:
        status_str = '<span class="status-open">● MARKET OPEN</span>'
    else:
        _, _, prev_label = get_prev_session_period()
        status_str = f'<span class="status-closed">● MARKET CLOSED</span> &nbsp;|&nbsp; <span class="status-prev">Last Session: {prev_label}</span>'

    st.markdown(f"""
    <div class="main-header">
      <h1>📈 NSE Intraday EMA Breakout Scanner</h1>
      <p>EMA 50 / EMA 200 Crossover Strategy &nbsp;|&nbsp; 2-Min Timeframe &nbsp;|&nbsp; {status_str}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    symbols, filters, show_prev = render_sidebar(market_open)

    # ── Metric row ────────────────────────────────────────────────────────────
    total_signals = len(st.session_state.active_signals) + len(st.session_state.prev_signals)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{total_signals}</div>
            <div class="metric-label">Signals Found</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{st.session_state.scan_count}</div>
            <div class="metric-label">Total Scans</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{st.session_state.last_scan_time}</div>
            <div class="metric-label">Last Scan (IST)</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{len(symbols)}</div>
            <div class="metric-label">Stocks Monitored</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scan trigger ─────────────────────────────────────────────────────────
    force = st.session_state.pop("force_scan", False)

    if market_open:
        # ── LIVE MODE ────────────────────────────────────────────────────────
        if refresh_count > 0 or force or st.session_state.scan_count == 0:
            new_count = do_live_scan(symbols, filters)
            if new_count > 0:
                st.toast(f"🟢 {new_count} new BUY signal(s) found!", icon="📈")

    else:
        # ── MARKET CLOSED MODE ───────────────────────────────────────────────
        # Weekend warning
        from datetime import datetime as dt
        today_wd = dt.now(IST).weekday()
        if today_wd >= 5:  # Sat=5, Sun=6
            day_name = "Saturday" if today_wd == 5 else "Sunday"
            st.info(
                f"📅 **Today is {day_name}.** Yahoo Finance sometimes has trouble serving "
                f"intraday 2-min data over the weekend. If the scan fails, it will work normally "
                f"on Monday 9:15 AM onwards. You can still try — click **Force Scan Now** in the sidebar."
            )

        if show_prev:
            # Run previous session scan once (or on force)
            if not st.session_state.prev_session_done or force:
                prev_sigs = do_prev_session_scan(symbols, filters)
                if prev_sigs:
                    st.toast(f"📊 {len(prev_sigs)} signal(s) found in previous session!", icon="📅")
                elif st.session_state.prev_session_done:
                    st.toast("No crossover signals found in previous session.", icon="ℹ️")
        else:
            st.info("⏰ Market is closed. Enable **Show last session data** in the sidebar to review previous session signals, or use **Force Scan Now** to test.")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Signals",
        f"📅 Previous Session ({st.session_state.prev_session_label or '—'})",
        "📜 History",
        "ℹ️ Strategy"
    ])

    with tab1:
        if market_open:
            if st.session_state.active_signals:
                st.success(f"✅ **{len(st.session_state.active_signals)} BUY signal(s)** found today (live)")
        else:
            st.warning("⏰ Market is currently closed. Live signals will appear here during trading hours (9:15 AM – 3:30 PM IST).")
        render_signals_table(st.session_state.active_signals)

    with tab2:
        if st.session_state.prev_signals:
            _, _, lbl = get_prev_session_period()
            st.markdown(f"""
            <div class="prev-session-banner">
              📅 <strong>Previous Session Replay — {st.session_state.prev_session_label}</strong><br>
              Showing EMA200 crossover signals from the last trading day's 2-minute data.
              These are <em>historical signals</em> for review and analysis only.
            </div>
            """, unsafe_allow_html=True)
            st.success(f"✅ **{len(st.session_state.prev_signals)} signal(s)** found in last session")
            render_signals_table(st.session_state.prev_signals, label=st.session_state.prev_session_label)
        elif not show_prev:
            st.info("Enable **Show last session data** toggle in the sidebar to load previous session signals.")
        elif st.session_state.prev_session_done:
            st.info("No EMA200 crossover signals were found in the previous trading session with current filter settings. Try relaxing the filters.")
        else:
            st.info("Previous session data not loaded yet. Toggle the option in the sidebar.")

    with tab3:
        render_history()

    with tab4:
        render_strategy_info(filters)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** This scanner is for educational purposes only. "
        "It does not constitute financial advice. Always do your own research before trading."
    )


def render_strategy_info(filters: dict):
    st.markdown("### 📋 Strategy Details")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
**Core Buy Conditions (ALL must be true)**

1. ✅ **Price > EMA 50** — Stock is in short-term uptrend
2. ✅ **EMA 200 Crossover** — Previous candle Close ≤ EMA200, Current candle Close > EMA200
3. ✅ **Once per day per stock** — No repeat signals

**Indicators Used**
- 🟠 EMA 50 (momentum)
- 🟢 EMA 200 (trend)
- RSI 14 (momentum strength)
- VWAP (intraday fair value)
- ATR 14 (volatility)
- Volume MA 20 (participation)

**Modes**
- 🟢 **Live Mode** — Real-time scan every 2 minutes (market hours only)
- 📅 **Previous Session Mode** — Replays last trading day's full session candle-by-candle to find all crossovers that occurred
        """)

    with col2:
        st.markdown("**Active Filters**")
        filter_labels = {
            "volume_filter":      "Volume > 20-period MA",
            "ema50_slope":        "EMA50 sloping upward",
            "ema200_flat_rising": "EMA200 flat or rising",
            "green_candle":       "Green breakout candle",
            "rsi_filter":         "RSI between 45–75",
            "vwap_filter":        "Price above VWAP",
            "atr_filter":         "Candle range ≥ 0.5× ATR",
        }
        for k, label in filter_labels.items():
            icon = "✅" if filters.get(k) else "❌"
            st.markdown(f"{icon} {label}")
        st.markdown(f"📏 Min candle body: **{filters.get('min_body_pct', 0):.1f}%**")

        st.markdown("""
**Scoring System (0–100)**
| Component | Max Points |
|-----------|-----------|
| Volume spike | 30 |
| Distance above EMA200 | 20 |
| RSI strength (55–65) | 20 |
| Candle body size | 15 |
| VWAP margin | 15 |
        """)


if __name__ == "__main__":
    main()