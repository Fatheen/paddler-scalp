#!/usr/bin/env python3
"""
NVDA Long-Only Pattern / Paddler Scalp Plan (BUY low, SELL high)

What it does (LONG ONLY):
1) Check manipulation candle (first 15m: 9:30–9:45 ET)
   - Manipulation if range >= 20% of Daily ATR(14)
2) Find a LONG reversal signal on 5m candles after 9:45:
   - John Wick (Hammer)
   - Power of Tower (Bullish Engulfing)
3) If a LONG signal is found:
   - ENTRY: buy when price breaks ABOVE the signal candle HIGH
   - STOP: below signal candle LOW (simple)
   - TAKE PROFIT:
       TP1 = midpoint of the opening 15m range
       TP2 = opening 15m HIGH (H15)
   - If entry is already above TP1/TP2, it adjusts targets automatically.
4) Otherwise: NO TRADE

Notes:
- On weekends/holidays, yfinance returns the last trading session.
- Run after ~9:55 AM ET so the first two 5m candles exist.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import time as dtime


# ======================
# CONFIG
# ======================
TICKER = "GOOG"
ATR_PERIOD = 14
ATR_FRAC = 0.20

MIN_FIRST15_BARS = 10
SCAN_5M_UNTIL = dtime(10, 30)   # scan for the first LONG signal until 10:30 ET


# ======================
# Helpers
# ======================
def ensure_et(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    return df.tz_convert("America/New_York")


def compute_atr14(daily: pd.DataFrame) -> float | None:
    if daily is None or daily.empty or len(daily) < ATR_PERIOD + 1:
        return None

    h = daily["High"].astype(float)
    l = daily["Low"].astype(float)
    c = daily["Close"].astype(float)

    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean().dropna()
    if atr.empty:
        return None
    val = float(atr.iloc[-1])
    return val if np.isfinite(val) and val > 0 else None


def first_15m_hl(intra_1m: pd.DataFrame) -> tuple[float | None, float | None]:
    if intra_1m is None or intra_1m.empty:
        return None, None

    intra = ensure_et(intra_1m)
    t = intra.index.time
    first15 = intra[(t >= dtime(9, 30)) & (t < dtime(9, 45))]

    if len(first15) < MIN_FIRST15_BARS:
        return None, None

    return float(first15["High"].max()), float(first15["Low"].min())


# ----------------------
# Candle patterns (LONG only)
# ----------------------
def _body(c) -> float:
    return abs(float(c.Close) - float(c.Open))

def _upper_wick(c) -> float:
    return float(c.High) - max(float(c.Open), float(c.Close))

def _lower_wick(c) -> float:
    return min(float(c.Open), float(c.Close)) - float(c.Low)

def is_hammer(c) -> bool:
    # Lower wick big, upper wick small-ish
    body = _body(c) or 1e-9
    return (_lower_wick(c) >= 2.0 * body) and (_upper_wick(c) <= 0.8 * body)

def is_bullish_engulfing(prev, curr) -> bool:
    return (
        float(prev.Close) < float(prev.Open)   # prev red
        and float(curr.Close) > float(curr.Open) # curr green
        and float(curr.Open) <= float(prev.Close)
        and float(curr.Close) >= float(prev.Open)
    )


def find_long_signal_5m(intra_5m: pd.DataFrame):
    """
    Returns:
      (signal_time, signal_high, signal_low, pattern_name)
      or (None, None, None, None)
    """
    if intra_5m is None or intra_5m.empty:
        return None, None, None, None

    intra_5m = ensure_et(intra_5m)
    t = intra_5m.index.time
    bars = intra_5m[(t >= dtime(9, 45)) & (t < SCAN_5M_UNTIL)]

    if len(bars) < 2:
        return None, None, None, None

    for i in range(1, len(bars)):
        prev = bars.iloc[i - 1]
        curr = bars.iloc[i]

        if is_hammer(curr):
            return bars.index[i], float(curr.High), float(curr.Low), "John Wick (Hammer)"
        if is_bullish_engulfing(prev, curr):
            return bars.index[i], float(curr.High), float(curr.Low), "Power of Tower (Bullish Engulfing)"

    return None, None, None, None


def pick_targets(entry: float, h15: float, l15: float) -> tuple[float, float]:
    """
    Simple 'buy low sell high' targets:
      TP1 = midpoint of opening range
      TP2 = H15 (top of the box)
    If entry is already above TP1, TP1 becomes H15.
    If entry is already near/above H15, TP2 becomes entry + (H15-L15)*0.25 as a small runner.
    """
    mid = (h15 + l15) / 2.0

    tp1 = mid
    tp2 = h15

    if entry >= tp1:
        tp1 = h15

    if entry >= h15 * 0.999:  # basically at/above H15 already
        tp2 = entry + (h15 - l15) * 0.25  # small extension

    return float(tp1), float(tp2)


# ======================
# MAIN
# ======================
def main():
    print(f"\n=== {TICKER} LONG-ONLY Plan (Pattern / Paddler Scalp) ===\n")

    # Daily ATR
    daily = yf.download(TICKER, period="6mo", interval="1d", progress=False, auto_adjust=False)
    atr = compute_atr14(daily)
    if atr is None:
        print("NO TRADE: ATR14 unavailable.")
        return

    # Intraday 1m for first 15m box
    intra_1m = yf.download(TICKER, period="1d", interval="1m", progress=False, auto_adjust=False)
    if intra_1m is None or intra_1m.empty:
        print("NO TRADE: No intraday 1m data.")
        return

    intra_et = ensure_et(intra_1m)
    session_date = str(intra_et.index.date[-1])

    h15, l15 = first_15m_hl(intra_1m)
    if h15 is None or l15 is None:
        print("NO TRADE: Not enough 1m data to build the first 15 minutes (run after 9:45 ET).")
        return

    range15 = float(h15 - l15)
    thresh = float(atr * ATR_FRAC)
    mid = (h15 + l15) / 2.0

    print(f"Session date (ET): {session_date}")
    print(f"ATR14: {atr:.2f}")
    print(f"Opening box: L15={l15:.2f} | MID={mid:.2f} | H15={h15:.2f}")
    print(f"15m Range: {range15:.2f} | Manip threshold (20% ATR): {thresh:.2f}")

    if range15 < thresh:
        print("\nNO TRADE: First 15m candle is NOT manipulation by the 20% ATR rule.\n")
        return

    print("\n✅ Manipulation candle detected. Looking for LONG reversal on 5m...")

    # Find LONG signal on 5m
    intra_5m = yf.download(TICKER, period="1d", interval="5m", progress=False, auto_adjust=False)
    sig_time, sig_high, sig_low, pattern = find_long_signal_5m(intra_5m)

    if sig_time is None:
        print("\nNO TRADE: No LONG reversal candle (hammer/engulfing) found in the scan window.\n")
        return

    # Entry / Stop / Targets
    entry = float(sig_high)            # buy when price breaks above signal candle high
    stop = float(sig_low)              # simple: stop below signal candle low
    tp1, tp2 = pick_targets(entry, h15, l15)

    risk = entry - stop
    r1 = (tp1 - entry) / risk if risk > 0 else None
    r2 = (tp2 - entry) / risk if risk > 0 else None

    print("\n====================")
    print("✅ LONG SETUP FOUND")
    print("====================")
    print(f"Pattern: {pattern}")
    print(f"Signal candle time (ET): {sig_time}")

    print("\n--- WHEN TO ENTER (BUY) ---")
    print(f"BUY when price breaks ABOVE: {entry:.2f}")

    print("\n--- WHEN TO EXIT (SELL) ---")
    print(f"STOP (sell if wrong): below {stop:.2f}")
    print(f"TAKE PROFIT 1 (sell some): {tp1:.2f}  (often the midpoint/box reclaim)")
    print(f"TAKE PROFIT 2 (sell rest): {tp2:.2f}  (often the top of the opening box)")

    if r1 is not None and r2 is not None:
        print("\n--- QUICK RISK/REWARD ---")
        print(f"Risk per share: {risk:.2f}")
        print(f"R to TP1: {r1:.2f}R | R to TP2: {r2:.2f}R")

    print("\nSimple execution idea:")
    print("- Buy the break above entry.")
    print("- If it hits TP1, sell part.")
    print("- If it hits TP2, sell the rest.")
    print("- If it hits STOP first, exit immediately.\n")


if __name__ == "__main__":
    main()
