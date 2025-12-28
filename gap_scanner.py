#!/usr/bin/env python3
"""
Gap Momentum Scalping Strategy (Gap Ups) — Scanner + Setup Finder

Implements the rules you described (simplified, testable):

A) Pre-market scan:
   - Find GAP UPS: today's open vs yesterday close (gap %)

B) "High-quality gaps" filters (daily):
   1) Ending a downtrend (recent trend down)
   2) Clearing resistance (opens above recent resistance range)
   3) Not too far above resistance (gap not "too extended")

C) Intraday execution setups within first 30 minutes (9:30–10:00 ET) using 5m bars:
   1) High-Low / Opening Range Breakout (first 5m candle narrow-range) -> buy above first candle high, stop below low
   2) Simple breakout base (2–3 small candles after push) -> buy above base high, stop below base low
   3) Retracement to rising 20MA (5m) -> buy on bounce above candle high, stop below retracement low

Outputs:
- CSV of candidates and signals with entry/stop/notes.

IMPORTANT:
- yfinance data is not true real-time and may be delayed.
- On weekends, you will see last trading day only.
"""

import sys
import math
import time
import datetime as dt
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ======================
# CONFIG (tweakable)
# ======================
UNIVERSE = "SP500"       # or "CUSTOM"
CUSTOM_TICKERS = ["NVDA", "AAPL", "TSLA"]

GAP_MIN_PCT = 3.0        # only consider gap ups >= this %
GAP_MAX_PCT = 25.0       # ignore insane gaps (optional)

DOWNTREND_LOOKBACK = 20  # days
DOWNTREND_SLOPE_MAX = 0  # slope must be negative to qualify as "downtrend ending"

RES_LOOKBACK = 20        # days resistance window
MAX_ABOVE_RES_PCT = 3.0  # open must be above resistance, but not > this % above it

VOLUME_LOOKBACK = 20
MIN_REL_VOL = 1.5        # today's volume proxy vs average (uses early session volume if available)

INTRADAY_SCAN_END = dt.time(10, 0)  # first 30 minutes of RTH
MIN_5M_BARS = 3

OUTPUT_CSV = "gap_strategy_signals.csv"


# ======================
# Utilities
# ======================
def ensure_et(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    return df.tz_convert("America/New_York")


def fetch_sp500_symbols() -> List[str]:
    """Fetch S&P500 tickers from Wikipedia; fallback if blocked."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        for tbl in tables:
            cols = [c.lower() for c in tbl.columns.astype(str)]
            if any("symbol" in c for c in cols):
                sym_col = [c for c in tbl.columns if "Symbol" in str(c)][0]
                syms = tbl[sym_col].astype(str).str.replace(".", "-", regex=False)
                syms = sorted(set(syms))
                if len(syms) >= 400:
                    return syms
        raise ValueError("No symbol column found.")
    except Exception as e:
        print(f"[WARN] SP500 fetch failed: {e}", file=sys.stderr)
        fallback = [
            "NVDA","AAPL","MSFT","AMZN","GOOGL","META","TSLA","AMD","JPM","XOM",
            "UNH","V","MA","HD","COST","LLY","AVGO","NFLX","BA","DIS"
        ]
        print(f"[INFO] Using fallback universe ({len(fallback)})", file=sys.stderr)
        return fallback


def linear_slope(y: np.ndarray) -> float:
    """Simple slope of y over index 0..n-1."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    # least squares slope
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


# ======================
# Daily filters (gap quality)
# ======================
def get_daily(ticker: str) -> Optional[pd.DataFrame]:
    df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    return df


def compute_gap_pct(daily: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Gap % = (today open - yesterday close) / yesterday close * 100
    Uses last two rows of daily candles (most recent trading day).
    """
    if daily is None or len(daily) < 2:
        return None

    y_close = float(daily["Close"].iloc[-2])
    t_open = float(daily["Open"].iloc[-1])
    if y_close <= 0:
        return None

    gap_pct = (t_open - y_close) / y_close * 100.0
    return {"y_close": y_close, "t_open": t_open, "gap_pct": gap_pct}


def is_downtrend_ending(daily: pd.DataFrame, lookback: int) -> bool:
    """
    "Ending a downtrend" proxy:
    - last N closes slope is negative
    """
    if daily is None or len(daily) < lookback + 1:
        return False
    closes = daily["Close"].astype(float).iloc[-(lookback+1):-1].values  # exclude today
    slope = linear_slope(closes)
    return slope < DOWNTREND_SLOPE_MAX


def clears_resistance(daily: pd.DataFrame, open_price: float, lookback: int) -> Dict[str, Any]:
    """
    Resistance proxy:
    - use max high of last N days (excluding today) as resistance
    - require open above that resistance, but not too far above
    """
    if daily is None or len(daily) < lookback + 1:
        return {"ok": False}

    recent = daily.iloc[-(lookback+1):-1]
    res = float(recent["High"].astype(float).max())
    if res <= 0:
        return {"ok": False}

    above_pct = (open_price - res) / res * 100.0
    ok = (open_price > res) and (above_pct <= MAX_ABOVE_RES_PCT)
    return {"ok": ok, "resistance": res, "above_res_pct": above_pct}


# ======================
# Intraday setups (first 30 min)
# ======================
def get_5m(ticker: str) -> Optional[pd.DataFrame]:
    """
    Pull 5m candles. yfinance returns recent days; we filter to RTH session.
    """
    df = yf.download(ticker, period="5d", interval="5m", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    return ensure_et(df)


def rth_first_30m(df_5m: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Get 9:30–10:00 ET bars for the most recent session day in df_5m.
    """
    if df_5m is None or df_5m.empty:
        return None
    df = df_5m.copy()
    # take last date in ET
    last_day = df.index.date[-1]
    day_df = df[df.index.date == last_day]
    if day_df.empty:
        return None

    t = day_df.index.time
    win = day_df[(t >= dt.time(9, 30)) & (t < INTRADAY_SCAN_END)]
    return win if not win.empty else None


def setup_opening_range_breakout(win: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    High-Low / ORB:
    - Use first 5m candle.
    - "Narrow range" proxy: range <= median(range of first 30m) * 0.8
    """
    if win is None or len(win) < 1:
        return None
    first = win.iloc[0]
    ranges = (win["High"] - win["Low"]).astype(float)
    med = float(ranges.median()) if len(ranges) else 0.0
    first_range = float(first["High"] - first["Low"])

    # Narrow-range gate (so you don't buy a giant first candle)
    if med > 0 and first_range <= med * 0.8:
        entry = float(first["High"])
        stop = float(first["Low"])
        return {
            "setup": "ORB (first 5m narrow)",
            "entry": entry,
            "stop": stop
        }
    return None


def setup_simple_breakout_base(win: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Simple breakout base within first 30 minutes:
    - Look for 2–3 consecutive small-range candles AFTER the first candle
    - Base high = max highs, base low = min lows
    - Entry above base high, stop below base low
    """
    if win is None or len(win) < 4:
        return None

    ranges = (win["High"] - win["Low"]).astype(float)
    med = float(ranges.median()) if len(ranges) else 0.0
    if med <= 0:
        return None

    # try bases starting from bar 1 onward
    for start in range(1, min(4, len(win)-2)):
        for length in (2, 3):
            end = start + length
            if end >= len(win):
                continue
            chunk = win.iloc[start:end]
            chunk_ranges = (chunk["High"] - chunk["Low"]).astype(float)
            # base: all candles relatively small
            if (chunk_ranges <= med * 0.8).all():
                base_high = float(chunk["High"].max())
                base_low = float(chunk["Low"].min())
                return {
                    "setup": f"Simple Breakout Base ({length} bars)",
                    "entry": base_high,
                    "stop": base_low
                }
    return None


def setup_retrace_to_20ma(win_full_5d: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Retracement to rising 20MA:
    - Compute 20MA on 5m closes over the full 5d data
    - Focus on last day's first 30m window, find a pullback candle whose low touches/near MA
    - "rising MA" proxy: last 5 MA values slope up
    Entry:
      - buy above the pullback candle high (bounce confirmation)
    Stop:
      - below pullback low
    """
    if win_full_5d is None or win_full_5d.empty:
        return None

    df = win_full_5d.copy()
    df["ma20"] = df["Close"].astype(float).rolling(20).mean()

    # Determine rising MA (last 5 points)
    ma_tail = df["ma20"].dropna().tail(5).values
    if len(ma_tail) < 5:
        return None
    if linear_slope(ma_tail) <= 0:
        return None

    # Focus on last day first 30m window
    win = rth_first_30m(df)
    if win is None or len(win) < 3:
        return None

    # Look for touch/pullback near MA20 within window (excluding first bar)
    for i in range(1, len(win)):
        c = win.iloc[i]
        ma = float(c["ma20"]) if not pd.isna(c["ma20"]) else None
        if ma is None or ma <= 0:
            continue

        low = float(c["Low"])
        high = float(c["High"])

        # "touch" condition: low within 0.2% of MA or below it slightly
        if abs(low - ma) / ma <= 0.002 or (low <= ma <= high):
            entry = high  # buy on bounce confirmation
            stop = low
            return {
                "setup": "Retracement to rising 20MA",
                "entry": entry,
                "stop": stop
            }
    return None


def compute_rel_volume(daily: pd.DataFrame) -> Optional[float]:
    """
    Relative volume proxy:
    - today's volume / avg volume last N days (excluding today if possible)
    NOTE: in early morning, today's daily volume may be incomplete.
    Still useful for backtesting on historical days.
    """
    if daily is None or len(daily) < VOLUME_LOOKBACK + 2:
        return None

    today_vol = float(daily["Volume"].iloc[-1])
    avg_vol = float(daily["Volume"].iloc[-(VOLUME_LOOKBACK+1):-1].mean())
    if avg_vol <= 0:
        return None
    return today_vol / avg_vol


# ======================
# Main scan logic
# ======================
def analyze_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    daily = get_daily(ticker)
    if daily is None:
        return None

    gap = compute_gap_pct(daily)
    if gap is None:
        return None

    gap_pct = gap["gap_pct"]
    if gap_pct < GAP_MIN_PCT or gap_pct > GAP_MAX_PCT:
        return None

    # Only gap ups for this strategy
    if gap["t_open"] <= gap["y_close"]:
        return None

    downtrend_ok = is_downtrend_ending(daily, DOWNTREND_LOOKBACK)
    res_info = clears_resistance(daily, gap["t_open"], RES_LOOKBACK)
    rel_vol = compute_rel_volume(daily)

    # Require both quality conditions (matches your description)
    if not downtrend_ok:
        return None
    if not res_info.get("ok", False):
        return None
    if rel_vol is not None and rel_vol < MIN_REL_VOL:
        return None

    # Intraday setups (first 30m)
    df5 = get_5m(ticker)
    if df5 is None:
        return None
    win = rth_first_30m(df5)
    if win is None or len(win) < MIN_5M_BARS:
        # not enough intraday yet
        return {
            "ticker": ticker,
            "signal": "NO_TRADE",
            "reason": "Not enough 5m bars in first 30m yet",
            "gap_pct": gap_pct,
            "open": gap["t_open"],
            "prev_close": gap["y_close"],
            "resistance": res_info.get("resistance"),
            "above_res_pct": res_info.get("above_res_pct"),
            "rel_vol": rel_vol
        }

    # Setup priority (your order):
    s1 = setup_opening_range_breakout(win)
    s2 = setup_simple_breakout_base(win)
    s3 = setup_retrace_to_20ma(df5)

    setup = s1 or s2 or s3
    if not setup:
        return {
            "ticker": ticker,
            "signal": "NO_TRADE",
            "reason": "Gap qualifies, but no clean setup in first 30m",
            "gap_pct": gap_pct,
            "open": gap["t_open"],
            "prev_close": gap["y_close"],
            "resistance": res_info.get("resistance"),
            "above_res_pct": res_info.get("above_res_pct"),
            "rel_vol": rel_vol
        }

    entry = float(setup["entry"])
    stop = float(setup["stop"])
    risk = entry - stop if entry > stop else None

    return {
        "ticker": ticker,
        "signal": "LONG",
        "setup": setup["setup"],
        "entry_buy_above": round(entry, 4),
        "stop_sell_below": round(stop, 4),
        "risk_per_share": round(risk, 4) if risk is not None else None,
        "gap_pct": round(gap_pct, 3),
        "open": round(gap["t_open"], 4),
        "prev_close": round(gap["y_close"], 4),
        "resistance": round(float(res_info.get("resistance")), 4),
        "above_res_pct": round(float(res_info.get("above_res_pct")), 3),
        "rel_vol": round(float(rel_vol), 3) if rel_vol is not None else None
    }


def main():
    if UNIVERSE == "SP500":
        tickers = fetch_sp500_symbols()
    else:
        tickers = CUSTOM_TICKERS

    results: List[Dict[str, Any]] = []
    total = len(tickers)

    print(f"[INFO] Universe: {UNIVERSE} | tickers={total}")
    print(f"[INFO] Gap min: {GAP_MIN_PCT}% | Downtrend lookback: {DOWNTREND_LOOKBACK} | Res lookback: {RES_LOOKBACK}")

    for i, t in enumerate(tickers, 1):
        try:
            r = analyze_ticker(t)
            if r:
                results.append(r)
                if r["signal"] == "LONG":
                    print(f"[LONG] {t} | {r['setup']} | entry>{r['entry_buy_above']} stop<{r['stop_sell_below']} gap={r['gap_pct']}%")
        except Exception as e:
            print(f"[WARN] {t} failed: {e}", file=sys.stderr)

        # tiny sleep to be nicer to Yahoo
        if i % 50 == 0:
            time.sleep(0.8)

    if not results:
        print("[INFO] No candidates found.")
        return

    df = pd.DataFrame(results)
    # show LONGs first
    df["sort_key"] = df["signal"].apply(lambda x: 0 if x == "LONG" else 1)
    df = df.sort_values(["sort_key", "gap_pct"], ascending=[True, False]).drop(columns=["sort_key"])

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Wrote {len(df)} rows to {OUTPUT_CSV}")

    # Print a small summary table
    longs = df[df["signal"] == "LONG"].copy()
    if not longs.empty:
        cols = ["ticker","gap_pct","setup","entry_buy_above","stop_sell_below","risk_per_share","rel_vol","above_res_pct"]
        print("\n=== LONG SIGNALS ===")
        print(longs[cols].head(30).to_string(index=False))
    else:
        print("\n[INFO] No LONG signals today.")


if __name__ == "__main__":
    main()
