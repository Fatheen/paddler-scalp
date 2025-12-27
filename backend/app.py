from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import time as dtime

app = FastAPI(title="NVDA Long-Only Pattern Scalp API")

# Allow frontend to call backend in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# CONFIG
# ======================
ATR_PERIOD = 14
ATR_FRAC = 0.20
MIN_FIRST15_BARS = 10
SCAN_5M_UNTIL = dtime(10, 30)


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


def _body(c) -> float:
    return abs(float(c.Close) - float(c.Open))

def _upper_wick(c) -> float:
    return float(c.High) - max(float(c.Open), float(c.Close))

def _lower_wick(c) -> float:
    return min(float(c.Open), float(c.Close)) - float(c.Low)

def is_hammer(c) -> bool:
    body = _body(c) or 1e-9
    return (_lower_wick(c) >= 2.0 * body) and (_upper_wick(c) <= 0.8 * body)

def is_bullish_engulfing(prev, curr) -> bool:
    return (
        float(prev.Close) < float(prev.Open)
        and float(curr.Close) > float(curr.Open)
        and float(curr.Open) <= float(prev.Close)
        and float(curr.Close) >= float(prev.Open)
    )

def find_long_signal_5m(intra_5m: pd.DataFrame):
    """
    Returns: (signal_time_str, signal_high, signal_low, pattern_name) or (None,...)
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
            return str(bars.index[i]), float(curr.High), float(curr.Low), "John Wick (Hammer)"
        if is_bullish_engulfing(prev, curr):
            return str(bars.index[i]), float(curr.High), float(curr.Low), "Power of Tower (Bullish Engulfing)"

    return None, None, None, None


def pick_targets(entry: float, h15: float, l15: float) -> tuple[float, float]:
    mid = (h15 + l15) / 2.0
    tp1 = mid
    tp2 = h15

    if entry >= tp1:
        tp1 = h15
    if entry >= h15 * 0.999:
        tp2 = entry + (h15 - l15) * 0.25

    return float(tp1), float(tp2)


def analyze_long_only(ticker: str):
    ticker = ticker.upper().strip()

    daily = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)
    atr = compute_atr14(daily)
    if atr is None:
        return {"ticker": ticker, "signal": "NO_TRADE", "reason": "ATR14 unavailable"}

    intra_1m = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=False)
    if intra_1m is None or intra_1m.empty:
        return {"ticker": ticker, "signal": "NO_TRADE", "reason": "No intraday 1m data"}

    intra_et = ensure_et(intra_1m)
    session_date = str(intra_et.index.date[-1])

    h15, l15 = first_15m_hl(intra_1m)
    if h15 is None or l15 is None:
        return {"ticker": ticker, "signal": "NO_TRADE", "session_date_et": session_date,
                "reason": "Not enough 1m bars (run after 9:45 ET)"}

    range15 = float(h15 - l15)
    thresh = float(atr * ATR_FRAC)
    mid = (h15 + l15) / 2.0

    if range15 < thresh:
        return {
            "ticker": ticker,
            "session_date_et": session_date,
            "signal": "NO_TRADE",
            "reason": "Not a manipulation candle (range < 20% ATR)",
            "atr14": round(atr, 4),
            "h15": round(h15, 4),
            "l15": round(l15, 4),
            "mid": round(mid, 4),
            "range15": round(range15, 4),
            "threshold": round(thresh, 4),
        }

    intra_5m = yf.download(ticker, period="1d", interval="5m", progress=False, auto_adjust=False)
    sig_time, sig_high, sig_low, pattern = find_long_signal_5m(intra_5m)
    if sig_time is None:
        return {
            "ticker": ticker,
            "session_date_et": session_date,
            "signal": "NO_TRADE",
            "reason": "Manipulation yes, but no LONG reversal pattern found",
            "atr14": round(atr, 4),
            "h15": round(h15, 4),
            "l15": round(l15, 4),
            "mid": round(mid, 4),
            "range15": round(range15, 4),
            "threshold": round(thresh, 4),
        }

    entry = float(sig_high)   # buy break above signal high
    stop = float(sig_low)     # stop below signal low
    tp1, tp2 = pick_targets(entry, h15, l15)

    risk = entry - stop
    r_tp1 = (tp1 - entry) / risk if risk > 0 else None
    r_tp2 = (tp2 - entry) / risk if risk > 0 else None

    return {
        "ticker": ticker,
        "session_date_et": session_date,
        "signal": "LONG",
        "pattern": pattern,
        "signal_time_et": sig_time,
        "atr14": round(atr, 4),
        "h15": round(h15, 4),
        "l15": round(l15, 4),
        "mid": round(mid, 4),
        "range15": round(range15, 4),
        "threshold": round(thresh, 4),
        "entry_buy_above": round(entry, 4),
        "stop_sell_below": round(stop, 4),
        "take_profit_1": round(tp1, 4),
        "take_profit_2": round(tp2, 4),
        "risk_per_share": round(risk, 4) if risk > 0 else None,
        "r_to_tp1": round(r_tp1, 3) if r_tp1 is not None else None,
        "r_to_tp2": round(r_tp2, 3) if r_tp2 is not None else None,
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/analyze")
def analyze(ticker: str = "NVDA"):
    return analyze_long_only(ticker)
