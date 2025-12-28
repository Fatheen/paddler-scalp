from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
import pytz
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

# ======================
# Gap Momentum Strategy (Single Ticker)
# ======================

def _linear_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm = x.mean()
    ym = y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def _market_closed_banner() -> dict:
    et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    # Weekend => closed
    if now.weekday() >= 5:
        return {"market_open": False, "note": "Market closed (weekend) — results use last trading day data."}
    # Before open
    if now.time() < dtime(9, 30):
        return {"market_open": False, "note": "Market not open yet — results may use last available data."}
    return {"market_open": True, "note": ""}


def gap_analyze_long_only(
    ticker: str,
    gap_min_pct: float = 3.0,
    downtrend_lookback: int = 20,
    res_lookback: int = 20,
    max_above_res_pct: float = 8.0,   # loosened so you actually see candidates
    min_rel_vol: float = 0.0,         # rel vol off by default (yfinance daily vol is weird early)
):
    ticker = ticker.upper().strip()

    daily = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)
    if daily is None or daily.empty or len(daily) < 3:
        out = {"ticker": ticker, "signal": "NO_TRADE", "reason": "Not enough daily data"}
        out.update(_market_closed_banner())
        return out

    # gap % using last two sessions
    y_close = float(daily["Close"].iloc[-2])
    t_open = float(daily["Open"].iloc[-1])
    if y_close <= 0:
        out = {"ticker": ticker, "signal": "NO_TRADE", "reason": "Invalid prices"}
        out.update(_market_closed_banner())
        return out

    gap_pct = (t_open - y_close) / y_close * 100.0
    if gap_pct < gap_min_pct:
        out = {"ticker": ticker, "signal": "NO_TRADE", "reason": f"Gap too small ({gap_pct:.2f}%)", "gap_pct": round(gap_pct, 3)}
        out.update(_market_closed_banner())
        return out

    if t_open <= y_close:
        out = {"ticker": ticker, "signal": "NO_TRADE", "reason": "Not a gap up", "gap_pct": round(gap_pct, 3)}
        out.update(_market_closed_banner())
        return out

    # Downtrend ending proxy: negative slope of closes over last N days (excluding today)
    if len(daily) < downtrend_lookback + 2:
        out = {"ticker": ticker, "signal": "NO_TRADE", "reason": "Not enough history for downtrend check", "gap_pct": round(gap_pct, 3)}
        out.update(_market_closed_banner())
        return out

    closes = daily["Close"].astype(float).iloc[-(downtrend_lookback+1):-1].values
    slope = _linear_slope(closes)
    if slope >= 0:
        out = {"ticker": ticker, "signal": "NO_TRADE", "reason": "Not ending a downtrend (slope not negative)", "gap_pct": round(gap_pct, 3)}
        out.update(_market_closed_banner())
        return out

    # Resistance proxy: max high of last N days (excluding today)
    recent = daily.iloc[-(res_lookback+1):-1]
    res = float(recent["High"].astype(float).max())
    above_pct = (t_open - res) / res * 100.0 if res > 0 else None

    if not (t_open > res):
        out = {"ticker": ticker, "signal": "NO_TRADE", "reason": "Did not open above resistance", "gap_pct": round(gap_pct, 3),
               "resistance": round(res, 4), "above_res_pct": round(above_pct, 3) if above_pct is not None else None}
        out.update(_market_closed_banner())
        return out

    if above_pct is not None and above_pct > max_above_res_pct:
        out = {"ticker": ticker, "signal": "NO_TRADE", "reason": "Too extended above resistance", "gap_pct": round(gap_pct, 3),
               "resistance": round(res, 4), "above_res_pct": round(above_pct, 3)}
        out.update(_market_closed_banner())
        return out

    # Rel vol (optional) — OFF by default
    rel_vol = None
    if min_rel_vol and min_rel_vol > 0:
        if len(daily) >= 22:
            today_vol = float(daily["Volume"].iloc[-1])
            avg_vol = float(daily["Volume"].iloc[-21:-1].mean())
            if avg_vol > 0:
                rel_vol = today_vol / avg_vol
                if rel_vol < min_rel_vol:
                    out = {"ticker": ticker, "signal": "NO_TRADE", "reason": "Relative volume too low",
                           "gap_pct": round(gap_pct, 3), "rel_vol": round(rel_vol, 3)}
                    out.update(_market_closed_banner())
                    return out

    # Intraday 5m window (first 30 min ET)
    df5 = yf.download(ticker, period="5d", interval="5m", progress=False, auto_adjust=False)
    if df5 is None or df5.empty:
        out = {"ticker": ticker, "signal": "NO_TRADE", "reason": "No 5m data", "gap_pct": round(gap_pct, 3)}
        out.update(_market_closed_banner())
        return out

    df5 = ensure_et(df5)
    last_day = df5.index.date[-1]
    day = df5[df5.index.date == last_day]
    t = day.index.time
    win = day[(t >= dtime(9, 30)) & (t < dtime(10, 0))]

    if win is None or len(win) < 3:
        out = {"ticker": ticker, "signal": "NO_TRADE", "reason": "Not enough 5m bars in first 30 minutes yet",
               "gap_pct": round(gap_pct, 3), "resistance": round(res, 4), "above_res_pct": round(above_pct, 3)}
        out.update(_market_closed_banner())
        return out

    # --- Setup 1: ORB (first 5m narrow-ish)
    ranges = (win["High"] - win["Low"]).astype(float)
    med = float(ranges.median()) if len(ranges) else 0.0
    first = win.iloc[0]
    first_range = float(first["High"] - first["Low"])

    setup = None
    entry = None
    stop = None

    if med > 0 and first_range <= med * 0.8:
        setup = "ORB (first 5m narrow)"
        entry = float(first["High"])
        stop = float(first["Low"])
    else:
        # --- Setup 2: simple base breakout (2–3 small bars after first)
        for start in range(1, min(4, len(win)-2)):
            for length in (2, 3):
                end = start + length
                if end >= len(win):
                    continue
                chunk = win.iloc[start:end]
                chunk_ranges = (chunk["High"] - chunk["Low"]).astype(float)
                if med > 0 and (chunk_ranges <= med * 0.8).all():
                    setup = f"Simple Breakout Base ({length} bars)"
                    entry = float(chunk["High"].max())
                    stop = float(chunk["Low"].min())
                    break
            if setup:
                break

    if setup is None:
        # --- Setup 3: retracement to rising 20MA (computed over df5)
        df5["ma20"] = df5["Close"].astype(float).rolling(20).mean()
        ma_tail = df5["ma20"].dropna().tail(5).values
        if len(ma_tail) >= 5 and _linear_slope(ma_tail) > 0:
            for i in range(1, len(win)):
                c = win.iloc[i]
                ma = float(df5.loc[win.index[i], "ma20"]) if win.index[i] in df5.index else None
                if ma is None or not np.isfinite(ma) or ma <= 0:
                    continue
                low = float(c["Low"])
                high = float(c["High"])
                if abs(low - ma) / ma <= 0.002 or (low <= ma <= high):
                    setup = "Retracement to rising 20MA"
                    entry = high
                    stop = low
                    break

    if setup is None:
        out = {
            "ticker": ticker,
            "signal": "NO_TRADE",
            "reason": "Gap qualifies, but no clean setup in first 30m",
            "gap_pct": round(gap_pct, 3),
            "open": round(t_open, 4),
            "prev_close": round(y_close, 4),
            "resistance": round(res, 4),
            "above_res_pct": round(above_pct, 3),
            "rel_vol": round(rel_vol, 3) if rel_vol is not None else None,
        }
        out.update(_market_closed_banner())
        return out

    risk = (entry - stop) if (entry is not None and stop is not None and entry > stop) else None

    out = {
        "ticker": ticker,
        "signal": "LONG",
        "strategy": "GAP_MOMENTUM",
        "setup": setup,
        "gap_pct": round(gap_pct, 3),
        "open": round(t_open, 4),
        "prev_close": round(y_close, 4),
        "resistance": round(res, 4),
        "above_res_pct": round(above_pct, 3),
        "rel_vol": round(rel_vol, 3) if rel_vol is not None else None,
        "entry_buy_above": round(float(entry), 4),
        "stop_sell_below": round(float(stop), 4),
        "risk_per_share": round(float(risk), 4) if risk is not None else None,
    }
    out.update(_market_closed_banner())
    return out


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/analyze")
def analyze(ticker: str = "NVDA"):
    return analyze_long_only(ticker)

@app.get("/gap_analyze")
def gap_analyze(ticker: str = "NVDA"):
    return gap_analyze_long_only(ticker)
