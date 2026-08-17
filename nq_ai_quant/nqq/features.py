"""
Feature engineering. Every feature at bar t uses ONLY bars <= t.

Lookahead discipline (enforced by tests/test_lookahead.py):
  - no .shift(-n), no centered windows, no full-series normalisation
  - all rolling stats are trailing and min_periods-complete
  - features are consumed by a model that enters at open[t+1], so bar t's
    close is legitimately known at decision time.

Features are grouped so the genome can switch whole blocks on/off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_GROUPS = [
    "trend", "momentum", "volatility", "volume", "patterns", "session", "returns",
]


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    rs = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / (
        dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() + 1e-12
    )
    return 100.0 - 100.0 / (1.0 + rs)


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR. Used both as a feature and to size the triple barriers."""
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr_n = true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / n, adjust=False, min_periods=n).mean() / (tr_n + 1e-12)
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / n, adjust=False, min_periods=n).mean() / (tr_n + 1e-12)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), pdi, mdi


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP anchored to the session date, cumulative and therefore lookahead-free."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    g = df.groupby("_session_date", sort=False)
    return (pv.groupby(g.ngroup()).cumsum() /
            (df["volume"].groupby(g.ngroup()).cumsum() + 1e-12))


# --------------------------------------------------------------------------
# builder
# --------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return (features, group -> column names). Index matches df."""
    f = pd.DataFrame(index=df.index)
    groups: dict[str, list[str]] = {g: [] for g in FEATURE_GROUPS}
    c, h, l, o, v = (df["close"], df["high"], df["low"], df["open"], df["volume"])

    a14 = atr(df, 14)
    scale = a14.replace(0, np.nan)                # price-normalising denominator

    # ---- trend -----------------------------------------------------------
    for n in (9, 21, 50, 200):
        e = ema(c, n)
        f[f"ema{n}_dist"] = (c - e) / scale        # ATR-normalised => regime stable
        f[f"ema{n}_slope"] = e.diff(5) / scale
    f["ema9_21"] = (ema(c, 9) - ema(c, 21)) / scale
    f["ema21_50"] = (ema(c, 21) - ema(c, 50)) / scale
    f["ema50_200"] = (ema(c, 50) - ema(c, 200)) / scale
    f["sma20_dist"] = (c - sma(c, 20)) / scale
    f["ema_stack"] = (
        np.sign(ema(c, 9) - ema(c, 21))
        + np.sign(ema(c, 21) - ema(c, 50))
        + np.sign(ema(c, 50) - ema(c, 200))
    )
    adx14, pdi, mdi = adx(df, 14)
    f["adx14"] = adx14
    f["di_diff"] = pdi - mdi
    groups["trend"] = [x for x in f.columns]

    # ---- momentum --------------------------------------------------------
    prev = set(f.columns)
    f["rsi14"] = rsi(c, 14)
    f["rsi7"] = rsi(c, 7)
    f["rsi14_slope"] = f["rsi14"].diff(3)
    macd = ema(c, 12) - ema(c, 26)
    macd_sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd"] = macd / scale
    f["macd_hist"] = (macd - macd_sig) / scale
    f["macd_hist_slope"] = f["macd_hist"].diff(2)
    f["roc10"] = c.pct_change(10) * 100
    f["roc30"] = c.pct_change(30) * 100
    st_lo, st_hi = l.rolling(14, min_periods=14).min(), h.rolling(14, min_periods=14).max()
    f["stoch_k"] = 100 * (c - st_lo) / (st_hi - st_lo + 1e-12)
    groups["momentum"] = [x for x in f.columns if x not in prev]

    # ---- volatility ------------------------------------------------------
    prev = set(f.columns)
    f["atr14_pct"] = a14 / c * 100
    f["atr_ratio"] = a14 / (atr(df, 50) + 1e-12)          # vol expansion/contraction
    bb_m = sma(c, 20)
    bb_s = c.rolling(20, min_periods=20).std()
    f["bb_width"] = (4 * bb_s) / (bb_m + 1e-12) * 100
    f["bb_pos"] = (c - bb_m) / (2 * bb_s + 1e-12)          # -1..1 inside the bands
    r1 = np.log(c / c.shift(1))
    f["rvol20"] = r1.rolling(20, min_periods=20).std() * 100
    f["rvol_ratio"] = f["rvol20"] / (r1.rolling(100, min_periods=100).std() * 100 + 1e-12)
    rng = (h - l)
    f["range_pctile50"] = rng.rolling(50, min_periods=50).rank(pct=True)
    groups["volatility"] = [x for x in f.columns if x not in prev]

    # ---- volume ----------------------------------------------------------
    prev = set(f.columns)
    vmean = v.rolling(50, min_periods=50).mean()
    f["vol_z"] = (v - vmean) / (v.rolling(50, min_periods=50).std() + 1e-12)
    f["vol_ratio"] = v / (vmean + 1e-12)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_slope"] = (obv.diff(10)) / (vmean * 10 + 1e-12)
    vw = session_vwap(df)
    f["vwap_dist"] = (c - vw) / scale
    f["vwap_cross"] = np.sign(c - vw) - np.sign(c.shift(1) - vw.shift(1))
    # Volume-at-price proxy: where is price vs the highest-volume price of the day so far.
    f["dollar_vol_z"] = ((c * v) - (c * v).rolling(50, min_periods=50).mean()) / (
        (c * v).rolling(50, min_periods=50).std() + 1e-12)
    groups["volume"] = [x for x in f.columns if x not in prev]

    # ---- candle patterns -------------------------------------------------
    prev = set(f.columns)
    body = (c - o)
    abs_body = body.abs()
    rng_safe = rng.replace(0, np.nan)
    upper = h - np.maximum(c, o)
    lower = np.minimum(c, o) - l
    f["body_ratio"] = body / rng_safe
    f["upper_wick_ratio"] = upper / rng_safe
    f["lower_wick_ratio"] = lower / rng_safe
    f["wick_imbalance"] = (upper - lower) / rng_safe
    f["is_doji"] = (abs_body / rng_safe < 0.1).astype(float)
    f["is_pin_bull"] = ((lower / rng_safe > 0.6) & (body > 0)).astype(float)
    f["is_pin_bear"] = ((upper / rng_safe > 0.6) & (body < 0)).astype(float)
    po, pc_, ph, pl = o.shift(1), c.shift(1), h.shift(1), l.shift(1)
    f["engulf_bull"] = ((c > o) & (pc_ < po) & (c >= po) & (o <= pc_)).astype(float)
    f["engulf_bear"] = ((c < o) & (pc_ > po) & (c <= po) & (o >= pc_)).astype(float)
    f["inside_bar"] = ((h <= ph) & (l >= pl)).astype(float)
    f["outside_bar"] = ((h > ph) & (l < pl)).astype(float)
    f["body_vs_atr"] = abs_body / scale
    f["gap_atr"] = (o - pc_) / scale
    f["consec_up"] = (np.sign(body).groupby((np.sign(body) != np.sign(body).shift()).cumsum())
                      .cumsum())
    groups["patterns"] = [x for x in f.columns if x not in prev]

    # ---- session / time --------------------------------------------------
    prev = set(f.columns)
    mod = df["_minute_of_day"].astype(float)
    f["tod_sin"] = np.sin(2 * np.pi * mod / 1440.0)
    f["tod_cos"] = np.cos(2 * np.pi * mod / 1440.0)
    f["is_rth"] = df["_is_rth"].astype(float)
    f["is_open_drive"] = df["_is_open_drive"].astype(float)
    f["is_close_hour"] = df["_is_close"].astype(float)
    f["dow"] = df["_dow"].astype(float)
    f["mins_from_rth_open"] = (mod - 570.0).clip(-1440, 1440)
    g = df.groupby("_session_date", sort=False).ngroup()
    f["bars_into_session"] = df.groupby(g).cumcount().astype(float)
    day_hi = h.groupby(g).cummax()
    day_lo = l.groupby(g).cummin()
    f["pos_in_day_range"] = (c - day_lo) / (day_hi - day_lo + 1e-12)
    f["dist_day_high"] = (day_hi - c) / scale
    f["dist_day_low"] = (c - day_lo) / scale
    groups["session"] = [x for x in f.columns if x not in prev]

    # ---- raw returns -----------------------------------------------------
    prev = set(f.columns)
    for n in (1, 2, 3, 5, 10, 20):
        f[f"ret{n}_atr"] = (c - c.shift(n)) / scale
    f["ret1_z"] = r1 / (r1.rolling(50, min_periods=50).std() + 1e-12)
    f["hl_ratio5"] = (h.rolling(5, min_periods=5).max() - c) / (
        c - l.rolling(5, min_periods=5).min() + 1e-12)
    groups["returns"] = [x for x in f.columns if x not in prev]

    f = f.replace([np.inf, -np.inf], np.nan)
    return f, groups


def select_columns(groups: dict[str, list[str]], enabled: list[str]) -> list[str]:
    cols: list[str] = []
    for g in enabled:
        cols.extend(groups.get(g, []))
    return cols
