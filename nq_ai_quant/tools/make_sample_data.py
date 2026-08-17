#!/usr/bin/env python3
"""
Generate SYNTHETIC NQ-like bars so you can verify the whole pipeline runs
before you have real data wired up.

    python tools/make_sample_data.py --bars 40000 --out data/sample/NQ_synthetic.csv

WARNING: this is a stochastic-volatility random walk with session structure. It
has no real edge in it. Any strategy the searcher "finds" on this file is noise.
It exists to prove the plumbing works, nothing else.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def generate(n_bars: int, tf_minutes: int = 5, seed: int = 42,
             start: str = "2021-01-04 00:00") -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # CME NQ trades 18:00 -> 17:00 next day ET with a 1h break at 17:00.
    idx = pd.date_range(start=start, periods=int(n_bars * 1.15), freq=f"{tf_minutes}min",
                        tz="America/New_York")
    mod = idx.hour * 60 + idx.minute
    open_mask = ~((mod >= 1020) & (mod < 1080))            # drop 17:00-18:00
    open_mask &= idx.dayofweek < 5
    idx = idx[open_mask][:n_bars]

    m = np.asarray(idx.hour * 60 + idx.minute, dtype=float)
    is_rth = (m >= 570) & (m < 960)
    # Intraday volatility smile: hot at the RTH open and close, quiet overnight.
    tod = np.where(is_rth, 1.0, 0.35)
    tod = tod * (1 + 0.9 * np.exp(-((m - 575) ** 2) / (2 * 25 ** 2)))   # open burst
    tod = tod * (1 + 0.5 * np.exp(-((m - 950) ** 2) / (2 * 20 ** 2)))   # close burst

    # GARCH-ish stochastic vol so ATR regimes actually change.
    n = len(idx)
    log_v = np.zeros(n)
    for i in range(1, n):
        log_v[i] = 0.995 * log_v[i - 1] + rng.normal(0, 0.03)
    vol = 0.00045 * np.exp(log_v) * tod * np.sqrt(tf_minutes / 5)

    # Fat tails + mild momentum so indicators are not pure noise.
    shock = rng.standard_t(df=4, size=n) / np.sqrt(4 / 2)
    ret = vol * shock
    for i in range(2, n):
        ret[i] += 0.045 * ret[i - 1] - 0.02 * ret[i - 2]

    close = 13000 * np.exp(np.cumsum(ret))
    # Build OHLC around the close path.
    o = np.concatenate([[close[0]], close[:-1]])
    span = np.abs(close - o) + vol * close * rng.uniform(0.4, 1.6, n)
    up = rng.uniform(0, 1, n)
    h = np.maximum(o, close) + span * up * 0.6
    l = np.minimum(o, close) - span * (1 - up) * 0.6
    v = (rng.gamma(3.0, 1.0, n) * 1200 * tod).round()

    df = pd.DataFrame(
        {"timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"),
         "open": o.round(2), "high": h.round(2), "low": l.round(2),
         "close": close.round(2), "volume": v}
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=40000)
    ap.add_argument("--tf-minutes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/sample/NQ_synthetic.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df = generate(a.bars, a.tf_minutes, a.seed)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} synthetic {a.tf_minutes}min bars -> {a.out}")
    print("REMINDER: synthetic data. Results from it mean nothing.")


if __name__ == "__main__":
    main()
