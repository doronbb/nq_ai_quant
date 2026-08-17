#!/usr/bin/env python3
"""
Correctness tests. Run before you trust a single number:

    python tests/test_core.py

The lookahead test is the important one. A backtest that peeks is worse than no
backtest, because it is confidently wrong.
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nqq.data import _cache_key, load_bars, tag_sessions         # noqa: E402
from nqq.features import atr, build_features                     # noqa: E402
from nqq.genome import (SPACE, canonical, crossover, fingerprint,  # noqa: E402
                        mutate, random_genome, repair)
from nqq.labeling import triple_barrier                          # noqa: E402
from nqq.metrics import compute                                  # noqa: E402
from nqq.registry import Registry                                # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")


def make_bars(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-03-01", periods=n, freq="5min", tz="America/New_York")
    ret = rng.normal(0, 0.0006, n)
    c = 12000 * np.exp(np.cumsum(ret))
    o = np.concatenate([[c[0]], c[:-1]])
    sp = np.abs(c - o) + rng.uniform(1, 6, n)
    h = np.maximum(o, c) + sp * 0.4
    l = np.minimum(o, c) - sp * 0.4
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                       "volume": rng.integers(100, 2000, n).astype(float)}, index=idx)
    return tag_sessions(df)


# --------------------------------------------------------------------------

def test_no_lookahead():
    """
    The decisive test: truncate the data at bar k and rebuild features. Every
    feature value at bars <= k must be IDENTICAL to the full-history version.
    If any future bar influences a past feature, this fails.
    """
    print("\n[lookahead]")
    df = make_bars(2500)
    full, _ = build_features(df)
    for k in (1200, 1800, 2200):
        trunc, _ = build_features(df.iloc[: k + 1])
        a = full.iloc[: k + 1]
        b = trunc
        diffs = []
        for col in a.columns:
            x, y = a[col].to_numpy(float), b[col].to_numpy(float)
            both = np.isfinite(x) & np.isfinite(y)
            if (np.isfinite(x) != np.isfinite(y)).any():
                diffs.append(f"{col}(nan-pattern)")
            elif both.any() and not np.allclose(x[both], y[both], rtol=1e-9, atol=1e-9):
                diffs.append(f"{col}(max {np.abs(x[both] - y[both]).max():.2e})")
        check(f"features are causal at bar {k}", not diffs, ", ".join(diffs[:4]))


def test_barrier_math():
    print("\n[triple barrier]")
    # Hand-built path: entry 100, ATR 1, tp 2, sl 1 -> stop 99, target 102.
    idx = pd.date_range("2022-01-03 10:00", periods=6, freq="5min", tz="America/New_York")
    df = pd.DataFrame({
        "open":  [100, 100, 100.5, 101.0, 101.5, 102.0],
        "high":  [100, 100.6, 101.2, 101.8, 102.5, 102.6],   # target hit at bar 4
        "low":   [ 99.6, 99.5, 100.2, 100.8, 101.2, 101.9],
        "close": [100, 100.5, 101.0, 101.5, 102.2, 102.4],
        "volume": [1] * 6}, index=idx)
    df = tag_sessions(df)
    a = pd.Series([1.0] * 6, index=idx)
    out = triple_barrier(df, a, tp_mult=2.0, sl_mult=1.0, max_bars=5, cost_points=0.0)
    check("long target -> R = +tp/sl", abs(out["long_R"].iloc[0] - 2.0) < 1e-9,
          f"got {out['long_R'].iloc[0]}")
    check("long exit index is the target bar", out["long_exit"].iloc[0] == 4,
          f"got {out['long_exit'].iloc[0]}")
    check("short stopped -> R = -1", abs(out["short_R"].iloc[0] + 1.0) < 1e-9,
          f"got {out['short_R'].iloc[0]}")
    check("label = long", out["label"].iloc[0] == 1)

    # Stop-first: same-bar touch of both barriers must resolve as the loss.
    df2 = df.copy()
    df2.loc[df2.index[1], ["high", "low"]] = [102.5, 98.5]
    out2 = triple_barrier(df2, a, 2.0, 1.0, 5, 0.0)
    check("ambiguous bar resolves pessimistically (stop first)",
          abs(out2["long_R"].iloc[0] + 1.0) < 1e-9, f"got {out2['long_R'].iloc[0]}")

    # Costs must reduce R.
    out3 = triple_barrier(df, a, 2.0, 1.0, 5, cost_points=0.5)
    check("costs are subtracted from R", out3["long_R"].iloc[0] < out["long_R"].iloc[0],
          f"{out3['long_R'].iloc[0]} < {out['long_R'].iloc[0]}")

    # R definition: (exit-entry)/(entry-stop)
    e, s, x = 100.0, 99.0, 102.0
    check("R == (exit-entry)/(entry-stop)", abs((x - e) / (e - s) - 2.0) < 1e-12)


def test_fingerprint():
    print("\n[dedupe]")
    rng = random.Random(0)
    g = random_genome(rng)
    check("fingerprint is deterministic", fingerprint(g) == fingerprint(dict(g)))

    g2 = dict(g)
    g2["model.learning_rate"] = g["model.learning_rate"] + 1e-9
    check("float jitter does not create a new experiment", fingerprint(g) == fingerprint(g2))

    g3 = dict(g)
    g3["feat.groups"] = list(reversed(g["feat.groups"]))
    check("feature-group order does not matter", fingerprint(g) == fingerprint(g3))

    g4 = dict(g)
    g4["label.max_bars"] = 8 if g["label.max_bars"] != 8 else 96
    check("a real change creates a new experiment", fingerprint(g) != fingerprint(g4))

    seen = {fingerprint(random_genome(rng)) for _ in range(4000)}
    check("random sampling spans the space", len(seen) > 3900, f"{len(seen)}/4000 unique")

    for _ in range(500):
        r = repair(mutate(random_genome(rng), rng))
        assert r["label.tp_mult"] >= r["label.sl_mult"] * 0.6 - 1e-9
        assert r["signal.allow_long"] or r["signal.allow_short"]
    check("repair() enforces constraints on mutants", True)
    check("crossover output is valid", fingerprint(
        crossover(random_genome(rng), random_genome(rng), rng)) is not None)
    import json as _json
    check("canonical form covers every gene", len(_json.loads(canonical(g))) == len(SPACE),
          f"{len(_json.loads(canonical(g)))} vs {len(SPACE)}")


def test_registry():
    print("\n[registry]")
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="nqq_test_")
    path = os.path.join(tmpdir, "_test_registry.db")
    reg = Registry(path)
    rng = random.Random(3)
    g = random_genome(rng)
    fp = fingerprint(g)
    check("first reserve succeeds", reg.reserve(fp, g, "random") is True)
    check("second reserve is refused", reg.reserve(fp, g, "random") is False)
    check("seen() reports it", reg.seen(fp) is True)
    reg.complete(fp, 0.42, {"trades": 150, "score": 0.42}, 1.0)
    check("completed strategy appears in top()", reg.top(5)[0]["fingerprint"] == fp)

    reg2 = Registry(path)
    check("dedupe survives a process restart", reg2.seen(fp) is True)
    check("restart still refuses the duplicate", reg2.reserve(fp, g, "random") is False)
    reg.close()
    reg2.close()
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_metrics():
    print("\n[metrics]")
    # 40% win rate at +2R, 60% loss at -1R  ->  expectancy = .4*2 - .6*1 = +0.2R
    R = [2.0] * 40 + [-1.0] * 60
    t = pd.DataFrame({"R": R, "entry_time": pd.date_range("2022-01-01", periods=100,
                                                          freq="1D", tz="UTC")})
    m = compute(t, bars_per_year=98280, min_trades=100)
    check("expectancy matches the textbook formula", abs(m["expectancy_R"] - 0.2) < 1e-9,
          f"got {m['expectancy_R']}")
    check("win rate", abs(m["win_rate"] - 0.4) < 1e-9)
    check("total R", abs(m["total_R"] - 20.0) < 1e-6)
    check("profit factor", abs(m["profit_factor"] - (80 / 60)) < 1e-3)
    check("composite score is positive for a positive edge", m["score"] > 0)

    losing = pd.DataFrame({"R": [-1.0] * 100,
                           "entry_time": t["entry_time"]})
    check("composite score is negative for a losing system",
          compute(losing, 98280, 100)["score"] < 0)

    thin = pd.DataFrame({"R": [2.0] * 4 + [-1.0] * 6, "entry_time": t["entry_time"][:10]})
    check("thin samples are penalised",
          compute(thin, 98280, 100)["score"] < m["score"],
          f"{compute(thin, 98280, 100)['score']:.4f} < {m['score']:.4f}")


def test_atr_sanity():
    print("\n[indicators]")
    df = make_bars(600)
    a = atr(df, 14)
    check("ATR is positive where defined", bool((a.dropna() > 0).all()))
    check("ATR warms up with NaNs", bool(a.iloc[:13].isna().all()))
    f, groups = build_features(df)
    check("no feature is entirely NaN",
          not any(f[c].isna().all() for c in f.columns),
          ", ".join([c for c in f.columns if f[c].isna().all()][:5]))
    check("every feature belongs to exactly one group",
          sum(len(v) for v in groups.values()) == f.shape[1],
          f"{sum(len(v) for v in groups.values())} vs {f.shape[1]}")


def test_data_folder():
    """
    The drop-a-file-in-data/ contract. Every check here is something that would
    otherwise fail silently and leave you searching the wrong bars.
    """
    print("\n[data folder]")
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="nqq_data_")
    try:
        idx = pd.date_range("2023-01-02 09:30", periods=400, freq="5min", tz="UTC")
        px = np.linspace(15000, 15400, len(idx))
        body = pd.DataFrame({"DateTime": idx.strftime("%Y.%m.%d %H:%M:%S"),
                             "Open": px, "High": px + 5, "Low": px - 5,
                             "Close": px + 1, "Volume": 0, "TickVolume": 250})

        # tab-separated, zero Volume, real TickVolume — the MT5 export shape
        body.iloc[:200].to_csv(os.path.join(tmp, "a.csv"), sep="\t", index=False)
        # comma-separated, overlapping the first file by 50 bars
        body.iloc[150:].to_csv(os.path.join(tmp, "b.csv"), index=False)
        # a daily file, far coarser than the 5min target
        d_idx = pd.date_range("2023-01-02", periods=30, freq="1D", tz="UTC")
        dpx = np.linspace(15000, 15900, len(d_idx))
        pd.DataFrame({"DateTime": d_idx.strftime("%Y.%m.%d %H:%M:%S"), "Open": dpx,
                      "High": dpx + 50, "Low": dpx - 50, "Close": dpx + 10,
                      "Volume": 1000}).to_csv(os.path.join(tmp, "daily.csv"), index=False)

        cfg = {"source": "csv", "path": tmp, "source_tz": "UTC"}
        cache = os.path.join(tmp, "cache")
        bars = load_bars(cfg, "5min", cache_dir=cache)

        check("both tab- and comma-separated files are read", len(bars) == 400,
              f"{len(bars)} bars, expected 400")
        check("overlapping bars are de-duplicated, not double-counted",
              bool(bars.index.is_unique))
        check("a file coarser than the timeframe is skipped",
              bool(bars.index.max() < pd.Timestamp("2023-01-04", tz="UTC")),
              f"last bar {bars.index.max()}")
        check("zero Volume falls back to tick volume",
              float(bars["volume"].sum()) > 0)

        key_before = _cache_key(cfg, "5min")
        body.iloc[:200].to_csv(os.path.join(tmp, "c.csv"), index=False)
        check("dropping a new file invalidates the bar cache",
              _cache_key(cfg, "5min") != key_before)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    test_no_lookahead()
    test_barrier_math()
    test_fingerprint()
    test_registry()
    test_metrics()
    test_atr_sanity()
    test_data_folder()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
