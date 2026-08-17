"""
Triple-barrier labelling, in R-multiples.

Convention used everywhere in this project
------------------------------------------
Signal is produced from bar t (using data <= t).
Entry fills at open[t+1].  Stop and target are set from ATR[t]:

    long :  stop = entry - sl_mult * atr[t]      target = entry + tp_mult * atr[t]
    short:  stop = entry + sl_mult * atr[t]      target = entry - tp_mult * atr[t]

Walk bars t+1 .. t+max_bars:
    stop touched first  -> R = -1
    target touched first-> R = +tp_mult/sl_mult
    both inside one bar -> assume STOP first (pessimistic; you cannot know the path)
    neither             -> exit at close, R = (exit-entry)/(entry-stop), sign adjusted

R-multiple is therefore exactly the project definition:
    R = (exit - entry) / (entry - stop),  sign-adjusted for direction.

Costs (commission + slippage, in points) are subtracted from the numerator so
every R you see is net.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Barrier outcome codes stored alongside R.
OUT_STOP, OUT_TARGET, OUT_TIME, OUT_NONE = 0, 1, 2, -1


def triple_barrier(
    df: pd.DataFrame,
    atr_series: pd.Series,
    tp_mult: float,
    sl_mult: float,
    max_bars: int,
    cost_points: float = 0.0,
) -> pd.DataFrame:
    """
    Vectorised-ish barrier walk. Returns a frame indexed like df with:
        long_R, short_R      net R-multiple if you had entered at open[t+1]
        long_exit, short_exit integer bar index of the exit
        long_out, short_out   outcome code
        label                 -1 / 0 / +1 training target
        entry_price, long_stop, short_stop, long_target, short_target
    Rows where the trade cannot complete (end of data / missing ATR) get NaN.
    """
    n = len(df)
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    a = atr_series.to_numpy(np.float64)

    long_R = np.full(n, np.nan)
    short_R = np.full(n, np.nan)
    long_exit = np.full(n, -1, np.int64)
    short_exit = np.full(n, -1, np.int64)
    long_out = np.full(n, OUT_NONE, np.int8)
    short_out = np.full(n, OUT_NONE, np.int8)
    entry_px = np.full(n, np.nan)
    l_stop = np.full(n, np.nan)
    s_stop = np.full(n, np.nan)
    l_tgt = np.full(n, np.nan)
    s_tgt = np.full(n, np.nan)

    rr = tp_mult / sl_mult                      # reward:risk at the target

    for t in range(n - 1):
        atr_t = a[t]
        if not np.isfinite(atr_t) or atr_t <= 0:
            continue
        entry = o[t + 1]
        if not np.isfinite(entry):
            continue
        risk = sl_mult * atr_t
        reward = tp_mult * atr_t
        entry_px[t] = entry
        ls, lt = entry - risk, entry + reward
        ss, st = entry + risk, entry - reward
        l_stop[t], l_tgt[t], s_stop[t], s_tgt[t] = ls, lt, ss, st

        last = min(t + max_bars, n - 1)

        # ---- long ----
        done = False
        for k in range(t + 1, last + 1):
            if l[k] <= ls:                       # stop checked first: pessimistic
                long_R[t], long_exit[t], long_out[t] = -1.0 - cost_points / risk, k, OUT_STOP
                done = True
                break
            if h[k] >= lt:
                long_R[t], long_exit[t], long_out[t] = rr - cost_points / risk, k, OUT_TARGET
                done = True
                break
        if not done:
            long_R[t] = (c[last] - entry - cost_points) / risk
            long_exit[t], long_out[t] = last, OUT_TIME

        # ---- short ----
        done = False
        for k in range(t + 1, last + 1):
            if h[k] >= ss:
                short_R[t], short_exit[t], short_out[t] = -1.0 - cost_points / risk, k, OUT_STOP
                done = True
                break
            if l[k] <= st:
                short_R[t], short_exit[t], short_out[t] = rr - cost_points / risk, k, OUT_TARGET
                done = True
                break
        if not done:
            short_R[t] = (entry - c[last] - cost_points) / risk
            short_exit[t], short_out[t] = last, OUT_TIME

    # Training target: which side actually reached its profit barrier.
    label = np.zeros(n, np.int8)
    label[long_out == OUT_TARGET] = 1
    label[short_out == OUT_TARGET] = -1
    # If (rarely) both targets are reachable, favour the one that got there sooner.
    both = (long_out == OUT_TARGET) & (short_out == OUT_TARGET)
    if both.any():
        label[both] = np.where(long_exit[both] <= short_exit[both], 1, -1)
    label[~np.isfinite(long_R)] = 0

    return pd.DataFrame(
        {
            "long_R": long_R, "short_R": short_R,
            "long_exit": long_exit, "short_exit": short_exit,
            "long_out": long_out, "short_out": short_out,
            "label": label,
            "entry_price": entry_px,
            "long_stop": l_stop, "short_stop": s_stop,
            "long_target": l_tgt, "short_target": s_tgt,
            "valid": np.isfinite(long_R),
        },
        index=df.index,
    )


class BarrierCache:
    """
    Barrier walks are the expensive part and many genomes share the same
    (atr_period, tp, sl, max_bars, cost) tuple. Memoise them.
    """

    def __init__(self, maxsize: int = 24):
        self._store: dict[tuple, pd.DataFrame] = {}
        self._order: list[tuple] = []
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, df: pd.DataFrame, atr_series: pd.Series, tp: float, sl: float,
            max_bars: int, cost: float, atr_period: int) -> pd.DataFrame:
        key = (atr_period, round(tp, 4), round(sl, 4), int(max_bars), round(cost, 4), len(df))
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        out = triple_barrier(df, atr_series, tp, sl, max_bars, cost)
        self._store[key] = out
        self._order.append(key)
        while len(self._order) > self.maxsize:
            self._store.pop(self._order.pop(0), None)
        return out
