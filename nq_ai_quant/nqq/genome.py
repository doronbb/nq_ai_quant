"""
Strategy genome: the complete, hashable description of one strategy.

Two genomes with the same hash are the same experiment, so the registry can
refuse to run it twice — that is the whole point of `fingerprint()`.

The hash is computed from a CANONICAL form: keys sorted, floats rounded to a
fixed grid, feature groups sorted. So `learning_rate=0.0500001` and `0.05`
collapse to one experiment instead of two.
"""
from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy

from .features import FEATURE_GROUPS

# --------------------------------------------------------------------------
# search space
# --------------------------------------------------------------------------
# ("choice", [...])            pick one
# ("int",   lo, hi, step)      integer grid
# ("float", lo, hi, step)      float snapped to `step` (this is what makes hashing stable)
# ("subset", [...], min, max)  random subset

SPACE = {
    # --- labelling / trade construction --------------------------------
    "label.atr_period":   ("choice", [10, 14, 20, 30]),
    # Ceiling raised 4.0 -> 5.5: the elite pool was pressing against the old top
    # (its highest-RR members sat exactly at tp/sl = 4.0), so the reachable
    # reward-to-risk was capped by the search space, not by the market.
    "label.tp_mult":      ("float", 0.75, 5.5, 0.25),
    "label.sl_mult":      ("float", 0.5, 2.5, 0.25),
    "label.max_bars":     ("choice", [8, 12, 16, 24, 32, 48, 64, 96]),

    # --- features -------------------------------------------------------
    "feat.groups":        ("subset", FEATURE_GROUPS, 2, len(FEATURE_GROUPS)),

    # --- model ----------------------------------------------------------
    "model.n_estimators":      ("int", 60, 500, 20),
    "model.learning_rate":     ("float", 0.01, 0.20, 0.01),
    "model.num_leaves":        ("choice", [7, 15, 31, 63]),
    "model.max_depth":         ("choice", [3, 4, 5, 6, 8, -1]),
    "model.min_child_samples": ("int", 20, 400, 20),
    "model.subsample":         ("float", 0.5, 1.0, 0.1),
    "model.colsample":         ("float", 0.4, 1.0, 0.1),
    "model.reg_lambda":        ("float", 0.0, 10.0, 0.5),

    # --- signal generation ----------------------------------------------
    "signal.long_thresh":  ("float", 0.30, 0.70, 0.02),
    "signal.short_thresh": ("float", 0.30, 0.70, 0.02),
    "signal.edge_margin":  ("float", 0.0, 0.20, 0.02),   # p(side) must beat p(other) by this
    "signal.allow_long":   ("choice", [True]),
    "signal.allow_short":  ("choice", [True, False]),

    # --- entry filters --------------------------------------------------
    "filter.session":      ("choice", ["all", "rth", "globex", "rth_no_open", "open_drive"]),
    "filter.adx_min":      ("choice", [0, 15, 20, 25, 30]),
    "filter.atr_pct_min":  ("float", 0.0, 0.25, 0.05),
    "filter.atr_pct_max":  ("choice", [0.5, 1.0, 2.0, 99.0]),
    "filter.max_trades_per_day": ("choice", [1, 2, 3, 5, 10, 999]),
    "filter.trend_align":  ("choice", [True, False]),    # only long above ema200, etc.
}

FLOAT_KEYS = {k for k, v in SPACE.items() if v[0] == "float"}


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def _sample_one(spec, rng: random.Random):
    kind = spec[0]
    if kind == "choice":
        return rng.choice(spec[1])
    if kind == "int":
        lo, hi, step = spec[1], spec[2], spec[3]
        return int(rng.randrange(lo, hi + 1, step))
    if kind == "float":
        lo, hi, step = spec[1], spec[2], spec[3]
        n = int(round((hi - lo) / step))
        return round(lo + step * rng.randint(0, n), 6)
    if kind == "subset":
        pool, lo, hi = spec[1], spec[2], spec[3]
        k = rng.randint(lo, hi)
        return sorted(rng.sample(list(pool), k))
    raise ValueError(kind)


def random_genome(rng: random.Random | None = None) -> dict:
    rng = rng or random.Random()
    g = {k: _sample_one(v, rng) for k, v in SPACE.items()}
    return repair(g)


def repair(g: dict) -> dict:
    """Enforce cross-parameter constraints so we never waste a slot on nonsense."""
    g = dict(g)
    # Reward must be at least risk. A target inside the stop needs a >50% hit
    # rate just to break even before costs, and after the round-trip cost drag
    # it needs considerably more than that — a losing proposition to search for.
    # Snapped to the tp grid so the fingerprint stays canonical.
    if g["label.tp_mult"] < g["label.sl_mult"]:
        lo, hi, step = SPACE["label.tp_mult"][1:4]
        want = round(g["label.sl_mult"] / step) * step
        g["label.tp_mult"] = round(min(hi, max(lo, want)), 6)
    # Need enough bars for the target to be reachable at all.
    min_bars = 8
    if g["label.max_bars"] < min_bars:
        g["label.max_bars"] = min_bars
    if not g["signal.allow_long"] and not g["signal.allow_short"]:
        g["signal.allow_long"] = True
    if g["filter.atr_pct_max"] <= g["filter.atr_pct_min"]:
        g["filter.atr_pct_max"] = 99.0
    if not g["feat.groups"]:
        g["feat.groups"] = ["trend", "momentum"]
    g["feat.groups"] = sorted(set(g["feat.groups"]))
    return g


# --------------------------------------------------------------------------
# variation
# --------------------------------------------------------------------------

def mutate(g: dict, rng: random.Random, rate: float = 0.25) -> dict:
    """Point mutation. Numeric genes take a local step; others resample."""
    out = deepcopy(g)
    keys = list(SPACE)
    n_mut = max(1, int(round(rate * len(keys))))
    for k in rng.sample(keys, n_mut):
        spec = SPACE[k]
        if spec[0] == "float" and rng.random() < 0.7:
            lo, hi, step = spec[1], spec[2], spec[3]
            delta = step * rng.choice([-3, -2, -1, 1, 2, 3])
            out[k] = round(min(hi, max(lo, out[k] + delta)), 6)
        elif spec[0] == "int" and rng.random() < 0.7:
            lo, hi, step = spec[1], spec[2], spec[3]
            out[k] = int(min(hi, max(lo, out[k] + step * rng.choice([-2, -1, 1, 2]))))
        elif spec[0] == "subset" and rng.random() < 0.7:
            pool, mn, mx = spec[1], spec[2], spec[3]
            cur = set(out[k])
            if cur and (len(cur) > mn) and rng.random() < 0.5:
                cur.discard(rng.choice(sorted(cur)))
            else:
                cand = [x for x in pool if x not in cur]
                if cand and len(cur) < mx:
                    cur.add(rng.choice(cand))
            out[k] = sorted(cur)
        else:
            out[k] = _sample_one(spec, rng)
    return repair(out)


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    """Uniform crossover, biased slightly toward parent A (the better parent)."""
    out = {}
    for k in SPACE:
        out[k] = deepcopy(a[k]) if rng.random() < 0.6 else deepcopy(b[k])
    return repair(out)


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------

def canonical(g: dict) -> str:
    """Stable JSON. Floats are snapped to the space's step grid before hashing."""
    c = {}
    for k in sorted(SPACE):
        v = g[k]
        if k in FLOAT_KEYS:
            step = SPACE[k][3]
            v = round(round(float(v) / step) * step, 6)
            v = 0.0 if v == 0 else v          # kill -0.0
        elif isinstance(v, list):
            v = sorted(v)
        elif isinstance(v, bool):
            v = bool(v)
        elif isinstance(v, (int,)):
            v = int(v)
        c[k] = v
    return json.dumps(c, sort_keys=True, separators=(",", ":"))


def fingerprint(g: dict) -> str:
    """The dedupe key. Same fingerprint == same experiment, never run twice."""
    return hashlib.sha256(canonical(g).encode()).hexdigest()[:20]


def describe(g: dict) -> str:
    return (
        f"{'L/S' if g['signal.allow_short'] else 'L-only'} "
        f"tp{g['label.tp_mult']}/sl{g['label.sl_mult']} x{g['label.max_bars']}b "
        f"atr{g['label.atr_period']} | {g['filter.session']} adx>={g['filter.adx_min']} "
        f"| thr {g['signal.long_thresh']:.2f}/{g['signal.short_thresh']:.2f} "
        f"| feats[{len(g['feat.groups'])}] {'+'.join(x[:3] for x in g['feat.groups'])}"
    )
