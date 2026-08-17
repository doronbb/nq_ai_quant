"""
R-multiple performance metrics and the composite objective.

Everything is in R. Nothing is in dollars — the whole point is that the results
are account-size agnostic.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Composite objective weights. Must sum to 1.0.
W_EXPECTANCY = 0.34
W_SHARPE = 0.22
W_RECOVERY = 0.22
W_PAYOFF = 0.22

# Scale constants: the value of each metric that maps to ~0.76 of its component.
SCALE_EXPECTANCY = 0.20      # +0.20R per trade is genuinely good
SCALE_SHARPE = 1.50          # annualised Sharpe of 1.5
SCALE_RECOVERY = 3.00        # total R = 3x max drawdown R
SCALE_PAYOFF = 2.00          # average winner is 2x the average loser

# The t-statistic a strategy must reach to keep its full score.
#
# This is a multiple-comparisons bar, not a textbook significance level. Searching
# N genomes and keeping the best is N draws from the distribution of noise: the
# expected best t-stat from N pure-noise strategies is ~sqrt(2*ln N), which is
# 2.9 at N=500 and 3.3 at N=5000. A result at t=2.8 after a 500-genome search is
# exactly what luck alone produces, and means nothing.
#
# 3.5 is the bar for a search in the thousands. Raise it as the registry grows —
# and rerun tools/rescore.py when you do, so old and new scores stay comparable.
T_TARGET = 3.50


def compute(trades: pd.DataFrame, bars_per_year: float, min_trades: int = 100) -> dict:
    """
    trades: needs at least columns  R (float)  and  entry_time (datetime).
    Returns a flat dict of metrics plus `score` (the composite objective).
    """
    n = len(trades)
    if n == 0:
        return _empty()

    R = trades["R"].to_numpy(np.float64)
    R = R[np.isfinite(R)]
    n = len(R)
    if n == 0:
        return _empty()

    wins, losses = R[R > 0], R[R <= 0]
    win_rate = len(wins) / n
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0     # negative
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss  # == R.mean()

    equity = np.cumsum(R)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    dd = peak - equity
    max_dd = float(dd.max()) if n else 0.0
    total_R = float(equity[-1])

    # Sharpe on the per-trade R series, annualised by realised trade frequency.
    sd = float(R.std(ddof=1)) if n > 1 else 0.0
    span_days = _span_days(trades)
    trades_per_year = (n / span_days * 365.25) if span_days > 0 else 0.0
    sharpe = (expectancy / sd * np.sqrt(trades_per_year)) if sd > 1e-9 and trades_per_year else 0.0
    # t-stat of mean R against zero. Unlike a raw trade count this scales with the
    # size of the edge, so a 130-trade strategy at +0.7R can outrank a 250-trade
    # one at +0.25R — which is the correct ordering, and the one the old
    # trades/min_trades penalty got backwards.
    t_stat = (expectancy / sd * np.sqrt(n)) if sd > 1e-9 else 0.0

    profit_factor = (float(wins.sum()) / abs(float(losses.sum()))) if losses.sum() < 0 else np.inf
    # Reward-to-risk actually realised, which is not the tp/sl geometry: timeouts
    # and cost drag pull it well below the nominal target-over-stop ratio.
    if avg_loss < -1e-9:
        payoff = avg_win / abs(avg_loss)
    else:
        payoff = 10.0 if avg_win > 0 else 0.0
    recovery = total_R / max_dd if max_dd > 1e-9 else (total_R if total_R > 0 else 0.0)

    # Longest losing streak — a practical "can you actually trade this" number.
    streak = worst = 0
    for r in R:
        streak = streak + 1 if r <= 0 else 0
        worst = max(worst, streak)

    m = {
        "trades": int(n),
        "win_rate": round(win_rate, 4),
        "avg_R": round(float(R.mean()), 4),
        "avg_win_R": round(avg_win, 4),
        "avg_loss_R": round(avg_loss, 4),
        "expectancy_R": round(float(expectancy), 4),
        "total_R": round(total_R, 2),
        "max_dd_R": round(max_dd, 2),
        "recovery_factor": round(float(recovery), 3),
        "sharpe": round(float(sharpe), 3),
        "profit_factor": round(float(min(profit_factor, 999)), 3),
        "payoff_ratio": round(float(min(payoff, 999)), 3),
        "t_stat": round(float(t_stat), 3),
        "max_losing_streak": int(worst),
        "trades_per_year": round(trades_per_year, 1),
        "span_days": round(span_days, 1),
        "best_R": round(float(R.max()), 2),
        "worst_R": round(float(R.min()), 2),
    }
    m["score"] = composite(m, min_trades)
    return m


def composite(m: dict, min_trades: int = 100) -> float:
    """
    Bounded blend of expectancy, Sharpe, recovery factor and realised
    reward-to-risk, each squashed by tanh so no single metric can be gamed to
    dominate, then penalised for thin sample size. Range is roughly (-1, +1).

        score = [0.34*tanh(E/0.2) + 0.22*tanh(S/1.5)
               + 0.22*tanh(RF/3)  + 0.22*tanh(payoff/2)] * sample_penalty

    The payoff term is credited only to systems that are already profitable.
    Otherwise a strategy could buy score with a fat average winner while losing
    money overall — high RR on a negative edge is not an achievement.

    evidence_penalty = min(1, t_stat / T_TARGET). A strategy keeps its full score
    only once its edge is large enough, and measured over enough trades, to stand
    above what a search of this size produces by luck. `min_trades` remains a
    separate hard gate in the search loop; this is the graded version.
    """
    e = np.tanh(m["expectancy_R"] / SCALE_EXPECTANCY)
    s = np.tanh(m["sharpe"] / SCALE_SHARPE)
    r = np.tanh(m["recovery_factor"] / SCALE_RECOVERY)
    p = np.tanh(m.get("payoff_ratio", 0.0) / SCALE_PAYOFF) if m["expectancy_R"] > 0 else 0.0
    raw = W_EXPECTANCY * e + W_SHARPE * s + W_RECOVERY * r + W_PAYOFF * p

    penalty = min(1.0, max(m.get("t_stat", 0.0), 0.0) / T_TARGET)
    # Trade count still matters on its own: a huge edge over 30 trades is a story,
    # not a strategy, and t alone would let it through.
    penalty *= min(1.0, m["trades"] / max(min_trades, 1)) ** 0.5
    if raw > 0:
        raw *= penalty
    else:
        # Weak evidence AND losing -> punished harder, but the divisor is floored:
        # with a t-stat near zero the old 1e-3 floor produced scores like -307,
        # which is outside the documented range and makes the leaderboard's
        # numbers unreadable. The ordering was never wrong, the scale was.
        raw /= max(penalty, 0.25)
    return round(float(max(-1.0, min(1.0, raw))), 5)


def noise_ceiling(n_searched: int) -> float:
    """
    The best t-statistic you should EXPECT from `n_searched` pure-noise
    strategies — the expected maximum of n standard normals (Gumbel
    approximation). Anything at or below this is what luck produces for free.

    n =   100 -> 2.51        n =  5000 -> 3.34
    n =  1000 -> 3.04        n = 20000 -> 3.61

    The bar rises as the search runs, which is the honest accounting: every
    additional genome tried is another lottery ticket bought.
    """
    n = max(int(n_searched), 3)
    a = math.sqrt(2.0 * math.log(n))
    return a - (math.log(math.log(n)) + math.log(4.0 * math.pi)) / (2.0 * a)


# A find has to clear the noise ceiling by this much before it is worth your
# attention. Half a t-unit is not arbitrary: it is roughly the spread between
# the expected maximum and the 90th percentile of that maximum.
T_MARGIN = 0.50

MIN_RR = 2.00              # realised reward-to-risk
MIN_FOLD_WIN_RATE = 0.65   # share of walk-forward folds that made money
MAX_CONCENTRATION = 0.45   # share of total R from the single best fold
MIN_RECOVERY = 2.00        # total R vs max drawdown R


def assess(m: dict, n_searched: int, min_trades: int = 100) -> dict:
    """
    Is this a real find, or another artifact of a long search?

    Returns {tier, checks:[(label, ok, detail)], passed, total}. The tiers:

        confirmed  every check passes AND it survived the frozen holdout
        candidate  every check passes, holdout not run yet
        watch      exactly one check fails
        (none)     everything else

    Deliberately strict. The whole failure mode of a search like this is a
    leaderboard full of numbers that look wonderful and mean nothing, so the
    bar is set where a result would actually be worth acting on.
    """
    t = m.get("t_stat", 0.0)
    need_t = noise_ceiling(n_searched) + T_MARGIN
    folds = m.get("fold_folds", 0)

    checks = [
        ("edge survives costs", m.get("expectancy_R", 0) > 0,
         f"{m.get('expectancy_R', 0):+.3f}R per trade"),
        ("beats the search's luck", t >= need_t,
         f"t {t:.2f} vs {need_t:.2f} needed at {n_searched:,} tried"),
        (f"reward/risk >= {MIN_RR:.1f}", m.get("payoff_ratio", 0) >= MIN_RR,
         f"{m.get('payoff_ratio', 0):.2f}"),
        ("enough trades", m.get("trades", 0) >= max(min_trades, 200),
         f"{m.get('trades', 0)} trades"),
        ("works in most folds", m.get("fold_fold_win_rate", 0) >= MIN_FOLD_WIN_RATE,
         f"{m.get('fold_folds_profitable', 0)}/{folds} profitable"),
        ("not one lucky fold", m.get("fold_concentration", 1.0) <= MAX_CONCENTRATION,
         f"best fold is {m.get('fold_concentration', 1.0):.0%} of total R"),
        ("drawdown is survivable", m.get("recovery_factor", 0) >= MIN_RECOVERY,
         f"recovery {m.get('recovery_factor', 0):.2f}x, max DD {m.get('max_dd_R', 0):.1f}R"),
    ]

    passed = sum(1 for _, ok, _ in checks if ok)
    tier = ""
    if passed == len(checks):
        tier = "candidate"
    elif passed == len(checks) - 1:
        tier = "watch"

    # The holdout is reported separately: it is not a check the search can game,
    # because the search cannot see the data.
    if "holdout_expectancy_R" in m:
        held = m["holdout_expectancy_R"] > 0 and m.get("holdout_trades", 0) >= 30
        checks.append(("survived the frozen holdout", held,
                       f"{m.get('holdout_trades', 0)} trades, "
                       f"{m['holdout_expectancy_R']:+.3f}R, "
                       f"RR {m.get('holdout_payoff_ratio', 0):.2f}"))
        if not held:
            tier = ""                       # failed the one test it cannot game
        elif tier == "candidate":
            tier = "confirmed"

    return {"tier": tier, "checks": checks, "passed": passed, "total": len(checks)}


def apply_fold_haircuts(score: float, cons: dict) -> float:
    """
    Anti-curve-fit haircuts, applied after the composite.

    consistency  : reward strategies that worked in most folds, not one.
    concentration: punish "all the R came from a single lucky fold".

    Lives here rather than in the evaluator so that anything which rescores
    stored metrics (tools/rescore.py) reproduces the evaluator exactly.
    """
    if cons.get("folds", 0) < 3 or score <= 0:
        return round(float(score), 5)
    score = score * (0.4 + 0.6 * cons["fold_win_rate"])
    if cons["concentration"] > 0.7:
        score = score * (1.0 - (cons["concentration"] - 0.7))
    return round(float(score), 5)


def fold_consistency(fold_metrics: list[dict]) -> dict:
    """
    Out-of-sample stability across walk-forward folds. A strategy that makes all
    its R in one fold is curve-fit; this is how you catch it.
    """
    prof = [f["total_R"] for f in fold_metrics if f["trades"] > 0]
    if not prof:
        return {"folds": 0, "folds_profitable": 0, "fold_win_rate": 0.0, "worst_fold_R": 0.0,
                "fold_R_std": 0.0, "concentration": 1.0}
    pos = sum(1 for p in prof if p > 0)
    total = sum(prof)
    concentration = (max(prof) / total) if total > 0 else 1.0
    return {
        "folds": len(prof),
        "folds_profitable": pos,
        "fold_win_rate": round(pos / len(prof), 3),
        "worst_fold_R": round(min(prof), 2),
        "fold_R_std": round(float(np.std(prof)), 2),
        "concentration": round(float(min(concentration, 1.0)), 3),
    }


def _span_days(trades: pd.DataFrame) -> float:
    if "entry_time" not in trades.columns or len(trades) < 2:
        return 0.0
    t = pd.to_datetime(trades["entry_time"])
    return max((t.max() - t.min()).total_seconds() / 86400.0, 1e-9)


def _empty() -> dict:
    return {
        "trades": 0, "win_rate": 0.0, "avg_R": 0.0, "avg_win_R": 0.0, "avg_loss_R": 0.0,
        "expectancy_R": 0.0, "total_R": 0.0, "max_dd_R": 0.0, "recovery_factor": 0.0,
        "sharpe": 0.0, "profit_factor": 0.0, "payoff_ratio": 0.0, "t_stat": 0.0,
        "max_losing_streak": 0, "trades_per_year": 0.0,
        "span_days": 0.0, "best_R": 0.0, "worst_R": 0.0, "score": -1.0,
    }
