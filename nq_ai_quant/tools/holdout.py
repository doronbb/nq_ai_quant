#!/usr/bin/env python3
"""
Test a strategy on the frozen holdout — data the search has never seen.

    python tools/holdout.py <fingerprint>
    python tools/holdout.py --top 5          # the current leaderboard's best N

This is the only honest verdict available. The walk-forward folds stopped being
out-of-sample the moment the search selected on them: with thousands of genomes
tried, the top of the leaderboard is chosen partly on luck, and its fold results
are contaminated by that selection. The holdout is not, because nothing in the
search loop can reach it (`Evaluator.n_searchable` truncates every fold).

Read the output as pass/fail on the strategy, not as a number to optimise. The
moment you start picking strategies by their holdout score, the holdout is
burned too, and you have no untouched data left.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml                                    # noqa: E402

from nqq.backtest import Evaluator             # noqa: E402
from nqq.data import load_bars                 # noqa: E402
from nqq.genome import describe                # noqa: E402
from nqq.registry import Registry              # noqa: E402


def _line(label: str, insample: dict, out: dict, key: str, fmt: str = "{:+.3f}") -> str:
    a = insample.get(key, 0) or 0
    b = out.get(key, 0) or 0
    return f"  {label:<18}{fmt.format(a):>12}{fmt.format(b):>12}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fingerprint", nargs="?")
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if not cfg["walkforward"].get("holdout_months"):
        print("walkforward.holdout_months is 0 — there is no frozen data to test on.")
        sys.exit(1)

    reg = Registry(os.path.join(cfg["output"]["dir"], "strategies.db"))
    if args.top:
        targets = [r["fingerprint"] for r in reg.top(args.top)]
    elif args.fingerprint:
        targets = [args.fingerprint]
    else:
        ap.error("give a fingerprint or --top N")

    bars = load_bars(cfg["data"], cfg["data"]["timeframe"],
                     cache_dir=os.path.join(cfg["output"]["dir"], "cache"))
    ev = Evaluator(bars, cfg)
    frozen_from = ev.index[ev.n_searchable].date()
    frozen_to = ev.index[-1].date()

    print(f"\nHoldout: {frozen_from} -> {frozen_to} "
          f"({len(bars) - ev.n_searchable} bars the search never saw)\n")

    for fp in targets:
        rec = reg.get(fp)
        if not rec:
            print(f"{fp}: not in the registry")
            continue
        ins = rec["metrics"]
        res = ev.evaluate(rec["genome"], collect_trades=True, holdout=True)
        out = res["metrics"]

        print(f"=== {fp}  {describe(rec['genome'])}")
        if out.get("reject_reason"):
            print(f"  could not evaluate: {out['reject_reason']}\n")
            continue
        print(f"  {'':<18}{'search':>12}{'HOLDOUT':>12}")
        print(_line("trades", ins, out, "trades", "{:.0f}"))
        print(_line("win rate", ins, out, "win_rate", "{:.1%}"))
        print(_line("expectancy R", ins, out, "expectancy_R"))
        print(_line("reward/risk", ins, out, "payoff_ratio", "{:.2f}"))
        print(_line("total R", ins, out, "total_R", "{:+.1f}"))
        print(_line("max DD R", ins, out, "max_dd_R", "{:.1f}"))
        print(_line("sharpe", ins, out, "sharpe", "{:.2f}"))
        print(_line("t-stat", ins, out, "t_stat", "{:.2f}"))

        # Persist onto the strategy so the leaderboard can promote it from
        # "candidate" to "confirmed" — the one verdict the search cannot game.
        ins.update({
            "holdout_trades": out.get("trades", 0),
            "holdout_expectancy_R": out.get("expectancy_R", 0.0),
            "holdout_payoff_ratio": out.get("payoff_ratio", 0.0),
            "holdout_total_R": out.get("total_R", 0.0),
            "holdout_t_stat": out.get("t_stat", 0.0),
            "holdout_from": str(frozen_from),
            "holdout_to": str(frozen_to),
        })
        reg.con.execute("UPDATE strategies SET metrics=? WHERE fingerprint=?",
                        (json.dumps(ins), fp))
        reg.con.commit()

        exp_in, exp_out = ins.get("expectancy_R", 0), out.get("expectancy_R", 0)
        if out.get("trades", 0) < 30:
            verdict = "INCONCLUSIVE — too few holdout trades to say anything"
        elif exp_out <= 0:
            verdict = "FAILED — the edge does not exist outside the search window"
        elif exp_out < exp_in * 0.5:
            verdict = "WEAK — survives, but at less than half the searched edge"
        else:
            verdict = "HELD UP"
        print(f"  -> {verdict}\n")

    reg.close()


if __name__ == "__main__":
    main()
