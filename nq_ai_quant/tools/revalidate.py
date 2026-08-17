#!/usr/bin/env python3
"""
Re-measure the leaderboard after a change to the TEST WINDOW.

Changing `walkforward.max_folds`, `test_bars`, `train_bars` or `holdout_months`
means every stored score was measured on a different period. Those numbers
cannot be rescored arithmetically the way an objective change can — the trades
themselves are different — so they have to be re-run or retired.

This does both:
  * every completed strategy is marked `legacy` (kept for dedupe, dropped from
    the leaderboard and the breeding pool — a stale score has no business
    selecting parents);
  * the previous best N are re-evaluated on the new window and restored to `ok`.

    python tools/revalidate.py --top 40

Stop the searcher first.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml                                    # noqa: E402

from nqq.backtest import Evaluator             # noqa: E402
from nqq.data import load_bars                 # noqa: E402
from nqq.registry import Registry              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    reg = Registry(os.path.join(cfg["output"]["dir"], "strategies.db"))
    keep = reg.top(args.top)
    n_ok = reg.con.execute("select count(*) from strategies where status='ok'").fetchone()[0]
    print(f"{n_ok} completed strategies measured on the OLD window; "
          f"re-running the best {len(keep)} on the new one.")

    reg.con.execute("UPDATE strategies SET status='legacy' WHERE status='ok'")
    reg.con.commit()

    bars = load_bars(cfg["data"], cfg["data"]["timeframe"],
                     cache_dir=os.path.join(cfg["output"]["dir"], "cache"))
    ev = Evaluator(bars, cfg)

    restored = 0
    for i, rec in enumerate(keep, 1):
        fp, old = rec["fingerprint"], rec["score"]
        t0 = time.time()
        try:
            res = ev.evaluate(rec["genome"], collect_trades=False)
        except Exception as e:
            print(f"{i:>3} {fp}  eval failed: {e}")
            continue
        m = res["metrics"]
        if m.get("reject_reason") or m["trades"] < cfg["search"]["min_trades"] \
                or res["n_folds"] < cfg["search"]["min_folds"]:
            reason = m.get("reject_reason") or f"only {m['trades']} trades on the new window"
            reg.con.execute(
                "UPDATE strategies SET status='rejected', score=?, metrics=?, error=?"
                " WHERE fingerprint=?", (-1e9, json.dumps(m), reason, fp))
            print(f"{i:>3} {fp}  {old:+.4f} -> rejected ({reason})")
        else:
            reg.con.execute(
                "UPDATE strategies SET status='ok', score=?, metrics=?, eval_secs=?"
                " WHERE fingerprint=?",
                (m["score"], json.dumps(m), time.time() - t0, fp))
            restored += 1
            print(f"{i:>3} {fp}  {old:+.4f} -> {m['score']:+.4f}  "
                  f"{m['trades']:>4} trades  RR {m.get('payoff_ratio', 0):.2f}  "
                  f"t {m.get('t_stat', 0):.2f}")
        reg.con.commit()

    print(f"\n{restored} strategies restored to the leaderboard, "
          f"{n_ok - restored} retired to legacy (still deduped, never re-tried).")
    reg.close()


if __name__ == "__main__":
    main()
