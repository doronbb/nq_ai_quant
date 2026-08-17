#!/usr/bin/env python3
"""
Rescore every completed strategy under the CURRENT objective.

Changing the composite objective silently invalidates every score already in the
database: the leaderboard would then rank strategies measured on two different
rulers, and — worse — the evolution loop would breed from elites selected by the
old one. This recomputes `score` from the stored metrics, which is exact, since
the metrics themselves are objective-independent.

    python tools/rescore.py                # rescore results/strategies.db
    python tools/rescore.py --dry-run      # show what would change

Stop the searcher first, or run it between restarts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nqq import metrics as M          # noqa: E402
from nqq.registry import Registry     # noqa: E402


def rescore_metrics(m: dict) -> float:
    """Reproduce the evaluator's scoring path from stored metrics alone."""
    if "t_stat" not in m:
        # Written before the t-stat existed. sharpe is the per-trade ratio
        # annualised by sqrt(trades_per_year), so un-annualise and rescale.
        tpy, n = m.get("trades_per_year", 0), m.get("trades", 0)
        m["t_stat"] = round(m.get("sharpe", 0.0) / (tpy ** 0.5) * (n ** 0.5), 3) if tpy > 0 else 0.0
    if "payoff_ratio" not in m:
        # Written before the payoff term existed; derivable from what is stored.
        avg_win, avg_loss = m.get("avg_win_R", 0.0), m.get("avg_loss_R", 0.0)
        if avg_loss < -1e-9:
            m["payoff_ratio"] = round(avg_win / abs(avg_loss), 3)
        else:
            m["payoff_ratio"] = 10.0 if avg_win > 0 else 0.0
    cons = {
        "folds": m.get("fold_folds", 0),
        "fold_win_rate": m.get("fold_fold_win_rate", 0.0),
        "concentration": m.get("fold_concentration", 1.0),
    }
    return M.apply_fold_haircuts(M.composite(m), cons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="results/strategies.db")
    ap.add_argument("--min-trades", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    reg = Registry(args.db)
    rows = reg.con.execute(
        "SELECT fingerprint, score, metrics FROM strategies WHERE status='ok'"
    ).fetchall()

    changed = []
    for r in rows:
        m = json.loads(r["metrics"]) if r["metrics"] else {}
        if not m:
            continue
        new = rescore_metrics(m)
        if abs(new - (r["score"] or 0.0)) > 1e-9:
            changed.append((r["fingerprint"], r["score"], new, m))

    changed.sort(key=lambda x: -x[2])
    print(f"{len(rows)} completed strategies, {len(changed)} change score")
    for fp, old, new, m in changed[:10]:
        print(f"  {fp}  {old:+.4f} -> {new:+.4f}   payoff {m['payoff_ratio']:.2f}  "
              f"exp {m.get('expectancy_R', 0):+.3f}R  {m.get('trades', 0)} trades")

    if args.dry_run:
        print("(dry run, nothing written)")
        return

    for fp, _old, new, m in changed:
        reg.con.execute("UPDATE strategies SET score=?, metrics=? WHERE fingerprint=?",
                        (new, json.dumps(m), fp))
    reg.con.commit()
    reg.close()
    print(f"rewrote {len(changed)} scores in {args.db}")


if __name__ == "__main__":
    main()
