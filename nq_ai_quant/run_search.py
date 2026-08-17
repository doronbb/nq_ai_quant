#!/usr/bin/env python3
"""
Autonomous NQ strategy search. Start it and walk away.

    python run_search.py                      # run forever
    python run_search.py --iters 50           # run 50 new strategies, then stop
    python run_search.py --status             # what has been found so far
    python run_search.py --report <fingerprint>   # rebuild one strategy's report
    python run_search.py --reset              # wipe the registry (asks first)

Ctrl+C once = finish the current strategy, write reports, exit cleanly.
Restarting always resumes: nothing already tried is ever tried again.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml                                                  # noqa: E402

from nqq import report as R                                  # noqa: E402
from nqq.data import load_bars                               # noqa: E402
from nqq.registry import Registry                            # noqa: E402
from nqq.search import SearchLoop                            # noqa: E402


def setup_logging(cfg: dict, verbose: bool):
    os.makedirs(cfg["output"]["dir"], exist_ok=True)
    fmt = "%(asctime)s %(levelname).1s %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    lf = cfg["output"].get("log_file")
    if lf:
        os.makedirs(os.path.dirname(lf) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(lf, encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format=fmt, datefmt="%H:%M:%S", handlers=handlers)
    logging.getLogger("nqq").setLevel(logging.DEBUG if verbose else logging.INFO)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def cmd_status(cfg, reg):
    st = reg.stats()
    print(json.dumps(st, indent=2, default=str))
    print("\nTop 15:")
    print(f"{'rank':>4} {'fingerprint':<22}{'score':>9}{'trades':>8}{'win%':>7}"
          f"{'expR':>8}{'RR':>6}{'totR':>9}{'DD':>7}{'shrp':>7}  strategy")
    from nqq.genome import describe
    for i, r in enumerate(reg.top(15), 1):
        m = r["metrics"]
        print(f"{i:>4} {r['fingerprint']:<22}{r['score']:>+9.4f}{m.get('trades', 0):>8}"
              f"{m.get('win_rate', 0) * 100:>7.1f}{m.get('expectancy_R', 0):>+8.3f}"
              f"{m.get('payoff_ratio', 0):>6.2f}"
              f"{m.get('total_R', 0):>+9.1f}{m.get('max_dd_R', 0):>7.1f}"
              f"{m.get('sharpe', 0):>7.2f}  {describe(r['genome'])}")
    print(f"\nLeaderboard: {os.path.join(cfg['output']['dir'], 'leaderboard.html')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--iters", type=int, default=0, help="0 = run forever")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--report", metavar="FINGERPRINT")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="force a fresh data pull")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg, args.verbose)
    log = logging.getLogger("nqq")

    db_path = os.path.join(cfg["output"]["dir"], "strategies.db")

    if args.reset:
        if input(f"Delete {db_path} and lose all search history? [y/N] ").strip().lower() == "y":
            for suffix in ("", "-wal", "-shm"):
                p = db_path + suffix
                if os.path.exists(p):
                    os.remove(p)
            print("registry cleared")
        return

    reg = Registry(db_path)

    if args.status:
        cmd_status(cfg, reg)
        return

    bars = load_bars(cfg["data"], cfg["data"]["timeframe"],
                     cache_dir=os.path.join(cfg["output"]["dir"], "cache"),
                     use_cache=not args.no_cache)

    need = cfg["walkforward"]["train_bars"] + cfg["walkforward"]["test_bars"] * 3
    if len(bars) < need:
        log.error("Only %d bars but the walk-forward config needs >= %d. "
                  "Either load more history or lower walkforward.train_bars / test_bars.",
                  len(bars), need)
        sys.exit(1)

    if args.report:
        loop = SearchLoop(cfg, bars, reg)
        rec = reg.get(args.report)
        if not rec:
            log.error("unknown fingerprint %s", args.report)
            sys.exit(1)
        res = loop.ev.evaluate(rec["genome"])
        print("report ->", R.write_strategy_report(cfg, reg, args.report, res))
        print("leaderboard ->", R.write_leaderboard(cfg, reg))
        return

    reg.start_run(cfg, note=f"timeframe={cfg['data']['timeframe']}")
    loop = SearchLoop(cfg, bars, reg)
    log.info("searching%s. Ctrl+C to stop cleanly.",
             " forever" if args.iters == 0 else f" for {args.iters} strategies")
    try:
        loop.run(max_iters=args.iters)
    except KeyboardInterrupt:
        log.warning("hard stop")
    finally:
        print("\nLeaderboard:", os.path.abspath(
            os.path.join(cfg["output"]["dir"], "leaderboard.html")))
        reg.close()


if __name__ == "__main__":
    main()
