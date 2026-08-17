"""
Reporting. Zero plotting dependencies — equity curves are hand-rolled inline SVG,
so the HTML opens anywhere and needs no server.

Outputs (all under results/):
    leaderboard.html            live ranking, rewrite every checkpoint
    strategies/<fp>.html        full report for a strategy: equity curve in R,
                                fold table, feature importance, trade log preview
    strategies/<fp>_trades.csv  the full trade log
    strategies/<fp>_genome.json the exact config to reproduce it
"""
from __future__ import annotations

import html
import json
import os
import time

import numpy as np
import pandas as pd

from . import metrics as M
from .genome import describe

CSS = """
:root{--bg:#0f1115;--fg:#e6e6e6;--mut:#8b93a7;--pos:#3ddc84;--neg:#ff6b6b;
--card:#171a21;--line:#262b36;--acc:#5aa9ff}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:14px/1.5 ui-monospace,SFMono-Regular,
Menlo,Consolas,monospace;margin:0;padding:24px}
h1,h2{font-weight:600;letter-spacing:-.01em} h1{font-size:20px;margin:0 0 4px}
h2{font-size:15px;margin:28px 0 10px;color:var(--acc)}
.sub{color:var(--mut);font-size:12px;margin-bottom:20px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--line);
white-space:nowrap}
th{color:var(--mut);font-weight:500;text-align:right;position:sticky;top:0;
background:var(--bg)}
td:first-child,th:first-child{text-align:left}
tr:hover td{background:#1c202a}
.pos{color:var(--pos)} .neg{color:var(--neg)} .mut{color:var(--mut)}
.cards{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:10px 14px;min-width:120px}
.card .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.card .v{font-size:18px;margin-top:2px}
a{color:var(--acc);text-decoration:none} a:hover{text-decoration:underline}
code{background:var(--card);padding:1px 5px;border-radius:4px;font-size:12px}
.bar{height:8px;background:var(--acc);border-radius:2px;display:inline-block}
.warn{background:#2b2113;border:1px solid #5c4415;border-radius:8px;padding:10px 14px;
color:#ffcf7a;margin:12px 0;font-size:12.5px}

/* ---- a find worth looking at -------------------------------------------- */
.hit{background:linear-gradient(90deg,#12301f,#152036 60%,var(--card));
border:1px solid #2f7d52;border-left:4px solid var(--pos);border-radius:8px;
padding:14px 18px;margin:14px 0}
.hit h3{margin:0 0 6px;font-size:15px;color:var(--pos);letter-spacing:-.01em}
.hit .why{color:#b9c6bd;font-size:12.5px;margin:2px 0}
.hit code{background:#0d1a12}
tr.tier-confirmed td{background:rgba(61,220,132,.13)}
tr.tier-confirmed td:first-child{box-shadow:inset 3px 0 var(--pos)}
tr.tier-candidate td{background:rgba(61,220,132,.07)}
tr.tier-candidate td:first-child{box-shadow:inset 3px 0 #2f7d52}
tr.tier-watch td:first-child{box-shadow:inset 3px 0 #5c4415}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10.5px;
letter-spacing:.04em;text-transform:uppercase}
.badge-confirmed{background:var(--pos);color:#07130c;font-weight:600}
.badge-candidate{background:#22432f;color:var(--pos);border:1px solid #2f7d52}
.badge-watch{background:#2b2113;color:#ffcf7a;border:1px solid #5c4415}
.nofind{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--mut);
border-radius:8px;padding:14px 18px;margin:14px 0}
.nofind h3{margin:0 0 6px;font-size:15px;color:var(--fg);letter-spacing:-.01em}
.nofind .why{color:var(--mut);font-size:12.5px;margin:2px 0}
.chk{list-style:none;padding:0;margin:8px 0}
.chk li{padding:3px 0;font-size:12.5px}
.chk .ok:before{content:"PASS  ";color:var(--pos);font-weight:600}
.chk .no:before{content:"FAIL  ";color:var(--neg);font-weight:600}
.chk .d{color:var(--mut)}
"""


def _sign(v, fmt="{:+.3f}"):
    cls = "pos" if v > 0 else ("neg" if v < 0 else "mut")
    return f'<span class="{cls}">{fmt.format(v)}</span>'


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style>{body}")


# --------------------------------------------------------------------------
# equity curve (inline SVG, no matplotlib)
# --------------------------------------------------------------------------

def equity_svg(R: np.ndarray, w=900, h=260) -> str:
    if len(R) == 0:
        return "<p class=mut>no trades</p>"
    eq = np.concatenate([[0.0], np.cumsum(R)])
    peak = np.maximum.accumulate(eq)
    lo, hi = float(min(eq.min(), 0)), float(max(eq.max(), 0.001))
    pad = (hi - lo) * 0.08 + 1e-9
    lo, hi = lo - pad, hi + pad
    n = len(eq)

    def X(i): return 46 + (w - 60) * i / max(n - 1, 1)
    def Y(v): return 20 + (h - 50) * (1 - (v - lo) / (hi - lo))

    step = max(1, n // 1400)
    pts = " ".join(f"{X(i):.1f},{Y(eq[i]):.1f}" for i in range(0, n, step))
    dd_pts = " ".join(f"{X(i):.1f},{Y(peak[i]):.1f}" for i in range(0, n, step))

    zero = Y(0)
    ticks = ""
    for frac in (0, .25, .5, .75, 1):
        v = lo + (hi - lo) * frac
        y = Y(v)
        ticks += (f'<line x1=46 y1={y:.1f} x2={w - 14} y2={y:.1f} stroke="#262b36"/>'
                  f'<text x=40 y={y + 4:.1f} fill="#8b93a7" font-size=10 '
                  f'text-anchor="end">{v:+.0f}R</text>')

    color = "#3ddc84" if eq[-1] >= 0 else "#ff6b6b"
    area = f"{X(0):.1f},{zero:.1f} " + pts + f" {X(n - 1):.1f},{zero:.1f}"
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
        f'xmlns="http://www.w3.org/2000/svg">{ticks}'
        f'<polygon points="{area}" fill="{color}" opacity=".10"/>'
        f'<polyline points="{dd_pts}" fill=none stroke="#4a5164" stroke-width="1" '
        f'stroke-dasharray="3 3"/>'
        f'<polyline points="{pts}" fill=none stroke="{color}" stroke-width="1.8"/>'
        f'<line x1=46 y1={zero:.1f} x2={w - 14} y2={zero:.1f} stroke="#8b93a7" '
        f'stroke-width=".8"/>'
        f'<text x=46 y={h - 8} fill="#8b93a7" font-size=10>trade 1</text>'
        f'<text x={w - 14} y={h - 8} fill="#8b93a7" font-size=10 text-anchor="end">'
        f'trade {n - 1}</text></svg>'
    )


# --------------------------------------------------------------------------
# leaderboard
# --------------------------------------------------------------------------

def write_leaderboard(cfg: dict, reg, extra: dict | None = None, top_n: int = 60) -> str:
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "leaderboard.html")
    st = reg.stats()
    rows = reg.top(top_n)

    cards = [
        ("strategies tried", f"{st['n'] or 0:,}"),
        ("scored", f"{st['ok'] or 0:,}"),
        ("rejected", f"{st['rejected'] or 0:,}"),
        ("best score", f"{(st['best'] or 0):.4f}"),
        ("generation", f"{st['gen'] or 0}"),
        ("avg eval", f"{(st['avg_secs'] or 0):.1f}s"),
    ]
    for k, v in (extra or {}).items():
        cards.append((k.replace("_", " "), v))
    card_html = "".join(
        f'<div class=card><div class=k>{html.escape(str(k))}</div>'
        f'<div class=v>{html.escape(str(v))}</div></div>' for k, v in cards)

    head = ("rank fingerprint verdict score trades win% expR RR totalR maxDD sharpe PF "
            "folds+ gen origin strategy").split()
    th = "".join(f"<th>{h}</th>" for h in head)
    body = ""
    n_searched = (st["n"] or 0)
    min_trades = int(cfg["search"].get("min_trades", 100))
    n_checks = len(M.assess({}, n_searched, min_trades)["checks"])
    finds = []
    for i, r in enumerate(rows, 1):
        m = r["metrics"]
        link = f'strategies/{r["fingerprint"]}.html'
        exists = os.path.exists(os.path.join(out_dir, link))
        fp = (f'<a href="{link}">{r["fingerprint"]}</a>' if exists else r["fingerprint"])
        a = M.assess(m, n_searched, min_trades)
        tier = a["tier"]
        if tier in ("confirmed", "candidate"):
            finds.append((r, a, link if exists else ""))
        badge = (f'<span class="badge badge-{tier}">{tier}</span>' if tier
                 else f'<span class=mut>{a["passed"]}/{a["total"]}</span>')
        body += (
            f'<tr class="tier-{tier}"><td>{i}</td><td>{fp}</td><td>{badge}</td>'
            f"<td>{_sign(r['score'], '{:+.4f}')}</td>"
            f"<td>{m.get('trades', 0)}</td>"
            f"<td>{m.get('win_rate', 0) * 100:.1f}</td>"
            f"<td>{_sign(m.get('expectancy_R', 0))}</td>"
            f"<td>{m.get('payoff_ratio', 0):.2f}</td>"
            f"<td>{_sign(m.get('total_R', 0), '{:+.1f}')}</td>"
            f"<td class=mut>{m.get('max_dd_R', 0):.1f}</td>"
            f"<td>{m.get('sharpe', 0):.2f}</td>"
            f"<td>{m.get('profit_factor', 0):.2f}</td>"
            f"<td class=mut>{m.get('fold_folds_profitable', 0)}/{m.get('fold_folds', 0)}</td>"
            f"<td class=mut>{r['generation']}</td>"
            f"<td class=mut>{r['origin']}</td>"
            f"<td class=mut>{html.escape(describe(r['genome']))}</td></tr>")

    hit_html = ""
    if not finds and rows:
        # Nothing has cleared the bar. Say so explicitly and show how close the
        # nearest miss is — otherwise an empty banner area is indistinguishable
        # from a broken one, and the honest "no find yet" reads as no answer.
        best_r, best_a = max(
            ((r, M.assess(r["metrics"], n_searched, min_trades)) for r in rows),
            key=lambda x: (x[1]["passed"], x[0]["score"]))
        bm = best_r["metrics"]
        missing = "".join(
            f'<li class=no>{html.escape(label)} <span class=d>— {html.escape(detail)}</span></li>'
            for label, ok, detail in best_a["checks"] if not ok)
        hit_html = f"""
<div class=nofind>
  <h3>No find yet — nothing clears all {n_checks} checks</h3>
  <div class=why>Closest is <b>{best_r['fingerprint']}</b> at
    {best_a['passed']}/{best_a['total']}, {bm.get('trades', 0)} trades,
    RR {bm.get('payoff_ratio', 0):.2f}, t-stat {bm.get('t_stat', 0):.2f}. Still failing:</div>
  <ul class=chk>{missing}</ul>
  <div class=why>The t-stat bar is <b>{M.noise_ceiling(n_searched) + M.T_MARGIN:.2f}</b> and
    rises as the search runs — {n_searched:,} genomes tried means {n_searched:,} chances
    for noise to look like an edge. A green banner appears here the moment
    something passes everything.</div>
</div>"""

    for r, a, link in finds[:5]:
        m = r["metrics"]
        confirmed = a["tier"] == "confirmed"
        name = (f'<a href="{link}">{r["fingerprint"]}</a>' if link else r["fingerprint"])
        next_step = (
            "Holdout confirmed. This is as far as the search can take it — "
            "everything from here is a decision about real money, not a backtest."
            if confirmed else
            f"Not yet tested on the frozen holdout. Run "
            f"<code>python tools/holdout.py {r['fingerprint']}</code> — and note that "
            f"the holdout is spent once you start picking strategies by it.")
        hit_html += f"""
<div class=hit>
  <h3>{'CONFIRMED — survived the frozen holdout' if confirmed
       else 'CANDIDATE — clears every statistical bar'}: {name}</h3>
  <div class=why>{html.escape(describe(r['genome']))}</div>
  <div class=why><b>{m.get('trades', 0)} trades</b> ·
    expectancy <b>{m.get('expectancy_R', 0):+.3f}R</b> ·
    reward/risk <b>{m.get('payoff_ratio', 0):.2f}</b> ·
    t-stat <b>{m.get('t_stat', 0):.2f}</b> vs
    {M.noise_ceiling(n_searched) + M.T_MARGIN:.2f} needed ·
    {m.get('fold_folds_profitable', 0)}/{m.get('fold_folds', 0)} folds profitable ·
    total <b>{m.get('total_R', 0):+.1f}R</b> on {m.get('max_dd_R', 0):.1f}R drawdown</div>
  <div class=why>{next_step}</div>
</div>"""

    warn = ""
    if cfg["data"]["source"] == "yfinance":
        warn = ('<div class=warn>Data source is <code>yfinance</code>: ~60 days of gappy '
                'intraday NQ. Fine for checking the plumbing, not for drawing conclusions. '
                'Switch <code>data.source</code> to <code>csv</code> or <code>databento</code> '
                'before you trust any of this.</div>')
    if cfg["data"].get("synthetic_warning"):
        warn += ('<div class=warn>Running on SYNTHETIC sample data. Every number below is '
                 'meaningless except as a proof that the pipeline works.</div>')

    html_doc = _page("NQ strategy leaderboard", f"""
<h1>NQ strategy search — leaderboard</h1>
<div class=sub>{time.strftime('%Y-%m-%d %H:%M:%S')} · {html.escape(cfg['data']['source'])} ·
{html.escape(cfg['data']['timeframe'])} · objective: composite (0.34 expectancy /
0.22 sharpe / 0.22 recovery / 0.22 reward-risk)</div>
{warn}
{hit_html}
<div class=cards>{card_html}</div>
<h2>Top {len(rows)} by composite score</h2>
<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>
<p class=sub>The <b>verdict</b> column scores each strategy against
{n_checks} checks — positive edge after costs,
a t-stat above what {n_searched:,} tried genomes produce by luck alone
(currently {M.noise_ceiling(n_searched) + M.T_MARGIN:.2f}), reward/risk over
{M.MIN_RR:.1f}, enough trades, profitable in most folds, not carried by one lucky
fold, and a survivable drawdown. <b>candidate</b> passes all of them;
<b>confirmed</b> also survived the frozen holdout; a bare number is how many it
passed. Anything unhighlighted has not cleared the bar, whatever its score says.</p>
<p class=sub>Every strategy above is stored in <code>results/strategies.db</code>.
The searcher never re-tests a fingerprint, so this list only grows with genuinely
new experiments.</p>""")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html_doc)
    return path


# --------------------------------------------------------------------------
# per-strategy report
# --------------------------------------------------------------------------

def write_strategy_report(cfg: dict, reg, fingerprint: str, res: dict) -> str:
    out_dir = os.path.join(cfg["output"]["dir"], "strategies")
    os.makedirs(out_dir, exist_ok=True)
    rec = reg.get(fingerprint) or {}
    g = rec.get("genome", {})
    m = res["metrics"]
    trades: pd.DataFrame = res["trades"]

    csv_path = os.path.join(out_dir, f"{fingerprint}_trades.csv")
    if len(trades):
        trades.to_csv(csv_path, index=False)
    with open(os.path.join(out_dir, f"{fingerprint}_genome.json"), "w", encoding="utf-8") as fh:
        json.dump({"fingerprint": fingerprint, "genome": g, "metrics": m}, fh, indent=2,
                  default=str)

    R_arr = trades["R"].to_numpy(np.float64) if len(trades) else np.array([])
    cards = [
        ("score", f"{m['score']:+.4f}"), ("trades", m["trades"]),
        ("win rate", f"{m['win_rate'] * 100:.1f}%"),
        ("expectancy", f"{m['expectancy_R']:+.3f}R"),
        ("total", f"{m['total_R']:+.1f}R"), ("max DD", f"{m['max_dd_R']:.1f}R"),
        ("sharpe", f"{m['sharpe']:.2f}"), ("profit factor", f"{m['profit_factor']:.2f}"),
        ("reward/risk", f"{m.get('payoff_ratio', 0):.2f}"),
        ("avg win", f"{m['avg_win_R']:+.2f}R"), ("avg loss", f"{m['avg_loss_R']:+.2f}R"),
        ("worst streak", m["max_losing_streak"]),
        ("recovery", f"{m['recovery_factor']:.2f}"),
    ]
    card_html = "".join(f'<div class=card><div class=k>{k}</div><div class=v>{v}</div></div>'
                        for k, v in cards)

    # verdict: the same bar the leaderboard highlights against, itemised
    st = reg.stats()
    a = M.assess(m, st["n"] or 0, int(cfg["search"].get("min_trades", 100)))
    chk = "".join(
        f'<li class="{"ok" if ok else "no"}">{html.escape(label)} '
        f'<span class=d>— {html.escape(detail)}</span></li>'
        for label, ok, detail in a["checks"])
    if a["tier"]:
        verdict = (f'<div class=hit><h3>{a["tier"].upper()} — passed '
                   f'{a["passed"]}/{a["total"]} checks</h3>'
                   f'<ul class=chk>{chk}</ul></div>')
    else:
        verdict = (f'<h2>Verdict — passed {a["passed"]}/{a["total"]}</h2>'
                   f'<ul class=chk>{chk}</ul>')

    # folds
    fh_rows = ""
    for i, f in enumerate(res["fold_metrics"]):
        fh_rows += (f"<tr><td>{i}</td><td>{f['trades']}</td>"
                    f"<td>{f['win_rate'] * 100:.1f}</td>"
                    f"<td>{_sign(f['expectancy_R'])}</td>"
                    f"<td>{_sign(f['total_R'], '{:+.1f}')}</td>"
                    f"<td class=mut>{f['max_dd_R']:.1f}</td></tr>")

    # importance
    imp = list(res["importance"].items())[:30]
    imp_rows = ""
    mx = max((v for _, v in imp), default=1e-9)
    for k, v in imp:
        w = int(220 * v / mx)
        imp_rows += (f"<tr><td>{html.escape(k)}</td><td>{v * 100:.2f}%</td>"
                     f"<td style='text-align:left'><span class=bar style='width:{w}px'>"
                     f"</span></td></tr>")

    # trade log preview
    tl = ""
    if len(trades):
        prev = pd.concat([trades.head(15), trades.tail(15)]).drop_duplicates()
        cols = ["entry_time", "direction", "entry_price", "stop_price", "target_price",
                "exit_price", "bars_held", "confidence", "R"]
        tl += "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
        for _, row in prev.iterrows():
            tl += "<tr>" + "".join(
                f"<td>{_sign(row[c], '{:+.2f}') if c == 'R' else html.escape(str(row[c]))}</td>"
                for c in cols) + "</tr>"

    doc = _page(f"strategy {fingerprint}", f"""
<h1>Strategy {fingerprint}</h1>
<div class=sub>{html.escape(describe(g)) if g else ''}<br>
origin: {rec.get('origin', '?')} · generation {rec.get('generation', 0)} ·
folds {res['n_folds']} · <a href="../leaderboard.html">back to leaderboard</a></div>
{verdict}
<div class=cards>{card_html}</div>

<h2>Equity curve (R-multiples, account-size agnostic)</h2>
{equity_svg(R_arr)}
<p class=sub>Solid line: cumulative R. Dashed line: running peak — the gap between
them is drawdown in R.</p>

<h2>Out-of-sample folds</h2>
<table><thead><tr><th>fold</th><th>trades</th><th>win%</th><th>expectancy</th>
<th>total R</th><th>max DD</th></tr></thead><tbody>{fh_rows}</tbody></table>
<p class=sub>fold win rate {m.get('fold_fold_win_rate', 0):.0%} ·
worst fold {m.get('fold_worst_fold_R', 0):+.1f}R ·
concentration {m.get('fold_concentration', 0):.2f}
(1.00 = all profit came from a single fold, which means curve fitting)</p>

<h2>Feature importance</h2>
<table><thead><tr><th>feature</th><th>share</th><th></th></tr></thead>
<tbody>{imp_rows}</tbody></table>

<h2>Trade log <span class=mut>(first/last 15 — full CSV:
<a href="{fingerprint}_trades.csv">{fingerprint}_trades.csv</a>)</span></h2>
<table>{tl}</table>

<h2>Genome</h2>
<pre style="background:#171a21;padding:14px;border-radius:8px;overflow:auto;font-size:12px">
{html.escape(json.dumps(g, indent=2, sort_keys=True))}</pre>""")

    path = os.path.join(out_dir, f"{fingerprint}.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path
