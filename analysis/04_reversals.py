"""Reversals: how often the eventual winner traded cheap late in the window, and whether
buying such "comebacks" pays. Prices are 5-second medians (at least 3 trades)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pm5m.config import TAKER_FEE, WINDOW
from pm5m.plotting import MID, SHADES, save
from pm5m.results import write
from pm5m.sample import Btc, load_markets, load_trades
from pm5m.stats import tstat

BUCKETS = ["5–4 min", "4–3 min", "3–2 min", "2–1 min", "60–30 s", "ostatnie 30 s"]

m = load_markets()
t = load_trades(m, Btc.load(), lags=())
t["up_px"] = np.where(t.outcome_index == 0, t.price, 1 - t.price)
t["b5"] = t.rel // 5
g = t.groupby(["start_ts", "b5"]).up_px.agg(["median", "size"]).reset_index()
g = g[g["size"] >= 3].merge(m[["up"]], left_on="start_ts", right_index=True)
g["left"] = WINDOW - (g.b5 * 5 + 5)
g["winner_px"] = np.where(g.up, g["median"], 1 - g["median"])
g["bucket"] = pd.cut(g.left, [-1, 30, 60, 120, 180, 240, 300], labels=BUCKETS[::-1])
n = g.start_ts.nunique()

low = g.groupby(["start_ts", "bucket"], observed=True).winner_px.min().unstack("bucket")
by_bucket = {f"<= {x:.2f}": {b: 100 * float((low[b] <= x).sum() / n) for b in BUCKETS} for x in (.05, .10, .20, .30, .40)}
low3 = g[g.left <= 180].groupby("start_ts").winner_px.min()
low1 = g[g.left <= 60].groupby("start_ts").winner_px.min()

# Conditional test: first time a side trades near a level, buy 5-10 s later.
g = g.sort_values(["start_ts", "b5"])
g["next_up"] = g.groupby("start_ts")["median"].shift(-1)
sides = []
for side in ("up", "down"):
    px = g["median"] if side == "up" else 1 - g["median"]
    nxt = g.next_up if side == "up" else 1 - g.next_up
    won = g.up if side == "up" else ~g.up
    sides.append(pd.DataFrame({"start_ts": g.start_ts, "bucket": g.bucket, "px": px, "next": nxt,
                               "won": won.astype(float), "side": side, "left": g.left}))
s = pd.concat(sides)
rows = []
for level in (.10, .20, .30, .40):
    hit = s[(s.px >= level - .05) & (s.px < level + .05) & s.next.notna()]
    first = hit.sort_values("left", ascending=False).groupby(["start_ts", "side", "bucket"], observed=True).head(1)
    for b, d in first.groupby("bucket", observed=True):
        price = d.next.clip(.01, .99)
        pnl = d.won - price - TAKER_FEE * price * (1 - price)
        rows.append({"level": level, "bucket": str(b), "n": len(d), "entry_price": price.mean(),
                     "win_rate": d.won.mean(), "pnl_c": 100 * pnl.mean(), "t": tstat(pnl)})
cond = pd.DataFrame(rows)

write("04_reversals", {
    "markets": n, "winner_cheap_by_bucket_pct": by_bucket,
    "winner_cheap_last_3min_pct": {f"<= {x:.2f}": 100 * float((low3 <= x).mean()) for x in (.10, .20, .30)},
    "winner_cheap_last_1min_pct": {f"<= {x:.2f}": 100 * float((low1 <= x).mean()) for x in (.10, .20, .30)},
    "conditional": cond.to_dict("records"), "max_abs_t": float(cond.t.abs().max()),
    "cells_t_above_2": int((cond.t > 2).sum()),
})

fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.5))
for x, color, style in ((.30, SHADES[1], "o-"), (.20, SHADES[3], "s-"), (.10, SHADES[5], "^-")):
    axs[0].plot(BUCKETS, [by_bucket[f"<= {x:.2f}"][b] for b in BUCKETS], style, color=color,
                label=f"zwycięzca kosztował ≤ {x:.2f} USD".replace(".", ","))
axs[0].set(ylabel="% rynków", xlabel="czas do końca okna", title="Jak często zwycięzca był tani")
axs[0].tick_params(axis="x", labelsize=7.5)
axs[0].legend(loc="upper right")
axs[1].plot([0, .5], [0, .5], color=MID, ls=":", lw=1, label="uczciwa cena")
for b, color in zip(BUCKETS, SHADES):
    e = cond[cond.bucket == b]
    axs[1].plot(e.entry_price, e.win_rate, "o-", color=color, label=b, ms=4, lw=1.4)
axs[1].set(xlabel="cena wejścia (5–10 s po dotknięciu)", ylabel="odsetek wygranych", title="Kalibracja w ogonach")
axs[1].legend(fontsize=7, title="czas do końca", title_fontsize=7, loc="upper left")
save(fig, "fig04_reversals.png")
