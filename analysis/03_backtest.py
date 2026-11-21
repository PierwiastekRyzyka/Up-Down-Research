"""Naive vs realistic backtest of trading on the model-market gap.

Naive: signal and fill both at the market price over [t-10, t) (stale relative to BTC).
Realistic: signal at the fresh price over [t-3, t), fill at trades 1-6 s later, plus fees
and 1 cent of slippage.
"""
import matplotlib.pyplot as plt
import numpy as np

from pm5m.config import TAKER_FEE
from pm5m.plotting import DARK, INK, MID, save
from pm5m.results import write
from pm5m.sample import load_markets, load_panel
from pm5m.stats import tstat

m = load_markets()
p = load_panel(m)


def backtest(g, threshold, slippage=0.0, realistic=True, fee=TAKER_FEE, t_max=240):
    g = g[g.t <= t_max].sort_values(["start_ts", "t"])
    signal = g.p_market_fresh if realistic else g.p_market
    fill = g.p_exec if realistic else g.p_market
    ok = signal.notna() & fill.notna()
    g, signal, fill = g[ok], signal[ok], fill[ok]
    edge = g.p_model - signal
    hit = np.abs(edge) > threshold
    s = g[hit].assign(edge=edge[hit], fill=fill[hit]).groupby("start_ts").head(1)
    if len(s) < 3:
        return {"n": len(s)}
    buy_up = (s.edge > 0).to_numpy()
    price = np.clip(np.where(buy_up, s.fill, 1 - s.fill) + slippage, .01, .99)
    won = np.where(buy_up, s.y, 1 - s.y)
    pnl = won - price - fee * price * (1 - price)
    return {"n": len(s), "pnl_c": 100 * pnl.mean(), "t": tstat(pnl), "hit_rate": won.mean(), "price": price.mean()}


res = {f"{th:.2f}": {"naive": backtest(p, th, realistic=False), "realistic": backtest(p, th, .01),
                     "realistic_gross": backtest(p, th, fee=0),
                     "realistic_H1": backtest(p[p.half == "H1"], th, .01), "realistic_H2": backtest(p[p.half == "H2"], th, .01)}
       for th in (.05, .10, .15, .20, .30)}
write("03_backtest", res)

grid = np.arange(.03, .31, .01)
fig, ax = plt.subplots(figsize=(6.6, 3.5))
ax.axhline(0, color="#333333", lw=.8)
ax.plot(grid, [backtest(p, x, realistic=False)["pnl_c"] for x in grid], ":", color=MID, lw=2, label="naiwny (cena sprzed sygnału)")
ax.plot(grid, [backtest(p, x, .01)["pnl_c"] for x in grid], "o-", ms=3, color=INK, label="realistyczny: po opłacie i poślizgu 1 ¢")
ax.plot(grid, [backtest(p, x, fee=0)["pnl_c"] for x in grid], "s--", ms=3, color=DARK, label="realistyczny brutto (bez opłaty)")
ax.set(xlabel="próg |model − cena rynkowa|", ylabel="średni PnL [¢ na akcję]", title="Backtest: bias nieaktualnej ceny zawyża wynik")
ax.legend(loc="upper left")
save(fig, "fig03_backtest.png")
