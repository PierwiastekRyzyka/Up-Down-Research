"""Who wins persistently. Wallet PnL is measured per market and tested out of sample:
wallets selected on the first half are evaluated on the second."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pm5m.plotting import INK, MID, save
from pm5m.results import write
from pm5m.sample import Btc, load_markets, load_trades

m = load_markets()
t = load_trades(m, Btc.load(), lags=(0,))

overall = {"records_in_window": len(t), "wallets": int(t.wallet.nunique()), "markets": int(t.start_ts.nunique()),
           "shares": float(t["size"].sum()), "notional_usd": float(t.notional.sum()),
           "taker_pnl_gross_usd": float(t.pnl_gross.sum()), "fees_usd": float(t.fee.sum()),
           "taker_pnl_gross_c_per_share": float(100 * t.pnl_gross.sum() / t["size"].sum()),
           "fee_c_per_share": float(100 * t.fee.sum() / t["size"].sum())}

per_market = t.groupby(["wallet", "half", "start_ts"]).agg(pnl=("pnl", "sum"), notional=("notional", "sum")).reset_index()
g = per_market.groupby(["wallet", "half"])
w = pd.DataFrame({"markets": g.size(), "pnl": g.pnl.sum(), "notional": g.notional.sum(), "mu": g.pnl.mean(), "sd": g.pnl.std(ddof=1)})
w["t"] = w.mu / (w.sd / np.sqrt(w.markets))
w = w.unstack("half")

active = w[(w["markets"]["H1"] >= 30) & (w["markets"]["H2"] >= 10)]
def summary(sel):
    return {"wallets": len(sel), "positive_H2_pct": float(100 * (sel["pnl"]["H2"] > 0).mean()),
            "roi_H2_pct": float(100 * sel["pnl"]["H2"].sum() / sel["notional"]["H2"].sum()),
            "pnl_H2_usd": float(sel["pnl"]["H2"].sum())}
persistence = {f"t_H1>{k}": summary(active[active["t"]["H1"] > k]) for k in (2, 3, 4)}
persistence["all_active"] = summary(active)
roi1 = active["pnl"]["H1"] / active["notional"]["H1"]
roi2 = active["pnl"]["H2"] / active["notional"]["H2"]
top_decile = active[pd.qcut(roi1.rank(method="first"), 10, labels=False) == 9]
persistence["top_roi_decile_H1"] = {"wallets": len(top_decile),
                                    "roi_H2_pct": float(100 * top_decile["pnl"]["H2"].sum() / top_decile["notional"]["H2"].sum())}
persistence["spearman_roi"] = float(spearmanr(roi1, roi2)[0])

stable = w[(w["t"]["H1"] > 5) & (w["t"]["H2"] > 5) & (w["markets"]["H1"] >= 20) & (w["markets"]["H2"] >= 20)].index
s = t[t.wallet.isin(stable)]
buy = s.side == "BUY"
right = np.where(buy, s.won, 1 - s.won)
profile = {"wallets": len(stable), "trades": len(s), "pnl_net_usd": float(s.pnl.sum()),
           "buy_share_pct": float(100 * buy.mean()), "median_price": float(s.price.median()),
           "price_ge_0.90_pct": float(100 * (s.price >= .9).mean()),
           "last_2min_pct": float(100 * (s.rel >= 180).mean()), "median_second": float(s.rel.median()),
           "losing_trades": int((right == 0).sum()), "hit_rate_size_weighted_pct": float(100 * np.average(right, weights=s["size"])),
           "pnl_gross_c_per_share": float(100 * s.pnl_gross.sum() / s["size"].sum()),
           "pnl_net_c_per_share": float(100 * s.pnl.sum() / s["size"].sum()),
           "model_edge_c": float(100 * np.average(np.where(buy, s.f0 - s.price, s.price - s.f0), weights=s["size"])),
           "notional_usd": float(s.notional.sum())}
write("06_wallets", {"overall": overall, "persistence": persistence, "stable_wallets": profile})

fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4))
keys = ["all_active", "t_H1>2", "t_H1>3", "t_H1>4"]
labels = ["wszystkie\naktywne", "t > 2\nw 1. poł.", "t > 3\nw 1. poł.", "t > 4\nw 1. poł."]
vals = [persistence[k]["positive_H2_pct"] for k in keys]
axs[0].bar(labels, vals, color=[MID, INK, INK, INK], width=.6)
for i, (v, k) in enumerate(zip(vals, keys)):
    axs[0].text(i, v + 1, f"{v:.0f}%\n(n={persistence[k]['wallets']})", ha="center", fontsize=7.5)
axs[0].set(ylim=(0, 80), ylabel="% portfeli z zyskiem w 2. połowie", title="Trwałość umiejętności")
bins = pd.cut(s.price, [0, .5, .9, .95, .98, .99, 1.0001], labels=["<0,50", "0,50–0,90", "0,90–0,95", "0,95–0,98", "0,98–0,99", "≥0,99"])
vol = s.groupby(bins, observed=False)["size"].sum() / s["size"].sum() * 100
axs[1].bar(vol.index.astype(str), vol.values, color=INK, width=.6)
axs[1].set(ylabel="% wolumenu (akcje)", xlabel="cena transakcji", title=f"{len(stable)} trwałych portfeli: po ile handlują")
axs[1].tick_params(axis="x", labelsize=7.5)
save(fig, "fig06_wallets.png")
