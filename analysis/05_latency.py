"""How fast the market absorbs BTC moves.

1. Information decay: logistic regression of the outcome on the fresh market price and
   the model, where the model sees BTC lagged by L seconds. When the model's coefficient
   stops being significant, the market has caught up.
2. Lead-lag: does the gap between the price-implied BTC level and Binance spot predict
   future spot returns? (It would if Polymarket traders saw faster feeds.)
3. The favourite rule as a taker, by signal delay.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from pm5m.model import p_up
from pm5m.plotting import DARK, INK, LIGHT, MID, save
from pm5m.results import write
from pm5m.sample import Btc, load_markets, load_panel, load_trades
from pm5m.stats import clustered_se_logit, clustered_se_ols, logit, logit_regression
from pm5m.strategies import ci95, favourite_taker

m = load_markets()
btc = Btc.load()
p = load_panel(m)
q = p[p.p_market_fresh.notna() & p.t.between(20, 240)]
idx = q.start_ts.to_numpy() - btc.base
t = q.t.to_numpy()
B = m.B.reindex(q.start_ts).to_numpy()
sigma = m.sigma.reindex(q.start_ts).to_numpy()
y = q.y.to_numpy()
groups = q.start_ts.to_numpy()

decay = {}
for lag in range(9):
    model = p_up(B, btc.px[idx + t - 1 - lag], np.nan, t - lag, sigma)
    X = np.column_stack([np.ones(len(q)), logit(q.p_market_fresh), logit(model)])
    b, H = logit_regression(X, y)
    se = clustered_se_logit(X, y, b, H, groups)
    decay[lag] = {"coef_model": float(b[2]), "t": float(b[2] / se[2])}

implied_gap = norm.ppf(np.clip(q.p_market_fresh, .02, .98)) * sigma * np.sqrt(np.maximum(260 - t, 1)) \
    - np.log(btc.px[idx + t - 1] / B)
lead = {}
for h in (1, 2, 3, 5, 10, 20):
    future = np.log(btc.px[idx + t - 1 + h] / btc.px[idx + t - 1])
    X = np.column_stack([np.ones(len(q)), implied_gap])
    beta = np.linalg.lstsq(X, future, rcond=None)[0]
    se = clustered_se_ols(X, future - X @ beta, groups)
    lead[h] = {"beta": float(beta[1]), "t": float(beta[1] / se[1])}

fav = favourite_taker(load_trades(m, btc))
write("05_latency", {"observations": len(q), "information_decay": decay, "lead_lag": lead,
                     "max_abs_t_lead": max(abs(v["t"]) for v in lead.values()), "favourite_taker": fav})

fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4))
lags = list(range(9))
axs[0].plot(lags, [decay[L]["t"] for L in lags], "o-", color=INK)
axs[0].axhline(2, color=MID, ls=":", lw=1)
axs[0].axhline(0, color="#333333", lw=.8)
axs[0].text(8, 2.15, "t = 2", ha="right", fontsize=7.5, color=MID)
axs[0].set(xlabel="opóźnienie danych BTC w modelu [s]", ylabel="t-stat informacji modelu ponad cenę", title="Informacja modelu znika w ok. 3 s")
lags = [0, 1, 2, 5, 10]
mean = [fav[f"{L}s"]["all"]["pnl_c"] for L in lags]
axs[1].errorbar(lags, mean, yerr=ci95(mean, [fav[f"{L}s"]["all"]["t"] for L in lags]), fmt="o-", capsize=3, color=INK, label="cały okres")
for half, color in (("H1", LIGHT), ("H2", DARK)):
    axs[1].plot(lags, [fav[f"{L}s"][half]["pnl_c"] for L in lags], "s--", ms=3.5, lw=1.2, color=color,
                label="1. połowa" if half == "H1" else "2. połowa")
axs[1].axhline(0, color="#333333", lw=.8)
axs[1].set(xlabel="opóźnienie sygnału [s]", ylabel="PnL po opłacie [¢ na akcję]", title="Reguła faworyta (taker) a opóźnienie")
axs[1].legend(loc="upper center", ncol=3)
save(fig, "fig05_latency.png")
