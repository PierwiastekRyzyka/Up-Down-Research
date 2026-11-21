"""Execution: limit orders (maker) vs immediate orders FOK/FAK (taker).

Maker side is reconstructed from taker records. A taker SELL of token X at q means a
maker bought X at q; a taker BUY of X at q means a maker sold X, which is economically a
purchase of the opposite token at 1 - q. Maker PnL = -(taker PnL before fees); makers pay
no fee and rebates are not included. Fill probabilities of our own resting orders cannot
be measured without order-book history, so maker results are an approximation.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pm5m.config import TAKER_FEE, TAKER_FEE_NEW, WINDOW
from pm5m.plotting import DARK, INK, LIGHT, MID, save
from pm5m.results import write
from pm5m.sample import Btc, load_markets, load_trades
from pm5m.stats import ratio_stats, tstat
from pm5m.strategies import MODEL_MIN, PRICE_MAX, PRICE_MIN, FIRST_SECOND, ci95, favourite_buys, favourite_taker

m = load_markets()
btc = Btc.load()
t = load_trades(m, btc)
days = (m.index.max() + WINDOW - m.index.min()) / 86400

# --- taker (FOK/FAK) --------------------------------------------------------------------------------
fav = favourite_taker(t)
buys = favourite_buys(t)
cap = buys[buys.f2 >= MODEL_MIN]
capacity = {"days": days, "shares_per_day_sample": float(cap["size"].sum() / days),
            "usd_per_day_sample": float(cap.notional.sum() / days),
            "markets_with_signal_pct": float(100 * cap.start_ts.nunique() / len(m)),
            "note": "sample is every third market; multiply by ~3 for all 5-minute markets"}

# --- maker reconstruction --------------------------------------------------------------------------
bought_same = (t.side == "SELL").to_numpy()
mk = pd.DataFrame({"start_ts": t.start_ts, "ts": t.ts, "rel": t.rel, "half": t.half, "phase": t.phase, "size": t["size"],
                   "route": np.where(bought_same, "taker_sell", "taker_buy_opposite"),
                   "price": np.where(bought_same, t.price, 1 - t.price),
                   "won": np.where(bought_same, t.won, 1 - t.won),
                   "token_is_up": np.where(bought_same, t.outcome_index == 0, t.outcome_index == 1)})
for lag in (0, 2, 5):
    mk[f"f{lag}"] = np.where(bought_same, t[f"f{lag}"], 1 - t[f"f{lag}"])
mk = mk[mk.price.between(.01, .99)].reset_index(drop=True)     # drop dust fills
mk["pnl"] = (mk.won - mk.price) * mk["size"]
mk["edge"] = mk.f0 - mk.price


def maker_summary(d):
    if len(d) == 0:
        return {"trades": 0}
    r, tt, n = ratio_stats(d.pnl.to_numpy(), d["size"].to_numpy(), d.start_ts.to_numpy())
    return {"trades": len(d), "markets": n, "shares": float(d["size"].sum()),
            "avg_price": float(np.average(d.price, weights=d["size"])),
            "hit_rate": float(np.average(d.won, weights=d["size"])), "pnl_c": 100 * r, "t": tt,
            "model_edge_c": float(100 * np.average(d.edge, weights=d["size"])), "pnl_usd": float(d.pnl.sum())}


price_bins = [.01, .05, .15, .25, .35, .45, .55, .65, .75, .85, .95, .99 + 1e-9]
price_labels = ["0,01–0,05", "0,05–0,15", "0,15–0,25", "0,25–0,35", "0,35–0,45", "0,45–0,55",
                "0,55–0,65", "0,65–0,75", "0,75–0,85", "0,85–0,95", "0,95–0,99"]
mk["price_bin"] = pd.cut(mk.price, price_bins, labels=price_labels, right=False)
mk["model_side"] = np.where(mk.edge > 0, "model_with_maker", "model_against_maker")
maker = {"all": maker_summary(mk),
         "by_price": {str(k): maker_summary(d) for k, d in mk.groupby("price_bin", observed=True)},
         "by_phase": {str(k): maker_summary(d) for k, d in mk.groupby("phase", observed=True)},
         "by_model": {k: maker_summary(d) for k, d in mk.groupby("model_side")},
         "by_phase_and_model": {f"{ph} | {side}": maker_summary(d) for (ph, side), d in mk.groupby(["phase", "model_side"], observed=True)}}


def favourite_maker(lag):
    s = mk[mk.price.between(PRICE_MIN, PRICE_MAX) & (mk[f"f{lag}"] >= MODEL_MIN) & (mk.rel >= FIRST_SECOND)].copy()
    s["pnl_c"] = 100 * (s.won - s.price)
    out = maker_summary(s)
    first = s.sort_values("ts").groupby("start_ts").head(1)
    out["first_opportunity"] = {"markets": len(first), "price": first.price.mean(), "hit_rate": first.won.mean(),
                                "losses": int((first.won == 0).sum()), "pnl_c": first.pnl_c.mean(), "t": tstat(first.pnl_c)}
    return out, s


limit_fav = {}
for lag in (0, 2, 5):
    limit_fav[f"{lag}s"], sel = favourite_maker(lag)
    if lag == 0:
        limit_fav["0s"]["by_route"] = {k: maker_summary(d) for k, d in sel.groupby("route")}
        limit_fav["0s"]["H1"] = maker_summary(sel[sel.half == "H1"])
        limit_fav["0s"]["H2"] = maker_summary(sel[sel.half == "H2"])
blind = mk[mk.price.between(PRICE_MIN, PRICE_MAX) & (mk.rel >= FIRST_SECOND)]
limit_fav["no_model_filter"] = maker_summary(blind)
maker["limit_on_favourite"] = limit_fav

# Adverse selection: change in model value of the maker's token 5 s and 30 s after the fill.
for h in (5, 30):
    ok = (mk.rel + h < WINDOW).to_numpy()
    sub = mk[ok]
    fu = btc.fair_up(sub.start_ts.to_numpy(), sub.rel.to_numpy() + h, m.sigma.reindex(sub.start_ts).to_numpy())
    mk.loc[ok, f"markout{h}"] = np.where(sub.token_is_up, fu, 1 - fu) - sub.f0.to_numpy()
maker["markout_c"] = {f"{h}s": float(100 * np.average(mk[f"markout{h}"].fillna(0), weights=mk["size"] * mk[f"markout{h}"].notna()))
                      for h in (5, 30)}

# Quoting around the model: the maker buys token Y at c only when c < fair(Y) - delta.
deltas = (0.0, .01, .02, .03, .05, .10)
quote = {}
for lag in (0, 2, 5):
    for d in deltas:
        sel = mk[(mk[f"f{lag}"] - mk.price) > d]
        quote[f"lag{lag}_delta{d:.2f}"] = {**maker_summary(sel), "volume_share_pct": float(100 * sel["size"].sum() / mk["size"].sum())}
for d in (.02, .05):
    sel = mk[(mk.f2 - mk.price) > d]
    quote[f"lag2_delta{d:.2f}_by_half"] = {h: maker_summary(x) for h, x in sel.groupby("half")}
    quote[f"lag2_delta{d:.2f}_price_0.05-0.95"] = maker_summary(sel[sel.price.between(.05, .95)])
maker["quote_around_model"] = quote

# --- FOK with a 0.98 limit: per-second liquidity at or below the limit --------------------------
fok = {}
for lag in (0, 1, 2, 5):
    sel = buys[buys[f"f{lag}"] >= MODEL_MIN]
    sec = sel.groupby(["start_ts", "ts"]).apply(
        lambda d: pd.Series({"volume": d["size"].sum(), "vwap": np.average(d.price, weights=d["size"]), "won": d.won.iloc[0]}),
        include_groups=False).reset_index()
    res = {}
    for qty in (1, 100, 500, 1000):
        filled = sec[sec.volume >= qty].sort_values("ts").groupby("start_ts").head(1)
        if len(filled) < 3:
            res[str(qty)] = {"markets": len(filled)}
            continue
        px = filled.vwap
        pnl = 100 * (filled.won - px - TAKER_FEE * px * (1 - px))
        pnl_new = 100 * (filled.won - px - TAKER_FEE_NEW * px * (1 - px))
        res[str(qty)] = {"markets": len(filled), "markets_pct": 100 * len(filled) / len(m), "price": px.mean(),
                         "hit_rate": filled.won.mean(), "losses": int((filled.won == 0).sum()), "pnl_c": pnl.mean(),
                         "t": tstat(pnl), "pnl_c_fee_0.05": pnl_new.mean(),
                         "usd_per_day_sample": float(pnl.sum() / 100 * qty / days),
                         "usd_per_day_all_markets": float(3 * pnl.sum() / 100 * qty / days)}
    first = sec.sort_values("ts").groupby("start_ts").head(1)
    res["fak_avg_fill_pct"] = {str(q): float(100 * np.minimum(first.volume, q).mean() / q) for q in (100, 500, 1000)}
    res["first_second_liquidity_median_shares"] = float(first.volume.median())
    fok[f"{lag}s"] = res

write("07_execution", {"favourite_taker": fav, "capacity": capacity, "maker": maker, "fok": fok})

# --- figures ---------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
lags = [0, 1, 2, 5]
mean = [fav[f"{L}s"]["all"]["pnl_c"] for L in lags]
axs[0].errorbar(lags, mean, yerr=ci95(mean, [fav[f"{L}s"]["all"]["t"] for L in lags]), fmt="o-", capsize=3, color=INK,
                label="taker (FOK/FAK), po opłacie 0,07")
axs[0].plot(lags, [fav[f"{L}s"]["fee_0.05"]["pnl_c"] for L in lags], "s--", color=LIGHT, ms=4, label="taker, opłata 0,05 (od 21.11.2026)")
lm = [limit_fav[f"{L}s"]["first_opportunity"]["pnl_c"] for L in (0, 2, 5)]
lt = [limit_fav[f"{L}s"]["first_opportunity"]["t"] for L in (0, 2, 5)]
axs[0].errorbar([0, 2, 5], lm, yerr=ci95(lm, lt), fmt="^-", capsize=3, color=MID, label="maker (limit), bez opłaty")
axs[0].axhline(0, color="#333333", lw=.8)
axs[0].set(xlabel="opóźnienie sygnału [s]", ylabel="PnL [¢ na akcję], 95% CI", title="Limit na faworyta vs FOK: pierwsza okazja w rynku")
axs[0].legend(loc="upper right", fontsize=7)
keys = list(maker["by_price"])
vals = [maker["by_price"][k]["pnl_c"] for k in keys]
axs[1].bar(range(len(keys)), vals, color=[MID if v >= 0 else INK for v in vals], width=.65)
axs[1].axhline(0, color="#333333", lw=.8)
axs[1].set_xticks(range(len(keys)), keys, rotation=45, ha="right", fontsize=7)
axs[1].set(xlabel="cena, po której maker kupił token", ylabel="PnL makera [¢ na akcję]", title="Wynik makerów wg ceny (wszystkie transakcje)")
save(fig, "fig07_limit_vs_fok.png")

fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4))
phases = list(maker["by_phase"])
x = np.arange(len(phases))
axs[0].bar(x - .18, [maker["by_phase_and_model"][f"{p} | model_with_maker"]["pnl_c"] for p in phases], .34, color=MID, label="model po stronie makera")
axs[0].bar(x + .18, [maker["by_phase_and_model"][f"{p} | model_against_maker"]["pnl_c"] for p in phases], .34, color=INK, label="model przeciw makerowi")
axs[0].axhline(0, color="#333333", lw=.8)
axs[0].set_xticks(x, phases)
axs[0].set(xlabel="faza okna", ylabel="PnL makera [¢ na akcję]", title="Wynik makerów wg fazy i modelu")
axs[0].legend(loc="lower left", fontsize=7)
for lag, color in ((0, INK), (2, DARK), (5, LIGHT)):
    axs[1].plot(deltas, [quote[f"lag{lag}_delta{d:.2f}"]["pnl_c"] for d in deltas], "o-", color=color, label=f"zrealizowany PnL, opóźnienie {lag} s")
    axs[1].plot(deltas, [quote[f"lag{lag}_delta{d:.2f}"]["model_edge_c"] for d in deltas], ":", color=color, lw=1.2)
axs[1].plot([], [], ":", color="#333333", lw=1.2, label="przewaga wg modelu (oczekiwana)")
axs[1].axhline(0, color="#333333", lw=.8)
axs[1].set(xlabel="δ: wymagana przewaga modelu nad ceną", ylabel="¢ na akcję", title="Kwotowanie wokół modelu: wynik vs przewaga")
axs[1].legend(loc="upper left", fontsize=7)
save(fig, "fig08_maker_quotes.png")

fig, ax = plt.subplots(figsize=(6.6, 3.4))
qtys = ["1", "100", "500", "1000"]
x = np.arange(len(qtys))
for j, (lag, color) in enumerate(((0, INK), (2, MID), (5, LIGHT))):
    ax.bar(x + (j - 1) * .26, [fok[f"{lag}s"][q].get("markets_pct", 0) for q in qtys], .24, color=color, label=f"opóźnienie {lag} s")
ax.set_xticks(x, ["dowolna", "100 akcji", "500 akcji", "1000 akcji"])
ax.set(xlabel="wielkość zlecenia FOK", ylabel="% rynków z wypełnieniem", title="FOK z limitem 0,98: jak często zlecenie się wypełnia")
ax.legend()
save(fig, "fig09_fok_fill_rate.png")
