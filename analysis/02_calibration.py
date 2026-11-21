"""Calibration of market prices vs the model, and the opening gap."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pm5m.plotting import DARK, INK, LIGHT, MID, save
from pm5m.results import write
from pm5m.sample import PHASES, load_markets, load_panel
from pm5m.stats import brier

m = load_markets()
p = load_panel(m)

res = {"observations": len(p), "markets": int(p.start_ts.nunique()),
       "brier": {"market": brier(p.p_market, p.y), "model": brier(p.p_model, p.y),
                 "model_dvol": brier(p.p_model_dvol, p.y), "coin_flip": 0.25},
       "brier_by_half": {h: {"market": brier(g.p_market, g.y), "model": brier(g.p_model, g.y), "n": len(g)}
                         for h, g in p.groupby("half")},
       "brier_by_phase": {str(ph): {"market": brier(g.p_market, g.y), "model": brier(g.p_model, g.y), "n": len(g)}
                          for ph, g in p.groupby("phase", observed=True)}}

bins = np.linspace(0, 1, 11)
def curve(col):
    return (p.assign(b=pd.cut(p[col], bins, include_lowest=True)).groupby("b", observed=True)
             .agg(x=(col, "mean"), y=("y", "mean"), n=("y", "size")))
cal_mkt, cal_mod = curve("p_market"), curve("p_model")
res["curve_market"] = cal_mkt.reset_index(drop=True).to_dict("records")
res["curve_model"] = cal_mod.reset_index(drop=True).to_dict("records")

fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]
ax.plot([0, 1], [0, 1], color=MID, ls=":", lw=1, label="idealna kalibracja")
ax.plot(cal_mkt.x, cal_mkt.y, "o-", color=INK, label="cena rynkowa")
ax.plot(cal_mod.x, cal_mod.y, "s--", color=MID, label="model TWAP-60s", ms=4)
ax.set(xlabel="prognozowane P(Up)", ylabel="zrealizowany odsetek wyników Up", title="Krzywa kalibracji", xlim=(0, 1), ylim=(0, 1))
ax.legend(loc="upper left")
ax = axs[1]
x = np.arange(len(PHASES))
ax.bar(x - .18, [res["brier_by_phase"][k]["market"] for k in PHASES], .34, color=DARK, label="cena rynkowa")
ax.bar(x + .18, [res["brier_by_phase"][k]["model"] for k in PHASES], .34, color=LIGHT, label="model TWAP-60s")
ax.set_xticks(x, PHASES)
ax.set(ylabel="Brier score (niżej = lepiej)", xlabel="faza okna", title="Brier score wg fazy okna")
ax.legend(loc="upper right")
save(fig, "fig01_calibration.png")

# Opening gap: at the start the price to beat (TWAP before the window) differs from spot,
# so the fair probability is not 50%. How much of that does the market price in?
o = p[p.t == 10].merge(m[["gap_bp"]], left_on="start_ts", right_index=True)
slope = lambda d: float(np.polyfit(d.p_model - .5, d.p_market - .5, 1)[0])
res["opening_gap"] = {"n": len(o), "slope_market_on_model": slope(o),
                      "slope_by_half": {h: slope(g) for h, g in o.groupby("half")},
                      "median_abs_model_minus_half_pp": float(100 * np.median(np.abs(o.p_model - .5))),
                      "median_abs_gap_bp": float(np.median(np.abs(o.gap_bp))),
                      "brier_market": brier(o.p_market, o.y), "brier_model": brier(o.p_model, o.y)}
dec = (o.assign(q=pd.qcut(o.gap_bp, 10)).groupby("q", observed=True)
        .agg(gap=("gap_bp", "mean"), market=("p_market", "mean"), model=("p_model", "mean"), realised=("y", "mean"), n=("y", "size")))
res["opening_gap"]["deciles"] = dec.reset_index(drop=True).to_dict("records")

fig, ax = plt.subplots(figsize=(6.6, 3.5))
ax.axhline(.5, color=MID, lw=.8, ls=":")
ax.plot(dec.gap, dec.realised, "o", color=INK, label="zrealizowany odsetek Up", ms=6)
ax.plot(dec.gap, dec.model, "s--", color=MID, label="model TWAP-60s", ms=4)
ax.plot(dec.gap, dec.market, "^-", color=DARK, label="cena rynkowa (0–10 s)", ms=5)
ax.set(xlabel="luka otwarcia ln(S₀ / B) [punkty bazowe], decyle", ylabel="P(Up)",
       title=f"Luka otwarcia: rynek wycenia ok. {100 * res['opening_gap']['slope_market_on_model']:.0f}% luki")
ax.legend(loc="upper left")
save(fig, "fig02_opening_gap.png")

write("02_calibration", res)
