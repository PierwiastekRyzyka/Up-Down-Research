"""Build the analysis tables from raw data.

markets.parquet  one row per market: outcome, opening price S0, price to beat B,
                 opening gap, realised and implied volatility.
panel.parquet    market x checkpoint t: market price of Up and model P(Up).

Market price at t is the median of trade prices (converted to Up) over [t-10 s, t).
The median filters out dust fills at 0.001/0.999. Two extra prices are stored for
realistic backtests: a fresh one over [t-3 s, t) and an execution price over [t+1, t+6).
"""
import json

import numpy as np
import pandas as pd

from .config import PROCESSED, RAW, TWAP, WINDOW, ensure_dirs
from .data.prices import load_btc, load_dvol
from .model import p_up, sigma_from_dvol

CHECKPOINTS = (10, 20, 30, 45, 60, 90, 120, 150, 180, 210, 240, 255, 270, 285)
HISTORY = 3600                     # seconds of BTC history used for realised volatility


def load_raw_markets() -> pd.DataFrame:
    rows = [json.loads(line) for line in (RAW / "markets.jsonl").open()]
    m = pd.DataFrame(rows).drop_duplicates("start_ts")
    return m[m.winner.notna()].set_index("start_ts").sort_index()


def build() -> None:
    ensure_dirs()
    markets = load_raw_markets()
    btc = load_btc()
    base, px = btc.index[0], btc.to_numpy()
    dvol = load_dvol()

    trades = pd.read_parquet(RAW / "trades.parquet", columns=["start_ts", "ts", "outcome_index", "price"])
    trades["rel"] = trades.ts - trades.start_ts
    trades["up_px"] = np.where(trades.outcome_index == 0, trades.price, 1 - trades.price)
    trades = trades[(trades.rel >= -60) & (trades.rel < WINDOW)]
    by_market = {k: (g.rel.to_numpy(), g.up_px.to_numpy()) for k, g in trades.groupby("start_ts")}

    mrows, prows = [], []
    for start, mk in markets.iterrows():
        i = start - base
        if i < HISTORY or i + WINDOW > len(px):
            continue
        hist, window = px[i - HISTORY:i], px[i:i + WINDOW]
        S0, B = hist[-1], hist[-TWAP:].mean()
        sigma = np.diff(np.log(hist)).std()
        dv = dvol.asof(start)
        sigma_dv = float(sigma_from_dvol(dv))
        up = mk.winner == "Up"
        mrows.append({"start_ts": start, "up": up, "volume": mk.volume, "S0": S0, "B": B,
                      "gap_bp": 1e4 * np.log(S0 / B), "sigma": sigma, "sigma_dvol": sigma_dv, "dvol": dv})
        if start not in by_market:
            continue
        rel, prices = by_market[start]
        for t in CHECKPOINTS:
            recent = (rel >= t - 10) & (rel < t)
            if recent.sum() < 3:
                continue
            fresh = (rel >= t - 3) & (rel < t)
            later = (rel >= t + 1) & (rel < t + 6)
            S_t = window[t - 1]
            R = window[WINDOW - TWAP:t].mean() if t > WINDOW - TWAP else np.nan
            prows.append({"start_ts": start, "t": t, "up": up,
                          "p_market": float(np.median(prices[recent])), "n_trades": int(recent.sum()),
                          "p_market_fresh": float(np.median(prices[fresh])) if fresh.any() else np.nan,
                          "p_exec": float(np.median(prices[later])) if later.any() else np.nan,
                          "p_model": float(p_up(B, S_t, R, t, sigma)),
                          "p_model_dvol": float(p_up(B, S_t, R, t, sigma_dv))})

    pd.DataFrame(mrows).to_parquet(PROCESSED / "markets.parquet")
    pd.DataFrame(prows).to_parquet(PROCESSED / "panel.parquet")
    print(f"panel: {len(mrows)} markets, {len(prows)} observations")


if __name__ == "__main__":
    build()
