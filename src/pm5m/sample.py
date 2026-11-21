"""Load the analysis sample. Shared by all scripts in analysis/.

Conventions
- First/second half (H1/H2): markets split at the median start time; used as the
  out-of-sample check. A result counts only if it holds in both halves.
- Trades: each record is the taker's fill. side is the taker's side; outcome_index 0 is
  Up, 1 is Down. Taker PnL before fees is +/-(won - price) * size.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PROCESSED, RAW, TAKER_FEE, TWAP, WINDOW
from .data.prices import load_btc
from .model import p_up

PHASES = ["0–60 s", "60–180 s", "180–240 s", "240–300 s"]


@dataclass
class Btc:
    """BTC 1 s prices as an array with prefix sums, for fast window means."""
    base: int
    px: np.ndarray
    cum: np.ndarray

    @classmethod
    def load(cls) -> "Btc":
        s = load_btc()
        return cls(int(s.index[0]), s.to_numpy(), np.concatenate([[0.0], np.cumsum(s.to_numpy())]))

    def mean(self, idx, a, b):
        """Mean over seconds [idx + a, idx + b) for an array of window-start offsets."""
        return (self.cum[idx + b] - self.cum[idx + a]) / (b - a)

    def fair_up(self, start_ts, rel, sigma):
        """Model P(Up) using information available at second rel of each window."""
        idx = np.asarray(start_ts) - self.base
        rel = np.asarray(rel)
        B = self.mean(idx, -TWAP, 0)
        S_t = self.px[idx + rel - 1]
        k = np.clip(rel - (WINDOW - TWAP), 0, TWAP)
        R = np.where(k > 0, (self.cum[idx + WINDOW - TWAP + k] - self.cum[idx + WINDOW - TWAP]) / np.maximum(k, 1), np.nan)
        return p_up(B, S_t, R, rel, sigma)


def load_markets() -> pd.DataFrame:
    m = pd.read_parquet(PROCESSED / "markets.parquet").set_index("start_ts").sort_index()
    m["half"] = np.where(m.index < np.median(m.index), "H1", "H2")
    return m


def load_panel(markets: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_parquet(PROCESSED / "panel.parquet")
    p = p[p.start_ts.isin(markets.index)].copy()
    p["y"] = p.up.astype(float)
    p["half"] = markets.half.reindex(p.start_ts).to_numpy()
    p["phase"] = pd.cut(p.t, [0, 60, 180, 240, 300], labels=PHASES)
    return p


def load_trades(markets: pd.DataFrame, btc: Btc, lags=(0, 1, 2, 5, 10)) -> pd.DataFrame:
    """In-window trades with outcome, PnL, fee and model value of the traded token.

    Column f{L} is the model probability of the traded token computed with BTC data
    lagged by L seconds, i.e. what a participant L seconds slower would have known.
    """
    t = pd.read_parquet(RAW / "trades.parquet")
    t = t[t.start_ts.isin(markets.index)]
    t = t.assign(rel=t.ts - t.start_ts)
    t = t[(t.rel >= 0) & (t.rel < WINDOW)].reset_index(drop=True)
    up = markets.up.reindex(t.start_ts).to_numpy()
    is_up = (t.outcome_index == 0).to_numpy()
    t["won"] = np.where(is_up, up, ~up).astype(float)
    t["half"] = markets.half.reindex(t.start_ts).to_numpy()
    t["phase"] = pd.cut(t.rel, [-1, 59, 179, 239, 299], labels=PHASES)
    sigma = markets.sigma.reindex(t.start_ts).to_numpy()
    for lag in lags:
        fu = btc.fair_up(t.start_ts.to_numpy(), t.rel.to_numpy() - lag, sigma)
        t[f"f{lag}"] = np.where(is_up, fu, 1 - fu)
    sign = np.where(t.side == "BUY", 1.0, -1.0)
    t["pnl_gross"] = sign * (t.won - t.price) * t["size"]
    t["fee"] = TAKER_FEE * t.price * (1 - t.price) * t["size"]
    t["pnl"] = t.pnl_gross - t.fee
    t["notional"] = t.price * t["size"]
    return t
