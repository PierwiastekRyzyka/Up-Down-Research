"""The "favourite" rule shared by the latency and execution analyses.

Buy the token priced 0.90-0.98 when the model gives it at least 99.9%, at the first
opportunity in each market (from second 15 of the window onwards).
"""
import numpy as np
import pandas as pd

from .config import TAKER_FEE, TAKER_FEE_NEW
from .stats import tstat

PRICE_MIN, PRICE_MAX, MODEL_MIN, FIRST_SECOND = 0.90, 0.98, 0.999, 15


def first_opportunity(df: pd.DataFrame, pnl_col: str = "pnl_c", price_col: str = "price") -> dict:
    first = df.sort_values("ts").groupby("start_ts").head(1)
    n = len(first)
    if n < 3:
        return {"markets": n}
    return {"markets": n, "price": first[price_col].mean(), "hit_rate": first.won.mean(),
            "losses": int((first.won == 0).sum()), "pnl_c": first[pnl_col].mean(), "t": tstat(first[pnl_col])}


def favourite_buys(trades: pd.DataFrame) -> pd.DataFrame:
    """Taker buys of a likely favourite, with per-share PnL in cents under both fee rates."""
    b = trades[(trades.side == "BUY") & trades.price.between(PRICE_MIN, PRICE_MAX) & (trades.rel >= FIRST_SECOND)].copy()
    b["pnl_c"] = 100 * (b.won - b.price - TAKER_FEE * b.price * (1 - b.price))
    b["pnl_c_fee05"] = 100 * (b.won - b.price - TAKER_FEE_NEW * b.price * (1 - b.price))
    b["pnl_c_gross"] = 100 * (b.won - b.price)
    return b


def favourite_taker(trades: pd.DataFrame, lags=(0, 1, 2, 5, 10)) -> dict:
    buys = favourite_buys(trades)
    out = {}
    for lag in lags:
        d = buys[buys[f"f{lag}"] >= MODEL_MIN]
        out[f"{lag}s"] = {"all": first_opportunity(d), "gross": first_opportunity(d, "pnl_c_gross"),
                          "fee_0.05": first_opportunity(d, "pnl_c_fee05"),
                          "H1": first_opportunity(d[d.half == "H1"]), "H2": first_opportunity(d[d.half == "H2"])}
    out["no_model_filter"] = {"all": first_opportunity(buys)}
    return out


def ci95(mean: list, t: list) -> list:
    return [1.96 * abs(m / s) if s else np.nan for m, s in zip(mean, t)]
