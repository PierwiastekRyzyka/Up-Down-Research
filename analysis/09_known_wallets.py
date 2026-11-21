"""Two wallets that were profitable on BTC Up/Down in early October, re-checked on the sample.

Addresses come from the Polymarket profile search. Only their taker fills are visible in
the trade data, and the sample covers every third 5-minute market.
"""
import numpy as np
import pandas as pd

from pm5m.results import write
from pm5m.sample import Btc, load_markets, load_trades
from pm5m.stats import tstat

WALLETS = {
    "NIULAI4": "0x44832d0d2ec11187c1e77d786feb15f6a50254c6",
    "Atomforge": "0x84389cfc4a652ea14d8d8be969769b1f69de3680",
}
PRICE_BINS = [0, 0.3, 0.5, 0.7, 0.9, 1.0]

m = load_markets()
t = load_trades(m, Btc.load(), lags=(0,))
t["wallet"] = t.wallet.str.lower()

res = {}
for name, address in WALLETS.items():
    x = t[t.wallet == address]
    per_market = x.groupby("start_ts").pnl.sum()
    halves = {h: g.groupby("start_ts").pnl.sum() for h, g in x.groupby("half")}
    trimmed = per_market.sort_values().iloc[: int(len(per_market) * 0.95)]
    by_price = x.groupby(pd.cut(x.price, PRICE_BINS), observed=True).agg(fills=("pnl", "size"), pnl=("pnl", "sum"))
    by_phase = x.groupby("phase", observed=True).agg(fills=("pnl", "size"), pnl=("pnl", "sum"))
    res[name] = {
        "fills": len(x),
        "markets": per_market.size,
        "notional_usd": x.notional.sum(),
        "pnl_gross_usd": x.pnl_gross.sum(),
        "fees_usd": x.fee.sum(),
        "pnl_usd": x.pnl.sum(),
        "t": tstat(per_market),
        "markets_profitable_pct": 100 * (per_market > 0).mean(),
        "pnl_usd_by_half": {h: s.sum() for h, s in halves.items()},
        "t_by_half": {h: tstat(s) for h, s in halves.items()},
        "pnl_usd_without_top5pct_markets": trimmed.sum(),
        "pnl_usd_by_price": {str(k): v for k, v in by_price.pnl.items()},
        "pnl_usd_by_phase": {str(k): v for k, v in by_phase.pnl.items()},
    }
    print(f"{name}: {len(x)} fills, {per_market.size} markets, net {x.pnl.sum():,.0f} USD, t = {tstat(per_market):.2f}")

write("09_known_wallets", res)
