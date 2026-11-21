"""Which settlement rule reproduces the official outcomes, using Binance as a proxy."""
import numpy as np
import pandas as pd

from pm5m.config import TWAP, WINDOW
from pm5m.results import write
from pm5m.sample import Btc, load_markets

m = load_markets()
btc = Btc.load()
idx = m.index.to_numpy() - btc.base
up = m.up.to_numpy()

last_twap = btc.mean(idx, WINDOW - TWAP, WINDOW)
full_mean = btc.mean(idx, 0, WINDOW)
candidates = {
    "twap60_last_vs_twap60_before": last_twap >= m.B.to_numpy(),
    "full_window_mean_vs_open": full_mean >= m.S0.to_numpy(),
    "full_window_mean_vs_twap60_before": full_mean >= m.B.to_numpy(),
}
agreement = {k: float((v == up).mean()) for k, v in candidates.items()}

margin_bp = np.abs(1e4 * np.log(last_twap / m.B.to_numpy()))
ok = candidates["twap60_last_vs_twap60_before"] == up
bins = pd.cut(margin_bp, [0, 1, 2, 5, 10, np.inf], right=False)
by_margin = pd.DataFrame({"bin": bins, "ok": ok}).groupby("bin", observed=True).ok.agg(["mean", "size"])

write("01_settlement_rule", {
    "markets": len(m), "first": str(pd.to_datetime(m.index.min(), unit="s")),
    "last": str(pd.to_datetime(m.index.max(), unit="s")),
    "share_up": float(m.up.mean()), "agreement": agreement,
    "agreement_by_margin_bp": {str(k): {"agreement": r["mean"], "n": int(r["size"])} for k, r in by_margin.iterrows()},
    "mismatches": int((~ok).sum()), "mismatches_below_2bp_pct": float(100 * (margin_bp[~ok] < 2).mean()),
})
