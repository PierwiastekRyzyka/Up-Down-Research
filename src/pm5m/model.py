"""Fair probability of "Up" under the TWAP-60s settlement rule.

Settlement: Up wins when the mean price over the last 60 s of the window is not lower than
the mean over the 60 s before the window starts (B, the price to beat).

Assume driftless arithmetic Brownian motion, S_u = S_t * (1 + sigma * W_{u-t}), with sigma
the per-second volatility. For t <= T - 60 the outcome depends on the mean of S over
[T-60, T]; since Var(int_a^{a+L} W) = L^2 a + L^3 / 3, the effective horizon is
T - 40 - t and

    P(Up) = Phi( ln(S_t / B) / (sigma * sqrt(T - 40 - t)) ).

Inside the last minute k = t - (T - 60) seconds of the average are already realised
(mean R) and the remaining integral over tau = T - t seconds has variance tau^3 / 3:

    P(Up) = Phi( (k R + tau S_t - 60 B) / (S_t sigma sqrt(tau^3 / 3)) ).

Both cases are checked against Monte Carlo in tests/test_model.py.
"""
import numpy as np
from scipy.stats import norm

T = 300.0
L = 60.0
SECONDS_PER_YEAR = 365 * 86400


def p_up(B, S_t, R, t, sigma):
    """P(Up) at second t. B: price to beat, S_t: current price, R: realised mean since T-60."""
    t = np.asarray(t, dtype=float)
    early = norm.cdf(np.log(S_t / B) / (sigma * np.sqrt(np.maximum(T - 40 - t, 1e-9))))
    tau = np.maximum(T - t, 1e-9)
    k = np.clip(t - (T - L), 0, L)
    late = norm.cdf((k * R + tau * S_t - L * B) / (S_t * sigma * np.sqrt(tau ** 3 / 3)))
    return np.where(t <= T - L, early, late)


def sigma_from_dvol(dvol):
    """Deribit DVOL (annualised implied vol, %) -> per-second volatility."""
    return np.asarray(dvol) / 100.0 / np.sqrt(SECONDS_PER_YEAR)
