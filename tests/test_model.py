"""Closed-form P(Up) against Monte Carlo simulation of 1-second paths."""
import numpy as np
import pytest

from pm5m.model import p_up

SIGMA = 1.2e-4
N_PATHS = 200_000


@pytest.mark.parametrize("t", [0, 60, 150, 250, 280])
def test_closed_form_matches_monte_carlo(t):
    rng = np.random.default_rng(t)
    history = 100 * (1 + SIGMA * np.cumsum(rng.standard_normal(60 + t + 1)))
    B, S_t = history[:60].mean(), history[-1]
    realised = history[60:]
    R = realised[240:].mean() if t > 240 else np.nan

    future = S_t * (1 + SIGMA * np.cumsum(rng.standard_normal((N_PATHS, 300 - t)), axis=1))
    paths = np.concatenate([np.repeat(realised[None, :], N_PATHS, 0), future], axis=1)[:, :300]
    simulated = (paths[:, 240:].mean(axis=1) >= B).mean()

    assert float(p_up(B, S_t, R, t, SIGMA)) == pytest.approx(simulated, abs=0.005)
