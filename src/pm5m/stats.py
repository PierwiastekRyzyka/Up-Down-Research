"""Small statistical helpers. Standard errors are clustered by market where it matters."""
import numpy as np
import pandas as pd
from scipy.special import expit


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def tstat(x) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def logit_regression(X, y, iters: int = 50):
    """Logistic regression by Newton's method. Returns coefficients and the Hessian."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = expit(X @ b)
        H = X.T @ (X * (p * (1 - p))[:, None])
        b += np.linalg.solve(H, X.T @ (y - p))
    return b, H


def clustered_se_logit(X, y, b, H, groups):
    scores = pd.DataFrame(X * (y - expit(X @ b))[:, None]).groupby(groups).sum().to_numpy()
    Hi = np.linalg.inv(H)
    return np.sqrt(np.diag(Hi @ (scores.T @ scores) @ Hi))


def clustered_se_ols(X, resid, groups):
    scores = pd.DataFrame(X * resid[:, None]).groupby(groups).sum().to_numpy()
    XtXi = np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(XtXi @ (scores.T @ scores) @ XtXi))


def ratio_stats(pnl, size, groups):
    """Size-weighted PnL per share with a market-clustered t-stat (ratio estimator)."""
    d = pd.DataFrame({"p": pnl, "s": size, "g": groups}).groupby("g").sum()
    r = d.p.sum() / d.s.sum()
    se = np.sqrt(((d.p - r * d.s) ** 2).sum()) / d.s.sum()
    return float(r), float(r / se) if se > 0 else float("nan"), int(len(d))
