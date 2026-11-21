"""Paths and constants shared across the project."""
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Sample: every third 5-minute market (one per 15 minutes) under the current settlement rule.
SAMPLE_START = "2026-10-23"
SAMPLE_END = "2026-11-20"          # exclusive
SAMPLE_STEP = 900                  # seconds between sampled markets

WINDOW = 300                       # market window length [s]
TWAP = 60                          # Chainlink TWAP length used for settlement [s]

# Taker fee per share: rate * p * (1 - p). Makers pay no fee.
TAKER_FEE = 0.07                   # crypto rate during the sample
TAKER_FEE_NEW = 0.05               # BTC 5m/15m rate from 2026-11-20 23:59 UTC


def utc_ts(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())


def ensure_dirs() -> None:
    for d in (RAW, PROCESSED, RESULTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
