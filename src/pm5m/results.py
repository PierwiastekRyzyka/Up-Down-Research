"""Write analysis results as rounded, JSON-safe dictionaries."""
import json

import numpy as np

from .config import RESULTS


def _clean(o, digits=4):
    if isinstance(o, dict):
        return {str(k): _clean(v, digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v, digits) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return round(float(o), digits) if np.isfinite(o) else None
    return o


def write(name: str, data: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(_clean(data), indent=2, ensure_ascii=False))
    print(f"results/{name}.json")
