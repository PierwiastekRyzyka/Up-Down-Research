"""Minimal JSON-over-HTTP client with retries (stdlib only)."""
import json
import time
import urllib.error
import urllib.request

USER_AGENT = "pm5m-research/1.0"


def get_json(url: str, retries: int = 6, timeout: int = 30):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as err:
            if err.code != 429 and err.code < 500:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1.5 * 2 ** attempt)
    raise RuntimeError(f"request failed after {retries} attempts: {url}")
