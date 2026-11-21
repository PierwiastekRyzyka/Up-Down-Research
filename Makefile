PY ?= python

.PHONY: install data panel analysis test all

install:
	$(PY) -m pip install -e ".[dev]"

data:                       ## ~20 min: markets, trades, BTC 1 s, DVOL
	$(PY) -m pm5m.data.polymarket markets
	$(PY) -m pm5m.data.polymarket trades
	$(PY) -m pm5m.data.prices

panel:
	$(PY) -m pm5m.panel

analysis:
	@for f in analysis/0*.py; do echo "== $$f"; $(PY) $$f || exit 1; done

test:
	$(PY) -m pytest -q

all: data panel analysis
