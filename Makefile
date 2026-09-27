.PHONY: help install install-runtime test demo serve version check clean

# Override on the command line, e.g. `make test PY=python3.11`.
PY ?= python
# The service has no authentication. Keep the default on the loopback interface
# and pass HOST=0.0.0.0 explicitly when you really do want it on the network --
# and only behind a gateway. See docs/04 section 7.
HOST ?= 127.0.0.1
PORT ?= 8080
SCORER ?= scorer.json

help:
	@echo "make install          - install the package + CLI + test deps (the documented quick start)"
	@echo "make install-runtime  - install runtime dependencies only (numpy, pandas)"
	@echo "make test             - run the test suite (offline, no GPU, no model weights)"
	@echo "make demo             - run the offline end-to-end demo"
	@echo "make serve            - run the HTTP service (HOST/PORT/SCORER overridable)"
	@echo "make check            - verify the scripts that need data report its absence"
	@echo "make clean            - remove caches and build artefacts"

# What the README and docs/03 tell you to run. Installs the `qacd` console
# script; `pip install -r requirements.txt` does not, which is why both targets
# exist rather than one.
install:
	$(PY) -m pip install -e ".[test]"

install-runtime:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q -rs

demo:
	$(PY) scripts/demo_offline.py

serve:
	$(PY) -m qacd.cli serve --scorer $(SCORER) --host $(HOST) --port $(PORT)

version:
	$(PY) -m qacd.cli version

# The repository ships without its experimental data. These scripts must say so
# and exit 2 rather than pass silently on empty input.
check:
	@for s in verify_feature_port verify_decomposition_v1 verify_pipeline_frozen; do \
		$(PY) scripts/$$s.py > /dev/null 2>&1; \
		code=$$?; \
		if [ $$code -eq 2 ]; then echo "ok   $$s exited 2 (data absent, as expected)"; \
		else echo "FAIL $$s exited $$code, expected 2"; exit 1; fi; \
	done

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache build dist *.egg-info .coverage htmlcov
