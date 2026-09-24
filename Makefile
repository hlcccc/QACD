.PHONY: help install test demo lint serve clean

PY ?= python

help:
	@echo "make install   - install runtime dependencies"
	@echo "make test      - run the test suite (offline, no GPU)"
	@echo "make demo      - run the offline end-to-end demo"
	@echo "make serve     - run the HTTP service on :8080 (needs the service extra)"
	@echo "make clean     - remove caches and build artefacts"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

demo:
	$(PY) scripts/demo_offline.py

serve:
	$(PY) -m qacd.cli serve --scorer $${SCORER:-scorer.json} --host 0.0.0.0 --port 8080

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info
