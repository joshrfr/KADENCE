# Convenience wrappers. Every target is a plain command from the README.
# Run from the repository root.
PY ?= python3

.PHONY: test adversary scale churn rust native clean help

help:
	@echo "targets: test adversary scale churn rust native clean"

test:            ## unit tests (needs pytest for one module: pip install -r requirements-dev.txt)
	$(PY) -m pytest -q tests

adversary:       ## regenerates results/desync_adversary.json (about 1 min)
	$(PY) experiments/simulation/desync_adversary.py

scale:           ## regenerates results/desync_scale.json (about 2 min)
	$(PY) experiments/simulation/desync_scale.py

churn:           ## regenerates results/churn_evaluation.json (seconds)
	$(PY) experiments/simulation/churn_evaluation.py --output results/churn_evaluation.json

rust:            ## builds the Rust prototype
	cd src/kadence-rs && cargo build --release

native:          ## builds and tests the C kernels
	$(MAKE) -C src/native test

clean:
	find . -name __pycache__ -prune -exec rm -rf {} +
	rm -rf src/kadence-rs/target src/native/build analysis/generated .pytest_cache
