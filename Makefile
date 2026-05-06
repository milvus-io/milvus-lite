PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
PYTEST := $(BIN)/pytest
PYTEST_ARGS ?= --tb=short -q

.PHONY: venv install dev build test test-fast test-core test-compat test-stability test-soak test-all benchmark coverage serve lint clean

# Create virtual environment
venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)"
	@echo "Run: source $(VENV)/bin/activate"

# Install package with all dependencies
install: venv
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e .

# Install with dev dependencies
dev: venv
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

# Build distribution artifacts for PyPI: wheel and source distribution
build: venv
	$(BIN)/python -m pip install build
	$(BIN)/python -m build

# Run the regular functional suite, excluding slow tests and benchmark
test:
	$(PYTEST) $(PYTEST_ARGS) -m "not slow and not soak" --ignore=tests/benchmark

# Run a fast pre-commit suite: no slow tests, no benchmark
test-fast:
	$(PYTEST) $(PYTEST_ARGS) -m "not slow and not soak" --ignore=tests/benchmark

# Run core engine/storage/search/index tests without gRPC compatibility tests
test-core:
	$(PYTEST) $(PYTEST_ARGS) -m "not soak" tests/schema tests/storage tests/search tests/index tests/analyzer tests/engine tests/function tests/rerank tests/embedding tests/test_db.py tests/test_smoke_e2e.py

# Run gRPC adapter and pymilvus compatibility tests
test-compat:
	$(PYTEST) $(PYTEST_ARGS) tests/adapter tests/compatibility

# Run stability-oriented long tests, excluding the performance benchmark
test-stability:
	$(PYTEST) $(PYTEST_ARGS) -m "slow and not soak" --ignore=tests/benchmark

# Run very large scale soak tests explicitly
test-soak:
	MILVUS_LITE_RUN_SOAK=1 $(PYTEST) $(PYTEST_ARGS) -m soak --ignore=tests/benchmark

# Run all tests including benchmark
test-all:
	$(PYTEST) $(PYTEST_ARGS) -m "not soak" --ignore=tests/benchmark
	$(PYTEST) tests/benchmark/ -v -s

# Run benchmark only
benchmark:
	$(PYTEST) tests/benchmark/ -v -s

# Run tests with coverage
coverage:
	$(PYTEST) --cov=milvus_lite --cov-report=term-missing -q -m "not slow and not soak" --ignore=tests/benchmark

# Start gRPC server
serve:
	$(BIN)/milvus-lite server --data-dir ./data --port 19530

# Clean up
clean:
	rm -rf $(VENV) .pytest_cache .coverage htmlcov build dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
