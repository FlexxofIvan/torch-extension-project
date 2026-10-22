PYTHON ?= python3
PIP := $(PYTHON) -m pip
TORCH_VERSION ?= 2.14.1
.DEFAULT_GOAL := wheel

.PHONY: setup wheel install test clean

setup:
	$(PIP) install "torch==$(TORCH_VERSION)+cpu" --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install "setuptools>=77" "numpy>=2.4.6" "pytest>=9.1.1" "ruff>=0.17.1"

wheel:
	rm -f dist/*.whl
	$(PIP) wheel . --no-deps --no-build-isolation -w dist

install:
	$(PIP) install --force-reinstall --no-deps dist/*.whl
	$(PIP) check

test:
	@set -eu; \
	test_python=$$($(PYTHON) -c 'import sys; print(sys.executable)'); \
	test_dir=$$(mktemp -d /tmp/fse-tests.XXXXXX); \
	trap 'rm -rf "$$test_dir"' EXIT; \
	cp -r tests "$$test_dir/tests"; \
	cd "$$test_dir"; \
	"$$test_python" -I -m pytest -q tests/

clean:
	rm -rf dist build __pycache__ .pytest_cache
	find src tests tools -type d -name __pycache__ -prune -exec rm -rf {} +
