PYTHON ?= python3
PIP := $(PYTHON) -m pip

.PHONY: wheel install test clean

wheel:
	$(PIP) wheel . --no-deps -w dist

install:
	$(PIP) install dist/*.whl

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf dist build __pycache__
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +