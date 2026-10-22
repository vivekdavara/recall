PY := .venv/bin/python
export RECALL_TEST_DATABASE_URL ?= postgresql://localhost:5544/recall_test

.PHONY: venv lint test check db-start db-stop coverage

venv:
	python3.11 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e '.[dev]'

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m mypy

# The app's database is a database no test may use: a test that forgets RECALL_DATABASE_URL fails here as it does in CI.
test:
	RECALL_DATABASE_URL=postgresql://localhost:5544/no_tests_here $(PY) -m pytest

# The tests that need PostgreSQL use RECALL_TEST_DATABASE_URL; `make db-start` runs the local cluster it points at.
check: lint test

db-start:
	scripts/pg.sh start

db-stop:
	scripts/pg.sh stop

coverage:
	$(PY) -m pytest --cov --cov-report=term-missing
