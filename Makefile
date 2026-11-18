.PHONY: setup ingest index eval test run docker index-cloud help

PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
PYTEST := .venv/bin/pytest
RUFF   := .venv/bin/ruff
MYPY   := .venv/bin/mypy

help:
	@echo "Citewise — available targets:"
	@echo "  make setup        create .venv and install all dependencies"
	@echo "  make ingest       discover, fetch, clean and chunk GOV.UK pages"
	@echo "  make index        embed and upsert chunks into local Qdrant"
	@echo "  make index-cloud  upsert to Qdrant Cloud (needs QDRANT_URL/QDRANT_API_KEY)"
	@echo "  make eval         run experiments on the dev split"
	@echo "  make test         ruff + mypy + pytest (unit + integration)"
	@echo "  make run          start app + local Qdrant via Docker Compose"
	@echo "  make docker       build and run Docker image standalone"

setup:
	/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "Setup complete. Activate with: source .venv/bin/activate"

ingest:
	$(PYTHON) -m citewise.ingest.discover
	$(PYTHON) -m citewise.ingest.fetch
	$(PYTHON) -m citewise.ingest.clean
	$(PYTHON) -m citewise.ingest.chunk

index:
	docker compose up -d qdrant
	$(PYTHON) -m citewise.index.build

index-cloud:
	$(PYTHON) -m citewise.index.build --target cloud

eval:
	$(PYTHON) eval/experiments.py --split dev

test:
	$(RUFF) check src/ tests/ eval/
	$(RUFF) format --check src/ tests/ eval/
	$(MYPY) src/
	$(PYTEST) tests/unit/ -v
	$(PYTEST) tests/integration/ -v -m "not slow"

run:
	docker compose up

docker:
	docker build -t citewise:local .
	docker run --env-file .env -p 8000:8000 citewise:local
