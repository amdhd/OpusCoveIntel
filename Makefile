.DEFAULT_GOAL := help
.PHONY: help install lint fmt type test test-live check run worker \
        up down down-volumes logs psql shell migrate migration migrate-down seed clean

UV := uv

help:  ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# -- development ---------------------------------------------------------

install:  ## Create the venv and install all dependencies
	$(UV) sync --all-groups
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example")

lint:  ## Run ruff (check + format check)
	$(UV) run ruff check app tests
	$(UV) run ruff format --check app tests

fmt:  ## Auto-fix lint issues and format
	$(UV) run ruff check --fix app tests
	$(UV) run ruff format app tests

type:  ## Run mypy
	$(UV) run mypy

test:  ## Run tests. Makes ZERO paid API calls (CLAUDE.md 7).
	$(UV) run pytest

test-live:  ## Run tests INCLUDING billable provider calls. Costs real money.
	RUN_LIVE_LLM_TESTS=1 $(UV) run pytest

check: lint type test  ## lint + type + test

run:  ## Run the API with autoreload
	$(UV) run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker:  ## Run the background worker
	$(UV) run python -m app.worker.main

# -- docker --------------------------------------------------------------

up:  ## Start db, api and worker
	docker compose up -d --build
	@echo "api      -> http://localhost:8000"
	@echo "docs     -> http://localhost:8000/docs"
	@echo "health   -> http://localhost:8000/health"

down:  ## Stop services, keep data
	docker compose down

down-volumes:  ## Stop services and DELETE the database volume
	docker compose down -v

logs:  ## Tail service logs
	docker compose logs -f

psql:  ## Open a psql shell against the compose database
	docker compose exec db psql -U opuscovintel -d opuscovintel

shell:  ## Open a shell in the api container
	docker compose exec api /bin/bash

# -- database ------------------------------------------------------------

migrate:  ## Apply migrations
	$(UV) run alembic upgrade head

migration:  ## Autogenerate a migration: make migration m="add widgets"
	$(UV) run alembic revision --autogenerate -m "$(m)"

migrate-down:  ## Roll back one migration
	$(UV) run alembic downgrade -1

seed:  ## Load synthetic demo data (idempotent)
	$(UV) run opuscovintel seed

# -- housekeeping --------------------------------------------------------

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
