.DEFAULT_GOAL := help
.PHONY: help install lint fmt type test test-live check run worker \
        frontend frontend-install frontend-test frontend-types frontend-serve \
        up down down-volumes logs psql shell migrate migration migrate-down seed user-add \
        sample-pdf ingest-sample corpus ingest-corpus index extract-sample \
        extract-llm-dry-run extract-llm \
        ocr ocr-dry-run \
        query-sample ask-sample golden eval eval-demo cost-report demo clean

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

audit:  ## Known vulnerabilities in the Python and client dependency trees. $0
	@# What CI's `dependency audit` job runs, so a red tick is reproducible
	@# locally. `uv pip audit` does not exist; the tool is PyPA's pip-audit,
	@# pinned so the scanner cannot silently change under us.
	$(UV) export --format requirements-txt --no-emit-project --all-groups \
		> /tmp/opuscovintel-audit.txt
	uvx pip-audit@2.10.1 --requirement /tmp/opuscovintel-audit.txt \
		--progress-spinner off --strict
	cd frontend && npm audit --audit-level=high

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

# -- frontend (Angular client app, served at /app) -----------------------
#
# Optional: the API and the server-rendered UI at /ui work without any of
# this. `app/main.py` mounts the build if it is there and says so if it is not.

frontend-install:  ## Install the client app's dependencies
	cd frontend && npm ci

frontend: frontend-install  ## Build the client app into frontend/dist
	cd frontend && npm run build

frontend-test:  ## Run the client app's unit tests (headless Chrome)
	cd frontend && npm run test:ci

frontend-serve:  ## Dev server on :4200, proxying the API on :8000
	cd frontend && npm start

frontend-types:  ## Regenerate the client's types from the API's OpenAPI schema
	$(UV) run python scripts/export_openapi.py
	cd frontend && npm run gen:api

run:  ## Run the API with autoreload
	$(UV) run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker:  ## Run the background worker
	$(UV) run python -m app.worker.main

# -- docker --------------------------------------------------------------

up:  ## Start db, api and worker
	docker compose up -d --build
	@echo "ui       -> http://localhost:8000"
	@echo "api docs -> http://localhost:8000/docs"
	@echo "health   -> http://localhost:8000/health"
	@echo
	@echo "The UI needs an account: make user-add u=<name> role=reviewer"

user-add:  ## Create a UI account: make user-add u=aminah role=reviewer
	@test -n "$(u)" || (echo "usage: make user-add u=<username> [role=analyst|reviewer]"; exit 2)
	$(UV) run opuscovintel user-add $(u) --role $(or $(role),analyst)

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

# -- ingestion -----------------------------------------------------------

SAMPLE_PDF := var/sample-prospectus.pdf

sample-pdf:  ## Generate the synthetic prospectus fixture
	$(UV) run python -m tests.fixtures.synthetic_pdf $(SAMPLE_PDF)

ingest-sample: sample-pdf  ## Ingest the synthetic prospectus (parse, score, chunk). $0
	$(UV) run opuscovintel ingest $(SAMPLE_PDF)

CORPUS_DIR := var/corpus

corpus:  ## Generate all three synthetic fixtures (prospectus, trust deed, rating report)
	$(UV) run python -m tests.fixtures.synthetic_pdf $(CORPUS_DIR)

# `make eval` scores all three labelled documents. Ingesting only the
# prospectus reports the other two as "never ingested" and computes extraction
# metrics over a third of the corpus.
ingest-corpus: corpus  ## Ingest the whole synthetic corpus. Idempotent by hash, $0
	@for pdf in $(CORPUS_DIR)/*.pdf; do $(UV) run opuscovintel ingest "$$pdf"; done

# -- search, rules and query ---------------------------------------------

index:  ## Embed + full-text index every document. $0 (offline embedder)
	$(UV) run opuscovintel index

extract-sample:  ## Run the deterministic extractor over every document. $0
	$(UV) run opuscovintel extract-rules

# The LLM extractor. These are the only targets in this file that can spend
# money, so they are named apart from `extract-sample` rather than folded into
# it -- `make demo` depends on that target and is documented as a $0 pipeline.
extract-llm-dry-run:  ## Price LLM extraction over every document without calling it. $0
	$(UV) run opuscovintel extract --all --dry-run

extract-llm:  ## Run LLM extraction over every document. **SPENDS MONEY** — prompts first
	$(UV) run opuscovintel extract --all

ocr-dry-run:  ## Report which pages would go to the vision model, and the cost. $0
	$(UV) run opuscovintel ocr --all --dry-run

ocr:  ## OCR flagged pages and chunk what was read. **SPENDS MONEY** — prompts first
	$(UV) run opuscovintel ocr --all

query-sample:  ## Answer a sample question over the deterministic path. $0
	$(UV) run opuscovintel query "Which holdings would breach their rating trigger at the current rating?"

ask-sample:  ## Answer the same question through the LangGraph agent. Logged + audited. $0
	$(UV) run opuscovintel ask "Which holdings would breach their rating trigger at the current rating?"

golden:  ## Run the golden question set. Phase 4 target: 6/10 with zero LLM calls
	$(UV) run opuscovintel golden

# -- evaluation ----------------------------------------------------------

eval:  ## Score extraction + answers -> evals/results/. $0, no model calls
	$(UV) run opuscovintel eval

eval-demo: migrate seed ingest-corpus index extract-sample eval  ## Corpus + eval from nothing

cost-report:  ## LLM spend by stage and by document, from the llm_calls ledger. $0
	$(UV) run opuscovintel cost-report

# `--dry-run` is hardcoded and there is no way to pass anything else through, so
# this target cannot spend. It exists because `.claude/settings.json` denies
# `extract --all` outright -- the shape that once cost $0.39 unattended -- and
# that deny cannot tell `--all --yes` from `--all --dry-run`, since permissions
# match on prefix. Pricing the corpus is the thing an agent *should* do before
# asking to spend, so it gets a door that is safe by construction rather than an
# incentive to work around the lock.
cost-preview:  ## What extracting every ingested document would cost, worst case. $0
	$(UV) run opuscovintel extract --all --dry-run

demo: migrate seed ingest-sample index extract-sample golden  ## Full $0 pipeline, end to end

# -- housekeeping --------------------------------------------------------

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
