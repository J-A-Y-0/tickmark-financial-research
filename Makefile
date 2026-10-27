.PHONY: install demo run test lint format eval index filings record docker

PY ?= .venv/bin/python

install:  ## Create .venv and install app + dev dependencies
	python3.12 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements-dev.txt

demo:  ## Run offline: no API keys, no accounts
	LLM_PROVIDER=offline ROUTER=keyword EMBEDDING_PROVIDER=hashing SEARCH_PROVIDER=none \
	$(PY) -m uvicorn web.app:app --reload --port 8000

run:  ## Run with the settings in .env
	$(PY) -m uvicorn web.app:app --reload --port 8000

test:  ## Unit, integration and web tests (offline, ~1 s)
	$(PY) -m pytest

lint:  ## Lint and format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check . --fix

eval:  ## Compare every configured router on the evaluation set (needs API keys)
	$(PY) -m evals.run

index:  ## Rebuild the search index
	$(PY) -m tickmark index --force

filings:  ## Re-download the 10-K excerpts from SEC EDGAR (needs SEC_USER_AGENT)
	$(PY) data/build_filings.py

record:  ## Re-record docs/images/demo.gif and screenshots (app running on :8000; needs Chrome)
	$(PY) -m pip install -q -r requirements-demo.txt
	$(PY) scripts/record_demo.py

docker:
	docker build -t tickmark .
	docker run --rm -p 8000:8000 --env-file .env tickmark
