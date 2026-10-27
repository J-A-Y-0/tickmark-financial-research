# Deployment

Tickmark is a single stateless-enough process: the index is rebuilt from `data/` at startup
when it is missing or out of date (about a second offline). Uploaded documents, the cache
and the audit log live in SQLite at `DATABASE_PATH`.

## Render

`render.yaml` defines a Python web service:

1. Create a Blueprint from the repository.
2. Set `LLM_PROVIDER` and the matching API key. Optionally set `TYPESAFE_API_KEY`, `SEARCH_PROVIDER` and its key.
3. Deploy. Health check: `/health`.

Render's free disk is ephemeral, so uploads, cache and audit history reset on redeploy.
For persistence, attach a disk and set `DATABASE_PATH=/var/data/tickmark.sqlite3`.

With `EMBEDDING_PROVIDER=openai`, the first start embeds about 550 chunks (under $0.01
at list price) and the index is reused after that.

## Docker

```sh
docker build -t tickmark .
docker run --rm -p 8000:8000 --env-file .env tickmark
```

Mount a volume at `/app/var` to keep the database.

## Configuration

Every setting is an environment variable; see [`.env.example`](../.env.example). The
defaults run fully offline. `/health` and `/api/config` report what is active: models,
router, embeddings, web search, verifier, document and chunk counts.

## Operating notes

- `DAILY_BUDGET_USD` and `DEEP_DAILY_CAP` are per user per UTC day, computed from the audit log.
- Changing `data/access.json` takes effect on restart. Bump `policy_version` when you edit it: the version is part of the cache key, so every previously cached answer becomes unreachable.
- `python -m tickmark index --force` rebuilds the corpus index. `make filings` re-downloads the 10-K excerpts from EDGAR (set `SEC_USER_AGENT`).
