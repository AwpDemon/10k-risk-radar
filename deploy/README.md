# Deploy (not live)

The dashboard is static over committed data (`data/sample/explorer.parquet`, `data/results/`),
so the container needs no API keys, no database, and no write access.

```bash
docker compose -f deploy/docker-compose.yml up -d --build   # http://127.0.0.1:8501
```

Before exposing it publicly: put it behind a reverse proxy with rate limiting, keep the
container read-only, and do not mount the EDGAR cache or LLM endpoint into it. A cloud
equivalent would be any container host (e.g. Cloud Run / App Runner) serving the same image.
