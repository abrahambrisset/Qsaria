# Docker Deployment

## Quick Start

**Prerequisites:** Docker 20.10+, Compose 2.0+, 4 GB RAM, 10 GB disk

**First-time build:**

```bash
docker compose build chainlit-app
```

**Run:**

```bash
./docker-start.sh                    # Interactive (prompts for API key)
# OR
cp .env.example .env && docker-compose up -d  # Edit .env first
```

**Access:** [App](http://localhost:8000) | [MinIO Console](http://localhost:9001) (cs_copilot / chempwd123) | PostgreSQL: localhost:5432

## Services

| Service | Purpose | Ports |
|---------|---------|-------|
| chainlit-app | Main application | 8000 |
| minio | S3-compatible storage | 9000, 9001 |
| postgres | Chat history DB | 5432 |
| minio-setup | One-time bucket init | - |
| chainlit-db-init | Prisma migrations | - |

## Secret Management

`CHAINLIT_AUTH_SECRET` is auto-generated on first start and persisted to `./data/.chainlit_secret`.

**Priority:** env var > persisted file > generate new

```bash
# Override
CHAINLIT_AUTH_SECRET=your-secret  # in .env

# Regenerate
docker-compose down && rm ./data/.chainlit_secret && docker-compose up -d

# Backup (for production)
cp ./data/.chainlit_secret chainlit_secret_backup.txt
```

## Development

```bash
# Hot-reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Rebuild
docker compose build chainlit-app

# Complete reset
docker compose down -v --remove-orphans --rmi all

# Local storage only (no MinIO)
USE_S3=false  # in .env
```

## Dependencies

Uses [uv](https://docs.astral.sh/uv/) for reproducible builds from `uv.lock`.
The Docker image installs the QSAR runtime backends (`chemprop`, `lightgbm`,
`tabicl`) into `/app/.venv` during build and starts Chainlit from that virtual
environment directly. Avoid replacing the Docker command with `uv run ...` unless
you also install/sync the QSAR backend extras, because a runtime sync can remove
additive backend packages.

```bash
uv sync                              # Update deps on host
docker compose build chainlit-app    # Rebuild image
```

## Common Operations

```bash
docker-compose ps                    # Status
docker-compose logs -f chainlit-app  # Logs
docker-compose down                  # Stop
docker-compose down -v               # Stop + delete data
docker-compose restart chainlit-app  # Restart
docker-compose exec chainlit-app bash                      # Shell
docker-compose exec postgres psql -U postgres -d chainlit  # DB shell
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| App won't start | `docker-compose logs chainlit-app`, check `DEEPSEEK_API_KEY` |
| QSAR backends missing | Rebuild without cache, then check `/app/.venv/bin/python -c "import chemprop, lightgbm, tabicl; print('QSAR_BACKENDS_OK')"` inside `chainlit-app` |
| Port conflict | Use `./docker-start.sh` (auto-detects free ports) |
| DB error "relation User does not exist" | `docker-compose up -d chainlit-db-init && docker-compose restart chainlit-app` |
| MinIO issues | `docker-compose logs minio-setup` |
| Out of memory | Docker Desktop > Settings > Resources > Memory >= 4 GB |
| Disk space | `docker system prune` |

## Production

```bash
# Strong secrets
CHAINLIT_AUTH_SECRET=$(openssl rand -hex 32)
MINIO_SECRET_KEY=$(openssl rand -base64 32)

# Backup
docker-compose exec postgres pg_dump -U postgres chainlit > backup.sql

# Restore
cat backup.sql | docker-compose exec -T postgres psql -U postgres -d chainlit
```

**Recommendations:** HTTPS via nginx/Traefik, don't expose PostgreSQL publicly, `restart: always`, external PostgreSQL/S3 for scaling.
