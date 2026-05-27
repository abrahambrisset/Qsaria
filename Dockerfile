FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# System dependencies
RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    apt-get -o Acquire::Retries=5 update; \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        openssl \
        libpq-dev \
        libboost-dev \
        libcairo2-dev \
        libeigen3-dev \
        nodejs \
        npm; \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependency metadata (for Docker layer caching)
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies via uv
# Chemprop and TabICL are installed as additive runtime dependencies for the
# prediction agent so we can validate both backends inside the app container
# without making the base project dependency graph mandatory for every
# environment.
RUN uv sync --frozen --no-dev \
    && uv pip install --python /app/.venv/bin/python "chemprop>=2.2.0" "lightgbm>=4.5.0" "tabicl>=2.0.3"

# Application source
COPY . .

# Prisma / Node dependencies
COPY package.json ./
COPY prisma ./prisma

RUN npm install \
    && npx prisma generate

# Runtime prep
RUN mkdir -p /app/data

# Copy and configure entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
# Use the already-synced virtualenv at runtime. Running through `uv run` here can
# resync the project without the additive QSAR backend packages installed above.
CMD ["/app/.venv/bin/chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]
