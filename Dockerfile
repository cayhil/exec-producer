# Executive Producer — engine + media toolchain.
# The image ships ONLY the committed starter show (examples/); private show
# directories are excluded by .dockerignore and mounted at runtime instead:
#   docker run --env-file .env -v ./my_show:/show exec-producer \
#       --data-dir /show --scene <id> --lines 6
FROM python:3.14-slim

# ffmpeg: --audio stitching + --video encoding · fonts: Pillow card text
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependency layer first, for build caching. All extras included; litellm is
# capped <1.92, which still ships pure-Python wheels — safe on cp314.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --all-extras --no-install-project

COPY . .
RUN uv sync --frozen --all-extras

# Least privilege: the engine only needs to write scripts/ under /app
RUN useradd --create-home producer && chown -R producer /app
USER producer

# --no-sync: the environment is fully baked above; container start must not
# try to re-resolve or reinstall anything
ENTRYPOINT ["uv", "run", "--no-sync", "exec-producer"]
CMD ["--list"]
