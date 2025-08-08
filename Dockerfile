# Dockerfile for Artisan Engine (Corrected for Build Dependencies)

# --- Builder Stage ---
# This stage uses a full Python image that includes build tools (like C++ compilers)
# necessary to compile packages like llama-cpp-python.
FROM python:3.12-bookworm AS builder

# Install uv by copying the binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml ./
COPY uv.lock ./

# Install dependencies, excluding development ones.
# This will now succeed because the build tools are available.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Copy the rest of the application source code
COPY ./artisan_engine ./artisan_engine
COPY ./main.py ./

# Install the application itself into the virtual environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --compile-bytecode --no-dev

# --- Production Stage ---
# This stage uses the small, secure slim image for the final product.
# It does not contain any of the build tools from the builder stage.
FROM python:3.12-slim AS production

# Install the required libgomp1 shared library that llama.cpp needs to run.
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd --system appuser && useradd --system --gid appuser appuser

WORKDIR /app

# Create models directory for mounting
RUN mkdir -p /app/models && chown appuser:appuser /app/models

# Copy the pre-compiled virtual environment and application code from the builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/artisan_engine ./artisan_engine
COPY --from=builder --chown=appuser:appuser /app/main.py ./

# Activate the virtual environment by adding it to the PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Switch to the non-root user
USER appuser

EXPOSE 8000

# Health check to ensure the API is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

# The command to run the application using a production-ready server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
