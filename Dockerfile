# Multi-stage Dockerfile for SEIZ Disinformation Trajectory Simulator
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-system

# Copy dependency specs and C++ core first for optimal build caching
COPY pyproject.toml uv.lock README.md CMakeLists.txt ./
COPY src/cpp_core/ ./src/cpp_core/

# Install Python dependencies with uv and compile the C++ pybind11/Eigen3 module
RUN uv sync --frozen --no-dev \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/app/.venv/bin/python \
    && cmake --build build --config Release

# Copy remaining application Python modules and Alembic migrations
COPY src/__init__.py ./src/__init__.py
COPY src/api/ ./src/api/
COPY src/ingestion/ ./src/ingestion/
COPY src/models/ ./src/models/
COPY src/simulation/ ./src/simulation/
COPY src/ui/ ./src/ui/
COPY src/workers/ ./src/workers/
COPY alembic.ini ./alembic.ini
COPY alembic/ ./alembic/


FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/alembic.ini /app/alembic.ini
COPY --from=builder /app/alembic /app/alembic

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
