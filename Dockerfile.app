FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv==0.8.15

# Install the locked runtime dependencies first so this layer is cached across code changes
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY data/abalone.csv ./data/abalone.csv
COPY bin ./bin

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:/app" \
    PREFECT_HOME="/app/.prefect"

# The model pickle is not versioned, so train it while building the image
RUN python -m modelling.main data/abalone.csv

# 8001: FastAPI, 4201: Prefect UI (see bin/run_services.sh)
EXPOSE 8001 4201

CMD ["bash", "bin/run_services.sh"]
