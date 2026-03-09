FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libmariadb-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY uv.lock ./

RUN pip install --upgrade pip && \
    pip install uv && \
    uv pip install --system -r pyproject.toml

COPY . .

RUN mkdir -p logs data files runtimes temps

EXPOSE 8000

CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
