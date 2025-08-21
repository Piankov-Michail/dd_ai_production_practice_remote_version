FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Копируем скомпилированный uv из официального образа
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Этот слой будет пересобираться только если изменить зависимости
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

COPY . .

CMD ["python", "bot.py"]