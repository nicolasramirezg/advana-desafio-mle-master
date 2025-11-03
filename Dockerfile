FROM python:3.12-slim

WORKDIR /app
COPY . /app

# Install uv
RUN pip install --no-cache-dir uv

# Sync dependencies using uv
RUN uv sync --frozen --no-dev

ENV PORT=8080
EXPOSE 8080

CMD ["uv", "run", "uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
