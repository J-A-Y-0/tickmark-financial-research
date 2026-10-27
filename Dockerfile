FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tickmark ./tickmark
COPY web ./web
COPY evals ./evals
COPY data ./data

RUN useradd --create-home tickmark && mkdir -p var && chown tickmark var
USER tickmark

EXPOSE 8000
CMD ["sh", "-c", "uvicorn web.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
