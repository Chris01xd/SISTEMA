FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional AI assistant (LangChain). Own layer so changing the core requirements does not
# reinstall it; remove these two lines to build without it (the endpoint then answers 501).
COPY requirements-assistant.txt .
RUN pip install --no-cache-dir -r requirements-assistant.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
