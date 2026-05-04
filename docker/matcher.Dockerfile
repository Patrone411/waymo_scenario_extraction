# docker/matcher.Dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_matcher.txt .
RUN pip install --no-cache-dir -r requirements_matcher.txt

COPY feature_extraction/ ./feature_extraction/
COPY external/ ./external/
COPY scenario_extraction/ ./scenario_extraction/

WORKDIR /app/scenario_extraction
CMD ["python", "run_matching.py"]