FROM python:3.10-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgeos-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY scenario_extraction/requirements_matcher.txt .
RUN pip install --no-cache-dir -r requirements_matcher.txt

# scenario_extraction als Python-Root
COPY scenario_extraction/ ./

# Externe Abhängigkeiten
COPY external/ ./external/

ENV PYTHONPATH=/app

CMD ["python", "run_matching.py"]