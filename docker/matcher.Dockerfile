# docker/matcher.Dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_matcher.txt .
RUN pip install --no-cache-dir -r requirements_matcher.txt

# dein Code
COPY scenario_matching/ ./scenario_matching/
COPY osc2_parser/ ./osc2_parser/
COPY parquet_source.py .
COPY run_matching.py .
COPY osc2_parser/osc/ ./osc2_parser/osc/
COPY certs/ ./certs/

CMD ["python", "run_matching.py"]