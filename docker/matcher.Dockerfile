FROM python:3.10-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgeos-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY scenario_extraction/requirements_matcher.txt .
RUN pip install --no-cache-dir -r requirements_matcher.txt

# Lokale Module
COPY scenario_extraction/osc2_parser/ ./osc2_parser/
COPY scenario_extraction/scenario_matching/ ./scenario_matching/
COPY external/ ./external/

# Matcher
COPY scenario_extraction/parquet_source.py .
COPY scenario_extraction/azure_results_writer.py .
COPY scenario_extraction/run_matching.py .

CMD ["python", "run_matching.py"]