FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

# App-Code
COPY app.py .
COPY app_stats.py .
COPY app_plots.py .
COPY scenario_plot.py .

# PNG Assets (Szenario-Referenzbilder)
COPY *.png ./

# OSC2 Dateien
COPY *.osc ./

# scenario_extraction Module
COPY scenario_extraction/parquet_source.py ./scenario_extraction/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]