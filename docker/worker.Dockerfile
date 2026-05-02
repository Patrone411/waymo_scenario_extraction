FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgeos-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_worker.txt .
RUN pip install --no-cache-dir -r requirements_worker.txt

# lokale Module inkl. Submodul
COPY feature_extraction/ ./feature_extraction/
COPY external/ ./external/
COPY worker.py .

CMD ["python", "worker.py"]