# Dockerfile
FROM bitnami/spark:3.3.2

# Copy jars into Spark jars dir
COPY jars/ /opt/bitnami/spark/jars/

# Python deps for the job
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy job(s)
COPY jobs/ /opt/bitnami/spark/jobs/

WORKDIR /opt/bitnami/spark
ENTRYPOINT ["/opt/bitnami/spark/bin/spark-submit", "--master", "local[*]", "/opt/bitnami/spark/jobs/convert_tfrecords.py"]