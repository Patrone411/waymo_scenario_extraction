from kubernetes import client, config

config.load_kube_config()          # or load_incluster_config() inside the cluster
batch_v1 = client.BatchV1Api()

GCS_BUCKET   = "my-training-bucket"
S3_BUCKET    = "my-output-bucket"
IMAGE        = "your-registry/scenario-worker:latest"
TOTAL_SHARDS = 1000
NAMESPACE    = "default"

for shard_idx in range(TOTAL_SHARDS):
    job = client.V1Job(
        metadata=client.V1ObjectMeta(name=f"scenario-shard-{shard_idx:05d}"),
        spec=client.V1JobSpec(
            backoff_limit=2,
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[client.V1Container(
                        name="worker",
                        image=IMAGE,
                        command=["python", "worker.py"],
                        env=[
                            client.V1EnvVar(name="GCS_BUCKET",    value=GCS_BUCKET),
                            client.V1EnvVar(name="S3_BUCKET",     value=S3_BUCKET),
                            client.V1EnvVar(name="SHARD_INDEX",   value=str(shard_idx)),
                            client.V1EnvVar(name="TOTAL_SHARDS",  value=str(TOTAL_SHARDS)),
                        ],
                        resources=client.V1ResourceRequirements(
                            requests={"cpu": "1", "memory": "2Gi"},
                            limits={"cpu": "2",   "memory": "4Gi"},
                        ),
                    )],
                )
            ),
        ),
    )
    batch_v1.create_namespaced_job(namespace=NAMESPACE, body=job)
    print(f"Created job for shard {shard_idx:05d}")