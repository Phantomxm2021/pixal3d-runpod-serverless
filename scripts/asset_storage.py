"""Private GLB delivery through S3-compatible object storage."""

from __future__ import annotations

GLB_MIME_TYPE = "model/gltf-binary"


def storage_client():
    import os

    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("PIXAL3D_OUTPUT_ENDPOINT_URL"),
        config=Config(signature_version="s3v4"),
    )


def download_url(client, bucket: str, key: str) -> str:
    return client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=3600
    )
