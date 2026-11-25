"""Object names and private delivery shared by the generation and export workers."""

from __future__ import annotations

GLB_MIME_TYPE = "model/gltf-binary"
USDZ_MIME_TYPE = "model/vnd.usdz+zip"


def asset_source_key(asset_id: str) -> str:
    return f"pixal3d/sources/{asset_id}.usdz"


def white_mesh_key(asset_id: str) -> str:
    return f"pixal3d/white-mesh/{asset_id}.glb"


def s3_error_code(error: Exception) -> str | None:
    return getattr(error, "response", {}).get("Error", {}).get("Code")


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
