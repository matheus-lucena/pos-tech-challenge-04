from typing import Tuple


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """Parse an S3 URI and return (bucket_name, object_key).

    Raises ValueError for any path that does not follow the s3://bucket/key format.
    """
    if not s3_path.startswith("s3://"):
        raise ValueError(
            f"S3 path must start with 's3://'. Received: {s3_path}"
        )
    parts = s3_path[5:].split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid S3 path format. Expected s3://bucket/key. Received: {s3_path}"
        )
    return parts[0], parts[1]
