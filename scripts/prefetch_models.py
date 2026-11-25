"""Populate the attached Global Volume with the direct GLB pipeline's models."""

import json

from scripts.gguf_models import gguf_snapshot
from scripts.global_models import prefill_global_models


if __name__ == "__main__":
    print(json.dumps(prefill_global_models()))
    snapshot, _ = gguf_snapshot()
    print(f"Prepared GGUF checkpoint snapshot at {snapshot}")
