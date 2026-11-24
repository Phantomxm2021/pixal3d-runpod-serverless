"""Runpod GitHub deployment entrypoint."""

import os

import runpod

from handler import handler as handle_job
from scripts.model_cache import use_global_volume


def handler(job):
    return handle_job(job)


if __name__ == '__main__':
    if os.environ.get('RUNPOD_WEBHOOK_GET_JOB') and not use_global_volume():
        # Preload the GPU models before accepting queued jobs.
        import inference_worker

    runpod.serverless.start({'handler': handler })
