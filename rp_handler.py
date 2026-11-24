"""Runpod GitHub deployment entrypoint."""

import os

import runpod

from handler import handler as handle_job


def handler(job):
    return handle_job(job)


if __name__ == '__main__':
    if os.environ.get('RUNPOD_WEBHOOK_GET_JOB'):
        # Preload the GPU models before accepting queued jobs.
        import inference_worker

    runpod.serverless.start({'handler': handler })
