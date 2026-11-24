"""Runpod GitHub deployment entrypoint."""

if __name__ == '__main__':
    print('[startup] Python entrypoint reached', flush=True)

import faulthandler
import os
import time

TRACE = os.environ.get('PIXAL3D_STAGE_TRACE') == '1'
if __name__ == '__main__':
    if TRACE:
        faulthandler.dump_traceback_later(60, repeat=True)
    print('[startup] Importing Runpod SDK', flush=True)

import runpod

if __name__ == '__main__':
    print('[startup] Runpod SDK imported', flush=True)

from handler import handler as handle_job
from scripts.model_cache import use_global_volume


def handler(job):
    job_id = job.get('id', 'unknown')
    started = time.monotonic()
    print(f'[request {job_id}] Entered handler', flush=True)
    if TRACE:
        faulthandler.dump_traceback_later(60, repeat=True)
    try:
        return handle_job(job)
    finally:
        if TRACE:
            faulthandler.cancel_dump_traceback_later()
        print(f'[request {job_id}] Handler returned after {time.monotonic() - started:.1f}s', flush=True)


if __name__ == '__main__':
    global_volume = use_global_volume()
    if os.environ.get('RUNPOD_WEBHOOK_GET_JOB') and not global_volume:
        # Preload the GPU models before accepting queued jobs.
        print('[startup] Preloading models before SDK registration', flush=True)
        import inference_worker

    print(f'[startup] Registering handler with Runpod SDK (global_volume={global_volume})', flush=True)
    runpod.serverless.start({'handler': handler })
