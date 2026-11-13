import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from app.files import read_json, write_json
from app.gpu import process_identity
from app.schema import WorkerInput


def worker_main(generate):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    registry_file = None
    try:
        request = WorkerInput.model_validate(read_json(args.input))
        registry = request.run_dir.parent / '.gpu-procs'
        registry_file = registry / f'worker-{os.getpid()}.json'
        write_json(registry_file, {**process_identity(os.getpid()), 'component': 'worker', 'heartbeat': time.time()})
        result = generate(request)
        write_json(args.output, result.model_dump(mode='json'))
    except Exception as error:
        write_json(args.output, {'error': str(error), 'cuda_oom': any(term in str(error).lower() for term in ('out of memory', 'cuda oom'))})
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        if registry_file:
            registry_file.unlink(missing_ok=True)
