#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${SHORTS_IN_CONTAINER:-}" == 1 || -d /opt/ComfyUI ]]; then
    exec python -m app.download "$@"
fi
exec docker compose run --rm --no-deps app python -m app.download "$@"
