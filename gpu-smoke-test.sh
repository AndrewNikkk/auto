#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker compose exec -T app nvidia-smi
./download_models.sh --check-only
run_id="gpu_smoke_$(date +%Y%m%d_%H%M%S)"
docker compose exec -T app python -m app.generate --prompt 'Что будет, если все корабли исчезнут? Рассмотри как мысленный эксперимент.' \
    --languages ru,en --duration 30 --type what_if --seed 42 --run-id "$run_id"
docker compose exec -T app python -m app.smoke_verify "$run_id"
