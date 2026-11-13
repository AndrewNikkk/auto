#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
dry=0; check=0; skip=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry=1;;
        --check-only) check=1;;
        --skip-models) skip=1;;
        *) echo "Unknown argument: $arg" >&2; exit 2;;
    esac
done
if ((dry)); then
    echo 'DRY RUN: check NVIDIA GPU, RAM, disk, Docker/Compose, NVIDIA Container Toolkit and .env.'
    echo 'Copy .env.example only if .env is missing; compose config; build; GPU container probe.'
    echo 'Verify/download models unless --skip-models; start UI; wait for HTTP readiness.'
    exit 0
fi
for cmd in nvidia-smi docker curl; do
    command -v "$cmd" >/dev/null || { echo "BLOCKED: missing $cmd" >&2; exit 1; }
done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
((gpu_mib >= 23000)) || { echo 'BLOCKED: first visible GPU must have 24 GB VRAM' >&2; exit 1; }
ram_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
((ram_kb >= 32*1024*1024)) || { echo 'BLOCKED: at least 32 GiB RAM required (64 recommended)' >&2; exit 1; }
docker compose version
docker info >/dev/null
root_dir=$(docker info --format '{{.DockerRootDir}}')
[[ -d "$root_dir" ]] || { echo "BLOCKED: cannot check free space in DockerRootDir=$root_dir" >&2; exit 1; }
free_kb=$(df -Pk "$root_dir" | awk 'NR==2{print $4}')
((free_kb >= 100*1024*1024)) || { echo 'BLOCKED: need 100 GiB free in Docker storage' >&2; exit 1; }
[[ -f .env ]] || { ((check)) && { echo 'BLOCKED: .env missing' >&2; exit 1; }; cp .env.example .env; }
docker compose config --quiet
check_container_gpu() {
    docker compose run --rm --no-deps app nvidia-smi
    docker compose run --rm --no-deps app python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; assert torch.cuda.is_bf16_supported(), 'BF16 required'; x=torch.zeros(1,device='cuda',dtype=torch.bfloat16); print('VERIFIED: CUDA/BF16 allocation',torch.cuda.get_device_name(0))"
}
if ((check)); then
    check_container_gpu
    ./download_models.sh --check-only
    echo 'VERIFIED: installation checks; inference requires GPU smoke test'
    exit 0
fi
docker compose build app
check_container_gpu
if ((skip)); then
    ./download_models.sh --check-only
else
    ./download_models.sh
fi
docker compose up -d app
# Read only GRADIO_PORT, never source .env as executable shell code.
port=$(awk -F= '/^GRADIO_PORT=/{gsub(/\r/,"",$2); print $2}' .env | tail -n 1)
port=${port:-7860}
[[ "$port" =~ ^[0-9]+$ ]] || { echo 'BLOCKED: invalid GRADIO_PORT' >&2; exit 1; }
ready=0
for ((i=0;i<120;i++)); do
    if curl -fsS --max-time 5 "http://127.0.0.1:$port/" >/dev/null; then ready=1; break; fi
    sleep 2
done
((ready)) || { docker compose logs --tail=100 app; echo 'BLOCKED: UI readiness failed' >&2; exit 1; }
echo "VERIFIED: UI is ready at http://127.0.0.1:$port"
echo "SSH tunnel: ssh -L $port:localhost:$port USER@SERVER_IP"
echo 'REQUIRES GPU: run ./gpu-smoke-test.sh before trusting full generation quality'
