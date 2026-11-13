#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$HF_HOME" "$OLLAMA_MODELS" "$COMFYUI_MODELS" "$OUTPUT_DIR" "$MUSIC_DIR"
python - <<'PY'
import os
from pathlib import Path
import yaml
path=Path('/opt/ComfyUI/extra_model_paths.yaml')
path.write_text(yaml.safe_dump({'shorts':{'base_path':os.environ['COMFYUI_MODELS'],
    'diffusion_models':'diffusion_models','text_encoders':'text_encoders','vae':'vae'}}))
PY
exec "$@"
