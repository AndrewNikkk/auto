# AGENTS.md — AI Shorts Generator

## Project Overview

AI Shorts Generator is a lightweight, fully local AI pipeline for generating high-quality vertical videos (YouTube Shorts, TikTok, Instagram Reels) from a text prompt.

**Primary goal:** Deploy on a single NVIDIA RTX 3090/4090 24 GB server, enter a prompt, and receive a finished MP4 without paid AI APIs or manual video editing.

This is a **single-user MVP**, not a SaaS platform.

Prioritize:
1. Working end-to-end video generation.
2. Video and audio quality.
3. Reliable GPU memory management.
4. Simple Docker deployment.
5. Minimal code and infrastructure complexity.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| LLM | Qwen3 8B Q4_K_M via Ollama |
| Image generation | ComfyUI + FLUX.1 Schnell |
| TTS | Qwen3-TTS CustomVoice |
| Alignment | WhisperX |
| Video rendering | FFmpeg |
| UI | Gradio |
| Deployment | Docker Compose, single app container |
| Configuration | YAML + .env |

Use commercially permitted models and dependencies. Verify licenses of model weights separately from repository licenses.

## Architecture Rules

- Use one sequential Python pipeline.
- Process only one generation request at a time.
- CLI and Gradio must call the same pipeline.
- Support `ru`, `en`, and `ru,en`.
- RU and EN use separate scripts, voices, subtitles, and scene timings.
- Generate shared image assets only once.
- Supported content: `auto`, `what_if`, `science`, `facts`, `story`, `news`.
- News content must be supplied manually; do not invent current events.
- Do not implement RSS ingestion yet. Keep a simple ContentProvider interface.
- Store intermediate assets and final videos in `outputs/{run_id}/`.
- Support simple file-based resume using `manifest.json`.
- Avoid unnecessary abstraction and overengineering.

**Do NOT introduce** Redis, Celery, RQ, SQLite, Alembic, Kubernetes, distributed queues, user accounts, complex APIs, or microservices unless explicitly requested.

## GPU and Process Management

The server has **one GPU with 24 GB VRAM**.

Never assume all AI models fit in memory simultaneously.

Use `GPUProcessManager` to run heavy components sequentially:

1. Ollama / Qwen3 — script generation.
2. ComfyUI / FLUX — image generation.
3. Qwen3-TTS worker — voice generation.
4. WhisperX worker — alignment.
5. FFmpeg — video rendering.

Unload or terminate each GPU component before starting the next.

- Ollama: `keep_alive=0` or explicit unload.
- ComfyUI: `/free`, then process shutdown.
- TTS and WhisperX: isolated subprocesses.
- Never mount Docker socket.
- Never rely only on `torch.cuda.empty_cache()`.
- Use timeouts and controlled cleanup.
- Handle CUDA OOM with clear error messages.
- Do not kill unrelated GPU processes.

TTS and WhisperX use file-based JSON IPC under `outputs/{run_id}/tmp/`, with exit codes 0/1. No HTTP service is required for these workers.

## Video Requirements

Output:
- MP4, H.264, AAC.
- 1080×1920, 30 FPS.
- Vertical 9:16.
- 30, 45 or 60 seconds target duration.

Scene count:

`clamp(ceil(duration_seconds / 5), 8, 12)`

Use:
- Cinematic AI images.
- Ken Burns zoom/pan.
- Smooth transitions.
- Speech-synchronized subtitles.
- Word highlighting where alignment succeeds.
- Safe subtitle area: y=150..1520 at 1080×1920.
- Optional licensed music, disabled by default.
- Audio normalization targeting -14 LUFS and true peak ≤ -1 dBTP.

Render RU and EN independently using their actual audio timings.

## Model and Dependency Rules

- Pin dependency versions and ComfyUI Git commit.
- Keep model revisions in `models.lock.yaml`.
- Verify Hugging Face checkpoints via `HfApi().model_info()`.
- Do not invent model names, URLs, revisions, voice IDs, or licenses.
- Use FLUX Schnell with a compatible ComfyUI API workflow.
- Prefer FP8/offloading for 24 GB VRAM.
- Do not download model weights into Git or Docker images.
- Model files must persist across container restarts.
- Never silently replace a model with a lower-quality or non-commercial alternative.

If a model cannot be verified, document the blocker explicitly.

## Testing Rules

Always run applicable tests after changing code.

Required:
- pytest unit tests for core logic.
- Mocked API tests for external local services.
- ComfyUI workflow JSON structural validation.
- FFmpeg integration test using generated PNG/WAV fixtures.
- `docker compose config` for deployment changes.
- Shell script syntax checks for installation changes.

Tests must not require GPU unless explicitly marked as GPU integration tests.

Never claim GPU inference is verified if no real GPU test was performed.

Use these result labels:
- `VERIFIED`
- `MOCK VERIFIED`
- `REQUIRES GPU`
- `BLOCKED`

If a tool or service is unavailable, report it rather than fabricating success.

## Coding Guidelines

- Keep code simple, modular and readable.
- Use type hints and Pydantic for structured AI output.
- Prefer standard Python libraries where practical.
- Use pathlib for filesystem paths.
- Use subprocess argument arrays, not `shell=True`, for commands involving user inputs.
- Set explicit timeouts for network requests and subprocesses.
- Log errors with enough context to debug failures.
- Validate generated files before marking stages complete.
- Preserve valid intermediate files when a later stage fails.
- Avoid unnecessary dependencies.
- Do not add placeholders or stubs that silently pretend to work.

## Docker and Setup

Target installation:

```bash
git clone <repository_url>
cd ai-shorts-generator
./setup.sh
```

`setup.sh` must be repeatable and support:
- `--dry-run`
- `--check-only`
- `--skip-models`

Use a single GPU-enabled `app` service in Docker Compose.

Keep the UI private by default and document SSH tunneling.

Do not modify host NVIDIA drivers from inside Docker.

The first installation must download required models; subsequent runs must reuse them.

## Performance Target

Target: **≤15 minutes for one 45-second video on RTX 3090 24 GB**, with models downloaded and no AI-video generation.

This is a target, not a guarantee.

Measure and report actual stage durations. Do not invent benchmark numbers.

## Documentation

Maintain:
- `README.md` — in Russian.
- `THIRD_PARTY_LICENSES.md`.
- `models.lock.yaml`.
- `docs/GPU_SMOKE_TEST.md`.

Update documentation whenever commands, models, configuration or installation procedures change.

Every command in README must correspond to real project functionality.

## Working Instructions for Codex

Before implementing a change:
1. Inspect the relevant existing files.
2. Reuse existing architecture and conventions.
3. Make the smallest change that fully solves the task.
4. Do not rewrite working components unnecessarily.

After implementing:
1. Run applicable tests.
2. Fix failures introduced by the change.
3. Summarize modified files.
4. Report which checks actually ran.
5. Identify remaining blockers honestly.

**The success criterion is a working, locally deployable Shorts generator — not an elaborate architecture.**
