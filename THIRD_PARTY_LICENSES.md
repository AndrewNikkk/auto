# Сторонние компоненты и веса

Проверка metadata выполнена 2026-11-13. Лицензия репозитория кода не переносится автоматически на веса. Таблица фиксирует заявленные upstream лицензии; обязательства при распространении контейнеров и используемой музыки нужно соблюдать отдельно. Веса не включаются в образ или Git.

| component | version | license | commercial_use_allowed | source_url |
|---|---|---|---|---|
| Qwen3 code | upstream, модель через Ollama | Apache-2.0 | yes, subject to license | https://github.com/QwenLM/Qwen3 |
| Qwen3-8B weights | qwen3:8b-q4_K_M; digest в lock | Apache-2.0 | yes, subject to license | https://huggingface.co/Qwen/Qwen3-8B |
| Ollama code/binary | 0.13.5, SHA256 в Dockerfile | MIT | yes | https://github.com/ollama/ollama/blob/v0.13.5/LICENSE |
| ComfyUI code | 9aac21f894a122ddb8d825c57ad61c0db5e630db | GPL-3.0 | yes, subject to GPL obligations | https://github.com/Comfy-Org/ComfyUI/blob/v0.3.26/LICENSE |
| ComfyUI frontend package | 1.14.6, dependency only | GPL-3.0 | yes, subject to GPL obligations | https://github.com/Comfy-Org/ComfyUI_frontend |
| FLUX.1 Schnell diffusion weights | 741f7c3ce8b383c54771c7003378a50191e9efe9 | Apache-2.0 | yes, subject to license/model card | https://huggingface.co/black-forest-labs/FLUX.1-schnell |
| FLUX ae.safetensors weights | тот же revision, отдельный файл | Apache-2.0 (model metadata) | yes, subject to license | https://huggingface.co/black-forest-labs/FLUX.1-schnell |
| CLIP-L / T5 FP8 repacked weights | 6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5 | Apache-2.0 (repack metadata) | yes per metadata; CLIP provenance см. ниже | https://huggingface.co/comfyanonymous/flux_text_encoders |
| Original CLIP code/weights | upstream ViT-L | MIT | yes, subject to license | https://github.com/openai/CLIP/blob/main/LICENSE |
| Original T5 v1.1 XXL | upstream | Apache-2.0 | yes, subject to license | https://huggingface.co/google/t5-v1_1-xxl |
| Qwen3-TTS code | qwen-tts 0.1.1 | Apache-2.0 | yes | https://github.com/QwenLM/Qwen3-TTS |
| Qwen3-TTS CustomVoice weights | 0c0e3051f131929182e2c023b9537f8b1c68adfe | Apache-2.0 | yes, subject to license | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice |
| Embedded Qwen3 speech tokenizer weights | speech_tokenizer/ того же snapshot | Apache-2.0 (checkpoint metadata) | yes, subject to license | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/tree/main/speech_tokenizer |
| WhisperX code | 3.7.4, forced alignment only | BSD-2-Clause | yes | https://github.com/m-bain/whisperX/blob/v3.7.4/LICENSE |
| RU wav2vec2 alignment weights | 2329100508896c6d9b157019803ab5601e6f3406 | Apache-2.0 | yes | https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-russian |
| EN wav2vec2 alignment weights | 54074b1c16f4de6a5ad59affb4caa8f2ea03a119 | Apache-2.0 | yes | https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self |
| FFmpeg | Debian Bookworm apt version, build dependent | GPL-2.0-or-later with libx264; build dependent | yes, subject to build/license/patent obligations | https://ffmpeg.org/legal.html |
| libx264 | Debian build dependency | GPL-2.0-or-later | yes, subject to GPL obligations | https://code.videolan.org/videolan/x264 |
| PyTorch / torchaudio / torchvision | 2.8.0 / 2.8.0 / 0.23.0 | BSD-3-Clause / BSD-2-Clause / BSD-3-Clause | yes | https://github.com/pytorch/pytorch |
| Transformers / Accelerate / huggingface_hub | 4.57.3 / 1.12.0 / 0.36.0 | Apache-2.0 | yes | https://github.com/huggingface/transformers |
| Gradio | 5.49.1 | Apache-2.0 | yes | https://github.com/gradio-app/gradio |
| Pydantic | 2.11.10 | MIT | yes | https://github.com/pydantic/pydantic |
| httpx | 0.28.1 | BSD-3-Clause | yes | https://github.com/encode/httpx |
| PyYAML | 6.0.3 | MIT | yes | https://github.com/yaml/pyyaml |
| Pillow | 11.3.0 | HPND | yes | https://github.com/python-pillow/Pillow |
| NumPy / SciPy / pandas | versions in requirements.lock | BSD-3-Clause | yes | https://numpy.org/doc/stable/license.html |
| NLTK code | 3.9.2 | Apache-2.0 | yes | https://github.com/nltk/nltk/blob/develop/LICENSE.txt |
| DejaVu Sans | Debian fonts-dejavu-core | Bitstream Vera + public-domain changes | yes, subject to font license | https://dejavu-fonts.github.io/License.html |
| nvidia-ml-py bindings | 13.580.82 | BSD-3-Clause | yes | https://pypi.org/project/nvidia-ml-py/ |
| NVIDIA CUDA runtime / NVENC libraries | transitive wheels/host driver | NVIDIA proprietary redistribution terms | unknown — requires manual check for redistribution | https://docs.nvidia.com/cuda/eula/index.html |
| Other transitive packages | exact versions in requirements.lock | per-package licenses | unknown — requires manual check before distributing image | https://pypi.org/ |
| User-provided music | not bundled | user supplied | unknown — requires manual check | assets/music/ |

HF licenses/revisions/file sizes были получены через HfApi; доказательства — `docs/model-verification.json`, полные hashes — `models.lock.yaml`. Репак CLIP/T5 декларирует Apache-2.0, оригинальный CLIP имеет MIT; сохраните оба upstream attribution при распространении. Runtime использует только указанные модельные файлы, не скачивает ASR/diarization или Punkt weights. Для sentence grouping WhisperX получает уже существующие TTS-сегменты, акустическая модель не заменяется.

Проверка лицензий и metadata не доказывает качество, отсутствие ошибок модели или пригодность конкретного пользовательского контента. Metadata установленных Python-пакетов дополнительно сохранена в `docs/dependency-license-metadata.json`; отсутствующие CUDA-пакеты и неполная metadata явно помечены unknown. Полный юридический аудит зависимостей и патентов не выполнен; неопределённые позиции явно отмечены.
