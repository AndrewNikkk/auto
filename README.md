# AI Shorts Generator

Локальный однопользовательский генератор вертикальных Shorts для RTX 3090/4090 24 GB. Промпт → Qwen3 → общие FLUX-изображения → отдельная озвучка и forced alignment RU/EN → MP4. Платные AI API не нужны. CLI и Gradio вызывают один последовательный `Pipeline`; файловая блокировка исключает одновременные генерации даже из разных процессов.

**Статус:** исполняемый MVP подготовлен для проверки на GPU. CPU/FFmpeg, Docker build и mocked-проверки описаны в `docs/VALIDATION.md`. Inference на GPU и целевой benchmark пока не подтверждены. Это не заявление о проверенной работе на RTX 3090/4090.

## Сервер

- Linux x86_64, одна NVIDIA RTX 3090/4090 с 24 GB VRAM.
- NVIDIA driver с поддержкой CUDA 12.8 (для закреплённых PyTorch wheels), Docker Engine, Compose v2, NVIDIA Container Toolkit.
- Минимум 32 GiB RAM; 64 GiB рекомендуется для FLUX offloading.
- Минимум 100 GiB свободного места в Docker storage. Практичнее иметь 150–200 GiB для layers, кеша и результатов.
- Интернет при первой сборке/загрузке моделей. Для FLUX может понадобиться принять условия Hugging Face и указать `HF_TOKEN`.

Драйверы и NVIDIA Container Toolkit устанавливаются на хост по [официальной инструкции NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Скрипты проекта не меняют драйверы. Docker socket в контейнер не монтируется.

Образ может занимать десятки гигабайт, ориентир 30–50 GB; первая сборка может занять 30+ минут. Это оценки, не измерения этой сборки.

## Быстрая установка

Получите проект из своего Git remote или скопируйте каталог на сервер. URL репозитория не задан и не выдуман.

```bash
./setup.sh
```

Скрипт сохраняет существующий `.env`, проверяет GPU/RAM/свободное место, собирает один контейнер `app`, проверяет GPU и реальное CUDA/BF16 выделение PyTorch внутри него, скачивает и проверяет веса, запускает Gradio и ждёт HTTP readiness. При ошибке обязательного шага завершает работу с ненулевым кодом.

```bash
./setup.sh --dry-run
./setup.sh --check-only
./setup.sh --skip-models
```

`--dry-run` только печатает действия, ничего не скачивает и не доказывает готовность. `--check-only` проверяет существующий образ, GPU и файлы моделей; образ должен быть уже собран. `--skip-models` пропускает загрузку, но требует валидные существующие модели. `.env.example` описывает пути **внутри** persistent volume `/data`; не задавайте там пути хоста. Пользовательские изменения конфигурации YAML включаются в образ: после изменений выполните сборку и пересоздание контейнера.

```bash
docker compose build app
docker compose up -d app
```

Gradio слушает `0.0.0.0:7860` внутри контейнера, порт хоста публикуется только на `127.0.0.1`. `GRADIO_HOST` управляет адресом публикации хоста, `GRADIO_PORT` — его портом. Ollama/ComfyUI слушают loopback контейнера, их порты наружу не публикуются. Публичного sharing нет.

С локального компьютера:

```bash
ssh -L 7860:localhost:7860 USER@SERVER_IP
```

Откройте [локальный интерфейс](http://localhost:7860). `USER` и `SERVER_IP` замените своими значениями.

## Первый ролик и CLI

Зависимости Python на хост не требуются:

```bash
docker compose exec app python -m app.generate \
  --prompt "Что будет, если все корабли исчезнут?" \
  --languages ru --duration 30 --type what_if
```

```bash
docker compose exec app python -m app.generate \
  --prompt "Почему лёд плавает в воде?" \
  --languages ru,en --duration 45 --type science \
  --style cinematic --seed 42
```

Языки: `ru`, `en`, `ru,en`. RU+EN — отдельные вызовы сценариста с общими ключевыми фактами/структурой и разными system prompts, не буквальный перевод. Изображения берутся из первого выбранного языка и генерируются один раз. WAV, ASS, SRT, длительности сцен и MP4 самостоятельные.

Длительность 30/45/60 секунд задаёт **цель сценария**, а окончательная длительность определяется фактической речью. Текст не растягивается искусственно до цели. Для этих целей используется 8/9/12 изображений соответственно. FLUX seed влияет на изображения; LLM seed не фиксируется.

Типы: `auto`, `what_if`, `science`, `facts`, `story`, `news`. Auto делает отдельную enum-классификацию с файловым кешем. `story` разрешает художественный сюжет; `what_if` оформляется как мысленный эксперимент. News требует содержательный исходный текст (минимум 100 символов и 15 слов), а не просьбу найти новости.

```bash
docker compose exec app python -m app.generate \
  --prompt "Полный текст новости, предоставленный пользователем. Вставьте здесь реальные сведения: что произошло, кто участвовал и какие подробности известны из исходного материала. Не заменяйте исходный текст одним URL." \
  --languages ru --duration 30 --type news \
  --source-url https://example.org/source
```

В этом примере `example.org` — демонстрационная ссылка. Источники сохраняются в metadata без скачивания и проверки; наличие URL не доказывает достоверность. Явный news пропускает классификацию. RSS и автоматический поиск отложены.

## Модели и воспроизводимость

`models.lock.yaml` содержит реальные полные ревизии, имена файлов, размеры и SHA256. HF checkpoints проверены через `HfApi().model_info(files_metadata=True)`; маленькие config/tokenizer файлы также получили SHA256. Снимок проверки — `docs/model-verification.json`. Репозиторные размеры там могут включать альтернативные форматы; загружаемый объём указан в lock.

| Набор | Загружаемый объём, примерно | Размещение |
|---|---:|---|
| Qwen3 8B Q4_K_M | 5.23 GB + metadata | Ollama persistent store |
| Оригинальный FLUX Schnell | 23.78 GB | `comfyui/diffusion_models/` |
| FLUX VAE | 0.34 GB | `comfyui/vae/` |
| CLIP-L + T5 FP8 | 5.14 GB | `comfyui/text_encoders/` |
| Qwen3-TTS 1.7B CustomVoice с speech tokenizer | 4.52 GB | HF local snapshot |
| Alignment RU + EN | 2.52 GB | HF local snapshots |

FLUX загружается из оригинального BF16 checkpoint и приводится к `fp8_e4m3fn` нодой UNETLoader; T5 — настоящий FP8-файл из `comfyanonymous/flux_text_encoders`. На диске FLUX не становится 12 GB. Workflow: 832×1216, 4 steps, Euler/simple, CFG=1.0, lowvram/offloading. Финальный кадр — Lanczos upscale и центральный crop без растягивания до 1080×1920. Prompts требуют держать важные объекты в центре; автоматического распознавания объектов для crop нет.

ComfyUI закреплён на commit версии v0.3.26 без БД; frontend dependency обновлена до существующей 1.14.6, используется только API. API schema проверяется перед генерацией через `/object_info`. `requirements.lock` фиксирует транзитивные зависимости. WhisperX устанавливается отдельно с `--no-deps`: используются только forced alignment/audio, без ASR, diarization, pyannote, Optuna, SQLAlchemy/Alembic. Это намеренно ограниченная установка; полный WhisperX CLI не поддерживается.

Повторная загрузка пропускает файлы с правильным размером и SHA256, кеш HF и рабочие snapshots используют hardlinks при возможности. Ollama tag сверяется с закреплённым digest: изменение upstream tag вызывает ошибку вместо тихого обновления.

```bash
./download_models.sh
./download_models.sh --check-only
```

Веса не входят в Git/образ. Named volume `shorts_data` сохраняет модели, музыку и результаты после пересоздания контейнера. Не удаляйте его для обычного перезапуска.

## Голоса, alignment и субтитры

`config/voices.yaml`: Ryan / Russian и Ryan / English. Checkpoint подтверждён; config содержит speaker `ryan` и русский/английский language IDs (`docs/tts-config-verified.json`). Worker проверяет реальные `get_supported_speakers()` и `get_supported_languages()` до синтеза, не меняет голос молча. Ryan — англоязычный голос с multilingual synthesis; естественность русского произношения требует прослушивания на GPU.

Каждая сцена — отдельный WAV с реальными границами. Повторяются только невалидные сегменты, максимум две дополнительные попытки; OOM не повторяется. У каждого сегмента проверяются sample rate, длительность, отсутствие тишины и целостность кеша.

WhisperX использует известный текст и закреплённые RU/EN wav2vec2 checkpoints. WhisperX 3.7.4 принудительно запрашивает обученный английский Punkt splitter; адаптер приложения сохраняет целый TTS-сегмент как одну группу предложений. Акустическая модель и word alignment не заменяются, tokenizer weights не скачиваются. Неполный alignment переводит весь язык в segment-level captions, используя реальные WAV-границы. Причина записывается в отчёт; вымышленных word timestamps нет. Успешный alignment включает подсветку текущего слова. UTF-8 SRT и ASS сохраняются отдельно. ASS использует DejaVu Sans, боковые поля 90 px и нижнее поле 420 px внутри safe zone `y=150..1520`.

## Монтаж и музыка

Рендер: отдельные сцены с Ken Burns → последовательные двухклиповые xfade → финальный проход ASS/аудио. Один огромный filter graph не строится. Проверяется наличие NVENC и настоящий тестовый encode; при недоступности выбирается libx264. Итог проверяется ffprobe: H.264/AAC, 1080×1920, 30 FPS и длительность озвучки.

Финальный микс проходит двухпроходный `loudnorm I=-14:TP=-1:LRA=11` и limiter. Итоговый AAC измеряется повторно: при превышении -1 dBTP выполняется один корректирующий аудиопроход без перекодирования видео и повторная проверка. Если true peak всё ещё превышен, этап завершается ошибкой. Измерения сохраняются в отчёте.

Музыка выключена; треки не поставляются. Используйте только файлы с подтверждённой лицензией. Положите их в `/data/assets/music` контейнера, например:

```bash
docker compose cp /absolute/path/licensed-track.wav app:/data/assets/music/track.wav
```

В UI нажмите «Обновить музыку». Поддержаны mp3/m4a/wav/ogg; список также обновляется при открытии. Короткая музыка зацикливается, длинная обрезается, последние 1.5 s — fade-out.

```bash
docker compose exec app python -m app.generate \
  --prompt "Почему небо голубое?" --languages en --duration 30 --type science \
  --music /data/assets/music/track.wav
```

## GPU, ошибки и таймауты

| Этап | Оценка VRAM, не максимум | Timeout по умолчанию |
|---|---:|---:|
| Qwen3 8B Q4 | ~7 GB | 120 s для LLM-вызовов |
| FLUX Schnell FP8 | ~14 GB | 600 s, 90 s / изображение |
| Qwen3-TTS 1.7B | ~5 GB | 300 s |
| WhisperX alignment | ~3 GB | 180 s |
| Render | зависит от encoder | 300 s / язык |

Readiness имеет отдельный бюджет 90 s и один повтор запуска. Таймауты задаются в `config/settings.yaml` (после изменения пересоберите образ) или через `TIMEOUT_SCRIPT`, `TIMEOUT_IMAGES`, `TIMEOUT_IMAGE`, `TIMEOUT_TTS`, `TIMEOUT_SUBTITLES`, `TIMEOUT_RENDER`, `TIMEOUT_READINESS` в окружении процесса. TTS/WhisperX timeout охватывает оба выбранных языка вместе. После дедлайна watchdog завершает только зарегистрированную группу worker-процесса; дополнительно отводится до 15 s на cleanup.

Процессы: Ollama → явный unload + shutdown → ComfyUI `/free` + SIGTERM/SIGKILL → TTS subprocess → alignment subprocess → FFmpeg. PID, cmdline, UID и время запуска сохраняются в `.gpu-procs`; чужие процессы не завершаются. Проверяются зарегистрированные orphan-процессы. Живой зарегистрированный Ollama при следующем запуске останавливается, а не переиспользуется. Никакой Docker API из приложения нет.

NVML семплируется раз в 500 ms: used/total, peak по этапу, память до/после и проверка возврата к baseline с допуском 512 MiB. Сторонняя нагрузка может вызвать `VRAM_NOT_RELEASED`; приложение сообщает об этом, не убивает чужой процесс. `nvidia-smi` используется только для диагностики.

CUDA OOM завершает этап без слепого retry. Для FLUX проверьте отсутствие чужой GPU-нагрузки, offloading, свободную RAM; при необходимости явно уменьшите 832×1216 preset с кратностью 16 и пересоберите. Никакая модель автоматически не заменяется. Comfy crash допускает один полный restart с первого отсутствующего/повреждённого изображения; два последовательных image timeout останавливают этап. stderr workers и ошибки FFmpeg попадают в `generation_report.json`.

## Результаты и resume

`/data/outputs/{run_id}/`:

```text
images/  audio/  subtitles/  tmp/
script.json
manifest.json
video_ru.mp4       # только выбранные языки
video_en.mp4
generation_report.json
```

Manifest хранит параметры, model lock, hashes и per-language timings. Валидируются PNG (>10 KB, Pillow, точные размеры, pixel std), WAV/тайминги и MP4. Повреждённое изображение восстанавливается отдельно. Изменение сценария или входов инвалидирует зависимые стадии. При смене параметров безопаснее использовать новый run ID. При успешной смене языков невыбранные старые MP4 перемещаются в `tmp/archive/`, поэтому в корне остаются только актуальные языки.

Повторите тот же запрос с прежним идентификатором:

```bash
docker compose exec app python -m app.generate \
  --prompt "Почему лёд плавает в воде?" --languages ru,en --duration 45 --type science \
  --run-id YOUR_PREVIOUS_RUN_ID
```

Отчёт содержит фактическое wall-clock время, GPU/peak VRAM, retries, fallback, отдельные границы сцен языков. При resume время переиспользованных стадий равно нулю; первоначальные аудиотайминги сохраняются. Цель ≤15 минут на 45 s RTX 3090 пока **не измерена**.

```bash
docker compose cp app:/data/outputs/YOUR_RUN_ID ./my-result
./stop.sh
```

## Проверки

CPU development требует Python 3.11:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
bash -n setup.sh download_models.sh start.sh stop.sh gpu-smoke-test.sh
docker compose config --quiet
```

Unit/API tests не требуют GPU; FFmpeg integration fixtures PNG/WAV создаются на лету из `testsrc`/`sine`. Тестируется реальное положение rendered subtitles.

```bash
./gpu-smoke-test.sh
```

Полный GPU smoke — [docs/GPU_SMOKE_TEST.md](docs/GPU_SMOKE_TEST.md). Он генерирует RU+EN ролик, проверяет картинки, озвучку обоих языков, forced alignment без fallback, финальные MP4, stage durations и освобождение VRAM.

## Troubleshooting и ограничения

- `NEWS_CONTENT_REQUIRED`: нужен исходный текст, одного URL недостаточно.
- `MODEL_CHECK_FAILED` / HF 401/403: примите условия FLUX, задайте HF_TOKEN в `.env`, повторите downloader. Не включайте токен в Git.
- `OLLAMA_DIGEST_CHANGED`: upstream tag изменился; проверьте новую версию вручную и обновите lock осознанно.
- `PORT_IN_USE`: приложение не подключается к чужим Ollama/ComfyUI; освободите внутренние 11434/8188.
- `GENERATION_BUSY`: дождитесь текущего запуска; в MVP нет очереди между процессами.
- `UNSUPPORTED_SPEAKER/LANGUAGE`: проверьте voices.yaml и фактический checkpoint, молчаливого fallback нет.
- NVENC отсутствует: libx264 — штатный CPU fallback; рендер может быть медленнее.
- WhisperX fallback: смотрите `alignment_reason`; субтитры останутся на реальных segment-level границах без подсветки слов.
- Docker/GPU не виден: проверьте toolkit и драйвер на хосте, не пытайтесь установить драйвер внутрь контейнера.

Качество фактов зависит от исходного материала и сценариста. Нет автоматического fact-checking, RSS, AI-видеогенерации, пользователей и базы данных. Первая GPU-проверка должна включать прослушивание RU/EN, просмотр crop/субтитров и замер громкости AAC. Коммерческие лицензии кода и весов перечислены отдельно в [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
