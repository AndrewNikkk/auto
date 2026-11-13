# Фактически выполненные проверки

Дата: 2026-11-13. Итоговый pytest: **67 passed**, включая 61 unit/mock и 6 FFmpeg integration tests. Среда CPU: Linux, Python 3.11.18, FFmpeg 8.0.1. NVIDIA GPU отсутствует. Исходный каталог содержал только AGENTS.md; рабочая Git metadata в окружении недоступна, commit не создавался.

## VERIFIED

- Python compile/import приложения и CLI `--help`.
- Unit tests с реальными файловыми операциями: Pydantic/config, режимы языков, news validation, PNG, hashes/resume, ASS/SRT, лицензии/revisions lock и single-service deployment structure.
- Настоящие subprocess lifecycle tests: регистрация PID/UID/cmdline/create_time, shutdown, orphan cleanup, проверка несовпадения identity, watchdog timeout с завершением worker и его дочернего процесса.
- Реальный JSON IPC failure path: `python -m app.tts_worker`, выход 1 при отсутствии весов, JSON error и удаление PID-записи. Это не тест TTS inference.
- Шесть реальных FFmpeg integration tests на динамических testsrc/sine PNG/WAV: MP4 H.264/AAC, 1080×1920, 30 FPS, RU/EN/RU+EN, независимые длительности, отсутствие лишнего языка, xfade, loop/trim/fade короткой музыкальной sine-дорожки, измерение loudness/true peak конечного AAC, libx264 при недоступном NVENC, измеренные bounding boxes обычных и длинных субтитров в safe zone.
- CPU imports настоящих qwen-tts 0.1.1 / WhisperX 3.7.4 / transformers 4.57.3; использовались CPU wheels torch/torchaudio/torchvision 2.8.0/2.8.0/0.23.0. GPU-образ использует CUDA wheels из lock.
- Реальный ComfyUI v0.3.26 (commit 9aac21f894a122ddb8d825c57ad61c0db5e630db) запущен с `--cpu`; HTTP readiness и `/object_info` подтвердили все class_type, имена input и enum-параметры workflow. Веса и image inference не загружались.
- Gradio Blocks сконструирован (22 элемента), зависимости UI импортируются.
- `docker compose config --quiet`: официальный standalone Compose v2.40.3. Установленный Docker daemon 29.8.0 также доступен вне sandbox.
- `bash -n` для setup/download/start/stop/smoke и `setup.sh --dry-run`.
- HfApi().model_info(files_metadata=True) для FLUX/VAE/CLIP/T5/TTS/RU/EN alignment: существование, полные revisions, лицензии, размеры, LFS SHA256. Для маленьких config/tokenizer файлов дополнительно рассчитан SHA256. Проверен TTS config с speaker ryan и language IDs.
- Официальные GitHub commit ComfyUI, Ollama 0.13.5 binary SHA256 и Ollama registry model digest зафиксированы без выдуманных значений.
- Статический Ruff F-check без ошибок. Это вспомогательная проверка, Ruff не требуется для запуска приложения.

## MOCK VERIFIED

- Ollama structured JSON Schema, temperature, отдельные system prompts RU/EN, общая структура, три попытки validation с передачей ошибки, остановка при неправильном числе сцен, auto classification cache и пропуск классификации для news.
- ComfyUI HTTP через httpx.MockTransport: отправка workflow, history/view, execution error/OOM, image timeout/interrupt. Никакие unit tests не обращаются к живому ComfyUI.
- Полный pipeline RU/EN/RU+EN с mocks AI/renderer: manifest/report, независимые WAV-тайминги, повторный запуск без повторения стадий, восстановление только повреждённого PNG, архив старого языка при переключении.
- Qwen3-TTS adapter: проверка голосов/языков, повтор только проблемного сегмента, WAV concatenation, сохранение и восстановление segment cache. Mock возвращает синус, это не озвучка.
- WhisperX adapter: word result, неполный alignment, fallback на реальные TTS-сегменты и немедленный fail при CUDA OOM.
- File-based worker success/error/timeout, readiness retry.

## REQUIRES GPU

- Настоящий Qwen3/Ollama inference и качество/соответствие фактов сценариев.
- FLUX inference и визуальное качество 832×1216 на 24 GB.
- Qwen3-TTS BF16/SDPA, RU/EN Ryan, скорость и качество речи.
- Реальная совместимость RU/EN checkpoints с forced alignment на аудио модели.
- NVENC на NVIDIA GPU, фактический peak VRAM и освобождение памяти между реальными моделями.
- Полный GPU smoke и benchmark ≤15 минут для 45 s: **не выполнялись**.
- Итоговая громкость/true peak AAC на реальном речевом материале и музыка на полном ролике.

Полная матрица — `docs/GPU_SMOKE_TEST.md`. Нет гарантии, что первый GPU-запуск не выявит аппаратных/модельных проблем. Metadata и CPU API checks не заменяют inference.

## Docker build

Реальная сборка выполнена. Первая попытка через legacy builder зависла на HTTP Debian download; остановлен только собственный промежуточный build-контейнер. Источники изменены на HTTPS с сетевыми таймаутами, сборка повторена через BuildKit. Обе повторные сборки завершились с кодом 0. Финальная версия: `ai-shorts-generator:mvp-verified`, manifest digest `sha256:f1476a1ad428d8a0685e4eb4d58c369e6671cf98077913aced360b4e8a55276d`.

**VERIFIED:** образ реально запущен без GPU. Проверены UID 1000, запись extra_model_paths из entrypoint, сборка Gradio Blocks, настоящие imports Qwen3-TTS/WhisperX, генерация PNG/WAV fixtures и конечного 1080×1920 H.264/AAC MP4 с проверкой loudness/true peak. Отсутствующий NVENC корректно переключился на libx264. Это CPU-проверка контейнера, не GPU inference.

**VERIFIED:** финальный образ также запущен с настоящим Gradio HTTP server. Запросы `/` и `/config` вернули корректные ответы с video components; порт опубликован только на host loopback. Созданный тестовый контейнер остановлен после проверки. Никакие модели при HTTP-проверке не запускались.
