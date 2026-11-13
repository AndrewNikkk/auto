# Qwen3-TTS: статус проверки

Блокера существования checkpoint нет: `HfApi().model_info()` подтвердил
`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, revision
`0c0e3051f131929182e2c023b9537f8b1c68adfe`, Apache-2.0.
Config содержит `ryan` и language IDs `russian`, `english`.
Источник: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

Реализован настоящий Qwen3TTSModel adapter через isolated JSON worker, stub не используется.
Пакет `qwen-tts==0.1.1`, transformers 4.57.3, accelerate 1.12.0.
Runtime API speakers/languages проверяется перед генерацией. При неподдерживаемом голосе — ошибка,
без тихой замены. Оригинальный speaker Ryan может иметь акцент на русском: требуется прослушивание.

**REQUIRES GPU:** синтез на RTX 3090/4090, качество RU/EN, время, peak VRAM,
поведение checkpoint с SDPA/BF16. Проверенная metadata не означает проверенный inference.
