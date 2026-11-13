import time
from pathlib import Path
import httpx
from pydantic import ValidationError
from app.config import ROOT, Settings
from app.content import require_news_text
from app.files import fingerprint, read_json, write_json
from app.schema import Classification, Params, ResearchResult, Script

class ScriptGenerator:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self.client = client or httpx.Client(base_url=settings.ollama_url, timeout=settings.timeouts.script)
        self.retries = 0
        self.deadline = time.monotonic() + settings.timeouts.script
        self.timings = {}

    def structured(self, schema, system: str, prompt: str, log_path: Path, validate=lambda x: x):
        messages = [{'role': 'system', 'content': system + '\n/no_think\nReturn only JSON.'},
                    {'role': 'user', 'content': prompt}]
        deadline = self.deadline
        for attempt in range(3):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Script timeout')
            response = self.client.post('/api/chat', timeout=remaining, json={
                'model': self.settings.llm_model, 'messages': messages, 'stream': False,
                'think': False, 'format': schema.model_json_schema(), 'keep_alive': 0,
                'options': {'temperature': .7, 'num_ctx': 8192}})
            response.raise_for_status()
            raw = response.json()['message']['content']
            log_path.write_text(raw, encoding='utf-8')
            try:
                value = schema.model_validate_json(raw)
                validate(value)
                return value
            except (ValidationError, ValueError) as error:
                if attempt == 2:
                    raise ValueError(f'LLM_INVALID_JSON: {error}; last response: {log_path}') from error
                self.retries += 1
                messages.extend([{'role': 'assistant', 'content': raw},
                                 {'role': 'user', 'content': f'Correct validation errors: {str(error)[:2000]}'}])

    def generate(self, params: Params, research: ResearchResult, run_dir: Path) -> dict[str, Script]:
        kind = params.content_type
        if kind == 'news':
            require_news_text(research)
        if kind == 'auto':
            key = fingerprint({'text': research.text, 'template': (ROOT/'config/prompts/classify.txt').read_text(),
                               'model': self.settings.llm_model})
            cache = self.settings.output_dir / 'cache/classify' / f'{key}.json'
            if cache.exists():
                kind = Classification.model_validate(read_json(cache)).content_type
            else:
                value = self.structured(Classification, (ROOT/'config/prompts/classify.txt').read_text(),
                                        research.text, run_dir/'tmp/classify_response.txt')
                kind = value.content_type
                write_json(cache, value.model_dump())
        if kind == 'news':
            require_news_text(research)
        template = (ROOT / f'config/prompts/{kind}.txt').read_text()
        scripts = {}
        outline = None
        for language in params.languages:
            language_started = time.monotonic()
            system = {'ru': 'Ты сценарист коротких видео. Пиши живым естественным русским языком.',
                      'en': 'You write engaging shorts in natural idiomatic English.'}[language]
            instructions = {'research': research.model_dump(), 'language': language, 'content_type': kind,
                            'scene_count': params.image_count, 'target_seconds': params.duration,
                            'target_words': round(params.duration * (2.1 if language == 'ru' else 2.5)),
                            'style': params.style, 'shared_outline': outline}
            def validate(script):
                if script.language != language or len(script.scenes) != params.image_count:
                    raise ValueError(f'Expected language={language} and exactly {params.image_count} scenes')
            script = self.structured(Script, system + '\n' + template,
                                     __import__('json').dumps(instructions, ensure_ascii=False),
                                     run_dir/f'tmp/script_{language}_response.txt', validate)
            self.timings[language] = time.monotonic() - language_started
            scripts[language] = script
            if outline is None:
                outline = [{'key_fact': s.key_fact, 'description': s.description, 'visual_prompt': s.visual_prompt}
                           for s in script.scenes]
        return scripts
