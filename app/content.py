import re
from typing import Protocol
from app.schema import ResearchResult

class ContentProvider(Protocol):
    def fetch(self, prompt: str, article_urls: list[str]) -> ResearchResult: ...

class ManualPromptProvider:
    def fetch(self, prompt: str, article_urls: list[str]) -> ResearchResult:
        return ResearchResult(text=prompt.strip(), source_urls=article_urls, verified=False)


def require_news_text(research: ResearchResult) -> None:
    text = re.sub(r'https?://\S+', '', research.text).strip()
    if len(text) < 100 or len(text.split()) < 15:
        raise ValueError('NEWS_CONTENT_REQUIRED: вставьте текст новости (минимум 100 символов и 15 слов), URL недостаточно')
