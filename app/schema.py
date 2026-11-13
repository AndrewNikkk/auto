from __future__ import annotations
import math
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ContentType = Literal['auto', 'what_if', 'science', 'facts', 'story', 'news']
Language = Literal['ru', 'en']

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Params(StrictModel):
    prompt: str = Field(min_length=1, max_length=20000)
    languages: list[Language] = Field(default_factory=lambda: ['ru'], min_length=1)
    duration: Literal[30, 45, 60] = 45
    content_type: ContentType = 'auto'
    style: Literal['cinematic', 'realistic', 'illustrated'] = 'cinematic'
    seed: int = Field(default=42, ge=0, le=2**63-1)
    music: Path | None = None
    source_urls: list[str] = Field(default_factory=list)

    @field_validator('languages')
    @classmethod
    def unique_languages(cls, value):
        if len(value) != len(set(value)):
            raise ValueError('Duplicate languages')
        return value

    @property
    def image_count(self) -> int:
        return max(8, min(12, math.ceil(self.duration / 5)))

class ResearchResult(StrictModel):
    text: str
    source_urls: list[str] = Field(default_factory=list)
    verified: bool = False

class Classification(StrictModel):
    content_type: Literal['what_if', 'science', 'facts', 'story', 'news']

class Scene(StrictModel):
    narration: str = Field(min_length=1, max_length=600)
    description: str = Field(min_length=1)
    visual_prompt: str = Field(min_length=1)
    duration_seconds: float = Field(gt=0, le=20)
    key_fact: str = Field(min_length=1)

class Script(StrictModel):
    title: str = Field(min_length=1)
    language: Language
    scenes: list[Scene] = Field(min_length=8, max_length=12)
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def narration(self) -> str:
        return ' '.join(s.narration for s in self.scenes)

class Segment(StrictModel):
    text: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    path: Path

    @model_validator(mode='after')
    def ordered(self):
        if self.end <= self.start:
            raise ValueError('Invalid segment interval')
        return self

class AudioResult(StrictModel):
    audio: Path
    sample_rate: int = Field(ge=16000, le=96000)
    segments: list[Segment] = Field(min_length=1)

class Word(StrictModel):
    text: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode='after')
    def ordered(self):
        if self.end <= self.start:
            raise ValueError('Invalid word interval')
        return self

class AlignmentResult(StrictModel):
    words: list[Word] = Field(default_factory=list)
    fallback: Literal['segment_level'] | None = None
    reason: str | None = None

class WorkerInput(StrictModel):
    run_dir: Path
    languages: list[Language]
    timeout: float = Field(gt=0)

class TTSOutput(StrictModel):
    languages: dict[Language, AudioResult]
    retries: int = 0
    timings: dict[Language, float] = Field(default_factory=dict)

class AlignmentOutput(StrictModel):
    languages: dict[Language, AlignmentResult]
    timings: dict[Language, float] = Field(default_factory=dict)
