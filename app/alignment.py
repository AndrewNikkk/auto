"""Known-text alignment for presegmented audio; no ASR or tokenizer model download."""
from contextlib import contextmanager

class SceneSentenceSplitter:
    def span_tokenize(self, text: str):
        if text:
            yield 0, len(text)

@contextmanager
def scene_segmentation(module):
    # WhisperX 3.7.4 otherwise loads a pretrained English Punkt model even for Russian.
    # Our WAV segments already define the scenes; this changes sentence grouping only, not CTC alignment.
    original = module.nltk_load
    module.nltk_load = lambda resource: SceneSentenceSplitter()
    try:
        yield
    finally:
        module.nltk_load = original
