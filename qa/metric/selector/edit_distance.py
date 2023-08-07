from .base import MetricSelector
from ...text_processor.tokenizer import get_default_tokenizer

class EditDistance(MetricSelector):
    
    def __init__(self, tokenizer = None, uncased = True):
        self.tokenizer = tokenizer
        self.uncased = uncased

    def _select(self, lang):
        from ..algorithms.levenshtein import Levenshtein
        if self.tokenizer is None:
            self.tokenizer = get_default_tokenizer(lang)
        return Levenshtein(tokenizer=self.tokenizer, uncased=self.uncased) 