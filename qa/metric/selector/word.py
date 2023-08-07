from .base import MetricSelector
from ...text_processor.tokenizer import get_default_tokenizer

class JaccardWordSimilarity(MetricSelector):

    def __init__(self, tokenizer = None, uncased = True):
        self.tokenizer = tokenizer
        self.uncased = uncased

    def _select(self, lang):
        from ..algorithms.jaccard_word import JaccardWord
        if self.tokenizer is None:
            self.tokenizer = get_default_tokenizer(lang)
        return JaccardWord(tokenizer=self.tokenizer, uncased=self.uncased) 