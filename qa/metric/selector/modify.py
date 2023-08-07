from .base import MetricSelector
from ...text_processor.tokenizer import get_default_tokenizer

class ModificationRate(MetricSelector):
    
    def __init__(self, tokenizer = None, uncased = True):
        self.tokenizer = tokenizer
        self.uncased = uncased

    def _select(self, lang):
        from ..algorithms.modification import Modification
        return Modification(get_default_tokenizer(lang), uncased=self.uncased)