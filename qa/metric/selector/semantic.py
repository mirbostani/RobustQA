from .base import MetricSelector

class SemanticSimilarity(MetricSelector):

    def __init__(self, uncased = True):
        self.uncased = uncased
        self.sim = None

    def _select(self, lang):
        if lang.name == "english":
            
            if self.sim is None:
                from ..algorithms.usencoder import UniversalSentenceEncoder
                self.sim = UniversalSentenceEncoder(uncased=self.uncased)

            return self.sim