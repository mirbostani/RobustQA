from .base import MetricSelector

class Fluency(MetricSelector):
    
    def _select(self, lang):
        if lang.name == "english":
            from ..algorithms.gptlm import GPT2LM
            return GPT2LM()