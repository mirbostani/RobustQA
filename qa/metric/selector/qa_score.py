from .base import MetricSelector
from ...victim.question_answering.transformers import TransformersQuestionAnswering

class QAScore(MetricSelector):

    def __init__(self, victim: TransformersQuestionAnswering, metric: str, sample_type: str):
        self.victim = victim
        self.metric = metric
        self.sample_type = sample_type

    def _select(self, lang):
        if self.metric == "f1":
            from ..algorithms.f1_score import F1Score
            return F1Score(victim=self.victim, sample_type=self.sample_type)
        elif self.metric == "em":
            from ..algorithms.exact_match import ExactMatch
            return ExactMatch(victim=self.victim, sample_type=self.sample_type)