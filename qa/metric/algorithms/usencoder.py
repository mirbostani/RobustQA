from .base import AttackMetric
import numpy as np
from ...tags import *
from OpenAttack.data_manager import DataManager


class UniversalSentenceEncoder(AttackMetric):

    NAME = "Semantic Similarity"

    TAGS = {TAG_English}

    def __init__(self, uncased=True):
        """
        Universal Sentence Encoder in tensorflow_hub.

        pdf: https://arxiv.org/pdf/1803.11175
        page: https://tfhub.dev/google/universal-sentence-encoder/4
        """

        import tensorflow_hub as hub

        # path: "./data/AttackAssist.UniversalSentenceEncoder/usencoder"
        self.embed = hub.load(DataManager.load(
            "AttackAssist.UniversalSentenceEncoder"))

        self.uncased = uncased

    def calc_score(self, sentA: str, sentB: str) -> float:
        """
        Computes the Cosine distance between two sentences.
        """
        if self.uncased:
            sentA = sentA.lower()
            sentB = sentB.lower()
        ret = self.embed([sentA, sentB]).numpy()
        return ret[0].dot(ret[1]) / (np.linalg.norm(ret[0]) * np.linalg.norm(ret[1]))

    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score(input["question"], adversarial_sample["question"])
        
        return None
