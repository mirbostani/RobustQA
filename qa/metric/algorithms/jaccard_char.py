from .base import AttackMetric
from ...tags import *

class JaccardChar(AttackMetric):

    NAME = "Jaccard Char Similarity"
    TAGS = { * TAG_ALL_LANGUAGE }

    def __init__(self, uncased = True):
        self.uncased = uncased

    def calc_score(self, sentA : str, sentB : str) -> float:
        """
        Computes the Jaccard char similarity of two sentences.
        """
        if self.uncased:
            sentA = sentA.lower()
            sentB = sentB.lower()
        AS=set()
        BS=set()
        for i in range(len(sentA)):
            AS.add(sentA[i])
        for i in range(len(sentB)):
            BS.add(sentB[i])

        return len(AS&BS)/len(AS|BS)
    
    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score(input["question"], adversarial_sample["question"])
        
        return None
    