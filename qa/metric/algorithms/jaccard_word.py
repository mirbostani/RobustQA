from typing import List
from .base import AttackMetric
from ...tags import *
from ...text_processor.tokenizer import Tokenizer

class JaccardWord(AttackMetric):
    
    NAME = "Jaccard Word Similarity"

    def __init__(self, tokenizer: Tokenizer, uncased = True):
        self.tokenizer = tokenizer
        self.uncased = uncased
    
    @property
    def TAGS(self):
        if hasattr(self.tokenizer, "TAGS"):
            return self.tokenizer.TAGS
        return set()

    def calc_score(self, tokenA: List[str], tokenB: List[str]) -> float:
        """
        Computes the Jaccard word similarity of two sentences.
        """

        AS=set()
        BS=set()
        for i in range(len(tokenA)):
            AS.add(tokenA[i])
        for i in range(len(tokenB)):
            BS.add(tokenB[i])

        return len(AS&BS)/len(AS|BS)
    
    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score(
                self.tokenizer.tokenize(input["question"],
                                        pos_tagging=False),
                self.tokenizer.tokenize(adversarial_sample["question"],
                                        pos_tagging=False)
            )
            
        return None
