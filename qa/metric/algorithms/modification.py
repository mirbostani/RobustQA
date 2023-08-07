from typing import List
from .base import AttackMetric
from ...text_processor.tokenizer import Tokenizer

class Modification(AttackMetric):
    
    NAME = "Word Modif. Rate"

    def __init__(self, tokenizer : Tokenizer, uncased = True):
        self.tokenizer = tokenizer
        self.uncased = uncased

    @property
    def TAGS(self):
        if hasattr(self.tokenizer, "TAGS"):
            return self.tokenizer.TAGS
        return set()
        
    def calc_score(self, tokenA : List[str], tokenB : List[str]) -> float:
        """
        Computes the Modification rate.
        """
        va = tokenA
        vb = tokenB
        if self.uncased:
            va = [x.lower() for x in va]
            vb = [x.lower() for x in vb]
        ret = 0
        if len(va) != len(vb):
            ret = abs(len(va) - len(vb))
        mn_len = min(len(va), len(vb))
        va, vb = va[:mn_len], vb[:mn_len]
        for wordA, wordB in zip(va, vb):
            if wordA != wordB:
                ret += 1
        return ret / len(va)
    
    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score( 
                self.tokenizer.tokenize(input["question"], pos_tagging=False), 
                self.tokenizer.tokenize(adversarial_sample["question"], pos_tagging=False)
            )
            
        return None
