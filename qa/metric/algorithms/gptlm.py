import math
import transformers
from ...tags import *
from .base import AttackMetric

class GPT2LM(AttackMetric):

    NAME = "Fluency (ppl)"
    TAGS = { TAG_English }

    def __init__(self):
        """
        pdf: https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
        repo: https://github.com/openai/gpt-2       
        """

        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
    
    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            ipt = self.tokenizer(adversarial_sample["question"], return_tensors="pt", verbose=False)
            return math.exp( self.lm(**ipt, labels=ipt.input_ids)[0] )
        return None