from typing import List
from OpenAttack import AttackMetric

class ExactMatch(AttackMetric):

    NAME = "Exact Match"

    def __init__(self, victim, sample_type) -> None:
        self.victim = victim
        self.tokenizer = victim.tokenizer
        self.sample_type = sample_type
        self.NAME += " (" + sample_type.capitalize() + ")"

    @property
    def TAGS(self):
        if hasattr(self.tokenizer, "TAGS"):
            return self.tokenizer.TAGS
        return set()

    def calc_score(self, orig_sample, adv_sample) -> float:
        if self.sample_type == "orig":
            return self.victim.get_em([orig_sample])
        elif self.sample_type == "adv":
            if adv_sample is not None:
                return self.victim.get_em([adv_sample])
            else:
                return self.victim.get_em([orig_sample])

        return None

    def after_attack(self, input, adversarial_sample):
        return self.calc_score(input, adversarial_sample)
