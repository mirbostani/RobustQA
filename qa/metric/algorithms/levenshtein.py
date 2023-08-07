from typing import List
from .base import AttackMetric
import torch
from ...text_processor.tokenizer import Tokenizer


class Levenshtein(AttackMetric):

    NAME = "Levenshtein Edit Distance"

    # Pass max_length, truncation, padding
    def __init__(self, tokenizer: Tokenizer, uncased = True) -> None:
        self.tokenizer = tokenizer
        self.uncased = uncased

    @property
    def TAGS(self):
        if hasattr(self.tokenizer, "TAGS"):
            return self.tokenizer.TAGS
        return set()

    def calc_score(self, a: List[str], b: List[str]) -> int:
        """
        Computes the Levenshtein edit distance between two sentences.
        """
        if self.uncased:
            a = [x.lower() for x in a]
            b = [x.lower() for x in b]
        la = len(a)
        lb = len(b)
        f = torch.zeros(la + 1, lb + 1, dtype=torch.long)
        for i in range(la + 1):
            for j in range(lb + 1):
                if i == 0:
                    f[i][j] = j
                elif j == 0:
                    f[i][j] = i
                elif a[i - 1] == b[j - 1]:
                    f[i][j] = f[i - 1][j - 1]
                else:
                    f[i][j] = min(f[i - 1][j - 1], f[i - 1]
                                  [j], f[i][j - 1]) + 1
        return f[la][lb].item()

    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score(
                self.tokenizer.tokenize(input["question"],
                                        pos_tagging=False),
                self.tokenizer.tokenize(adversarial_sample["question"],
                                        pos_tagging=False)
            )
        else:
            return None