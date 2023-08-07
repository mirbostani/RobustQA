import re
import string
import enum
from collections import Counter
from OpenAttack.attack_assist.goal.base import AttackGoal


class QuestionAnsweringGoalMetric(enum.Enum):
    EM = 1
    F1 = 2

class QuestionAnsweringGoal(AttackGoal):
    def __init__(self,
                 target,
                 targeted: bool,
                 metric: QuestionAnsweringGoalMetric = QuestionAnsweringGoalMetric.EM):

        r"""
        Create a goal for the attack.

        Args:
            target (Any): Victim's prediction for an original example or for a generated adversarial sample
            targeted (bool): For targeted attack goals, pass victim's prediction for an adversarial sample; 
                                otherwise pass victim's prediction for an original example
            metric (QuestionAnsweringGoalMetric): Not used yet
        """

        self.target = target
        self.targeted = targeted
        self.metric = metric

    @property
    def is_targeted(self):
        return self.targeted

    def check(self, input_adversarial_sample: dict, prediction):
        r"""
        Check if the goal is achieved.

        Args:
            adversarial_sample (str): Generated adversarial samples by the attack
            prediction (str): Victim's prediction for an adversarial sample

        Returns:
            bool
        """

        if self.targeted:
            raise NotImplementedError(
                "Targeted attacks are not implemented for question answering.")
        else:

            if len(self.target) == 0:
                return False
            if len(prediction) == 0:
                return False

            ground_truth = input_adversarial_sample["answers"]["text"][0]
            
            # EM
            # Exact match can be used
            # only if `prediction` returns em=false and `target` returns em=true (considering the answer which is the ground_truth)
            # return self.exact_match_score(prediction, ground_truth) != self.exact_match_score(self.target, ground_truth)
            
            # F1
            threshold = 0.9
            prediction_f1 = self.f1_score(prediction, ground_truth)
            target_f1 = self.f1_score(self.target, ground_truth)
            if target_f1 == 0: target_f1 = 1.0
            criteria = (target_f1 - prediction_f1) / target_f1
            return criteria > threshold

    def exact_match_score(self, prediction, ground_truth):
        return (self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def normalize_answer(self, raw):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(raw))))
