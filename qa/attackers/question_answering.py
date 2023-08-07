from typing import Any

from qa.victim.question_answering.base import QuestionAnswering
from OpenAttack.attackers.base import Attacker
from qa.attack_assist.goal import QuestionAnsweringGoal
from qa.tags import *


class QuestionAnsweringAttacker(Attacker):
    """
    The base class of all question answering attackers.
    """

    def __call__(self, victim: QuestionAnswering, input_: Any):
        if not isinstance(victim, QuestionAnswering):
            raise TypeError("`victim` is an instance of `%s`, but `%s` expected" % (
                victim.__class__.__name__, "QuestionAnswering"))

        if Tag("get_pred", "victim") not in victim.TAGS:
            raise AttributeError("`%s` needs victim to support `%s` method" % (
                self.__class__.__name__, "get_pred"))
                
        self._victim_check(victim)

        if TAG_QuestionAnswering not in victim.TAGS:
            raise AttributeError(
                "Victim model `%s` must be a question answering" % victim.__class__.__name__)

        if "target" in input_:
            raise NotImplementedError("Targeted attacks are not implemented for question answering.")
        else:
            pred, _ = victim.get_ans([input_])
            goal = QuestionAnsweringGoal(target=pred[0], targeted=False) # store original input prediction inside goal

        adversarial_sample = self.attack(victim=victim, input=input_, goal=goal)

        if adversarial_sample is not None:
            y_adv, _ = victim.get_ans([adversarial_sample])
            
            if not goal.check(adversarial_sample, y_adv[0]):
                raise RuntimeError("Check attacker result failed: result ([%d] %s) expect (%s%d)" % (
                    y_adv, adversarial_sample, "" if goal.targeted else "not ", goal.target))
        
        
        return adversarial_sample
