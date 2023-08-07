from typing import Callable, Dict, List, Tuple
import numpy as np
from OpenAttack.victim.classifiers.base import Victim
from qa.victim.question_answering.methods import *
from qa.tags import Tag, TAG_QuestionAnswering

QUESTION_ANSWERING_METHODS : Dict[str, VictimMethod] = {
    "get_f1": GetF1(),
    "get_em": GetEM(),
    "get_ans": GetAnswer(),
    "get_ans_span": GetAnswerSpan(),
    "get_pred": GetPredict(),
    "get_prob": GetProbability(),
    "get_grad": GetGradient(),
    "get_embedding": GetEmbedding()
}

class QuestionAnswering(Victim):
    """
    QuestionAnswering base class.
    """
    get_f1: Callable[[List[Dict]], Tuple[np.ndarray]]
    get_em: Callable[[List[Dict]], Tuple[List[bool]]]
    get_ans: Callable[[List[Dict]], Tuple[List[str], np.ndarray]]
    get_ans_span: Callable[[List[Dict]], Tuple[List[Tuple[int, int]]]]
    get_pred: Callable[[List[Dict]], Tuple[np.ndarray, np.ndarray]]
    get_prob: Callable[[List[Dict]], Tuple[np.ndarray, np.ndarray]]
    get_grad: Callable[[List[Dict]], Tuple[np.ndarray, np.ndarray, np.ndarray]]
    def __init_subclass__(cls):
        invoke_funcs = []
        tags = [ TAG_QuestionAnswering ]

        for func_name in QUESTION_ANSWERING_METHODS.keys():
            if hasattr(cls, func_name):
                invoke_funcs.append((func_name, QUESTION_ANSWERING_METHODS[func_name]))
                tags.append( Tag(func_name, "victim") )
                setattr(cls, func_name, QUESTION_ANSWERING_METHODS[func_name].method_decorator( getattr(cls, func_name) ) )
        
        super().__init_subclass__(invoke_funcs, tags)