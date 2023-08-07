import numpy as np
from typing import List, Dict, Union, Optional
import copy
from OpenAttack.attack_assist.substitute.word import WordSubstitute, get_default_substitute
from OpenAttack.utils import get_language, check_language, language_by_name
from OpenAttack.exceptions import WordNotInDictionaryException
from OpenAttack.attack_assist.filter_words import get_default_filter_words
from qa.attackers.question_answering import QuestionAnsweringAttacker, QuestionAnswering, QuestionAnsweringGoal
from qa.text_processor.tokenizer import Tokenizer, get_default_tokenizer
from qa.tags import Tag


class PWWSAttacker(QuestionAnsweringAttacker):

    @property
    def TAGS(self):
        return {self.__lang_tag,
                Tag("get_pred", "victim"),
                Tag("get_prob", "victim"),
                Tag("get_ans_span", "victim"),
                Tag("get_f1", "victim"),
                Tag("get_em", "victim")}

    def __init__(self,
                 tokenizer: Optional[Tokenizer] = None,
                 max_length=512,
                 truncation=True,
                 padding=True,
                 substitute: Optional[WordSubstitute] = None,
                 filter_words: List[str] = None,
                 saliency_metric: str = "prob",
                 lang=None
                 ):

        lst = []
        if tokenizer is not None:
            lst.append(tokenizer)
        if substitute is not None:
            lst.append(substitute)
        if len(lst) > 0:
            self.__lang_tag = get_language(lst)
        else:
            self.__lang_tag = language_by_name(lang)
            if self.__lang_tag is None:
                raise ValueError("Unknown language `%s`" % lang)

        if substitute is None:
            substitute = get_default_substitute(self.__lang_tag)
        self.substitute = substitute

        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding

        check_language([self.tokenizer, self.substitute], self.__lang_tag)


        self.unk_token = self.tokenizer.unk_token
        self.unk_token_id = self.tokenizer.unk_token_id

        if filter_words is None:
            filter_words = get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

        self.saliency_metric = saliency_metric

    def attack(self,
               victim: QuestionAnswering,
               input: Dict,
               goal: QuestionAnsweringGoal
               ):

        tokens = self.tokenizer.tokenize(
            question=input["question"],
            context=input["context"],
            # answers=input["answers"], # generates answer span during tokenization
            answers=None,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            pos_tagging=True
        )
        input.update(tokens)

        S = self.get_saliency(clsf=victim,
                              input=input,
                              goal=goal,
                              metric=self.saliency_metric)
        S_softmax = np.exp(S - S.max())
        S_softmax = S_softmax / S_softmax.sum()

        qs = input["input_stats"]["question_span"][0]
        qe = input["input_stats"]["question_span"][1]
        w_star = [self.get_wstar(clsf=victim,
                                 input=input,
                                 idx=i,
                                 goal=goal,
                                 metric="prob")
                  for i in range(qs, qe + 1)]

        H = [(idx, w_star[idx][0], S_softmax[idx] * w_star[idx][1])
             for idx in range(0, qe - qs + 1)]

        H = sorted(H, key=lambda x: -x[2])

        # Adversarial sentence (tokenized)
        ret_sent = input["input_tokens"][qs:qe + 1].copy()
        for i in range(len(H)):
            idx, wd, _ = H[i]
            if ret_sent[idx] in self.filter_words:
                continue
            ret_sent[idx] = wd

            curr_sent = self.tokenizer.detokenize(ret_sent)

            input_ = {
                "question": curr_sent,  # generated adversarial sentence
                "context": input["context"],
                "answers": input["answers"],
                "tokenized": False
            }

            preds, _ = victim.get_ans([input_])
            pred = preds[0]
            if goal.check(input_, pred):
                return input_
        return None

    def get_saliency(self,
                     clsf: QuestionAnswering,
                     input: Dict,
                     goal: QuestionAnsweringGoal,
                     metric: str = "prob"):
        r"""
        get_saliency

        Args:
            clsf (QuestionAnswering): Victim model
            input (Dict): A single sample of dataset
            goal (QuestionAnsweringGoal): Goal of the attack
            metric (str): Metric for word saliency calculation in QA

        Returns:
            array[float]
        """

        if metric not in ["prob", "f1", "em"]:
            metric = "prob"

        qs, qe = input["input_stats"]["question_span"]
        input_ids = input["input_ids"]
        input_tokens = input["input_tokens"]
        input_poss = input["input_poss"]

        x_hat_raw = []
        for i in range(qs, qe + 1):
            input_ = input.copy()

            left = input_ids[qs:i]
            right = input_ids[i + 1:qe + 1]
            x_i_hat_id = left + [self.unk_token_id] + right  # question ids
            input_["input_ids"] = input_ids[:qs] + \
                x_i_hat_id + input_ids[qe + 1:]

            left = input_tokens[qs:i]
            right = input_tokens[i + 1:qe + 1]
            x_i_hat = left + [self.unk_token] + right  # question tokens
            input_["input_tokens"] = input_tokens[:qs] + \
                x_i_hat + input_tokens[qe + 1:]

            left = input_poss[qs:i]
            right = input_poss[i + 1:qe + 1]
            x_i_hat_pos = left + \
                [self.tokenizer.pos_other] + right  # question pos
            input_["input_poss"] = input_poss[:qs] + \
                x_i_hat_pos + input_poss[qe + 1:]

            question = self.tokenizer.detokenize(x_i_hat)
            input_["question"] = question

            x_hat_raw.append(input_)

        # Add original sample to the end
        input_ = input.copy()
        x = input_["input_tokens"][qs:qe + 1]
        question = self.tokenizer.detokenize(x)
        input_["question"] = question
        x_hat_raw.append(input_)

        # Word Saliency
        # S(x, w_i) = P(y_true|x) - P(y_true|x_i_hat)
        # x = w_1 w_2 ... w_i ... w_d
        # x_i_hat = w_1 w_2 ... UNK ... w_d
        if metric == "prob":
            if "answer_span" in input["input_stats"]:
                ans_start, ans_end = input["input_stats"]["answer_span"]
            else:
                ans_start, ans_end = clsf.get_ans_span([input])[0]

            start_probs, end_probs = clsf.get_prob(x_hat_raw)

            x_start_prob = start_probs[-1, ans_start]  # last item is orig x
            x_end_prob = end_probs[-1, ans_end]  # last item is orig x

            x_hat_i_start_prob = start_probs[:-1, ans_start]
            x_hat_i_end_prob = end_probs[:-1, ans_end]

            S_x_start = x_start_prob - x_hat_i_start_prob
            S_x_end = x_end_prob - x_hat_i_end_prob
            S_x = S_x_start

            return S_x

        if metric == "f1":
            f1_scores = clsf.get_f1(x_hat_raw)

            x_f1_score = f1_scores[-1]
            x_hat_i_f1_score = f1_scores[:-1]

            S_x = x_f1_score - x_hat_i_f1_score

            return S_x

        if metric == "em":
            em = clsf.get_em(x_hat_raw)

            x_em = em[-1]
            x_hat_i_em = em[:-1]

            S_x = x_em - x_hat_i_em

            return S_x

    def get_wstar(self,
                  clsf: QuestionAnswering,
                  input: Dict,
                  idx: int,
                  goal: QuestionAnsweringGoal,
                  metric: str = "prob"):
        r"""
        get_wstar

        Args:
            clsf (QuestionAnswering): Victim model
            input (Dict): A single sample of dataset
            idx (int): Index of the token
            goal (QuestionAnsweringGoal): Goal of the attack

        Returns:
            Tuple[str, float]: A tuple of token and its probability
        """

        qs = input["input_stats"]["question_span"][0]
        qe = input["input_stats"]["question_span"][1]
        pos = input["input_poss"][idx]
        sent = input["input_tokens"]
        word = input["input_tokens"][idx]

        try:
            rep_words = list(map(lambda x: x[0], self.substitute(word, pos)))
        except WordNotInDictionaryException:
            rep_words = []
        rep_words = list(filter(lambda x: x != word, rep_words))
        if len(rep_words) == 0:
            return (word, 0)

        sents = []
        words = []
        for rw in rep_words:
            # In some cases `rw` might be tokenized into subtokens
            new_sent = sent[qs:idx] + [rw] + sent[idx + 1:qe + 1]
            question = self.tokenizer.detokenize(new_sent)

            # Skip the words that are tokenized to more than one tokens
            # due to sub-words creation by the transformers tokenizer
            # Do not consider starting [CLS] and ending [SEP]
            tkns = self.tokenizer.tokenize(
                question=question,
                truncation=self.truncation)["input_tokens"][1:-1]

            # `rw` (word with subtokens) is ignored
            if len(tkns) != (qe - qs + 1):
                continue

            input_ = input.copy()
            input_["question"] = question
            input_["tokenized"] = False
            sents.append(input_)
            words.append(rw)

        if len(sents) == 0:
            return (word, 0)

        question = self.tokenizer.detokenize(sent[qs:qe + 1])
        input_ = input.copy()
        input_["question"] = question
        input_["tokenized"] = False
        sents.append(input_)

        # Word Substitution Strategy
        # w_i_star = R(w_i, L_i) = argmax<w'_i in L_i> {P(y_true|x) - P(y_true|x'_i)}
        # x = w_1 w_2 ... w_i ... w_d
        # x'_i = w_1 w_2 ... w'_i ... w_d
        if metric == "prob":
            if "answer_span" in input["input_stats"]:
                ans_start, ans_end = input["input_stats"]["answer_span"]
            else:
                ans_start, ans_end = clsf.get_ans_span([input])[0]

            start_probs, end_probs = clsf.get_prob(sents)

            x_start_prob = start_probs[-1, ans_start]  # prob_orig
            x_end_prob = end_probs[-1, ans_end]  # prob_orig
            x_prob = x_start_prob

            res_start_prob = start_probs[:-1, ans_start]
            res_end_prob = end_probs[:-1, ans_end]
            res_prob = res_start_prob

            return (words[res_prob.argmin()], x_prob - res_prob.min())
