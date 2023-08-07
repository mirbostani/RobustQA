
from qa.attackers.question_answering import QuestionAnsweringAttacker, QuestionAnswering, QuestionAnsweringGoal
from OpenAttack.text_process.tokenizer import Tokenizer, get_default_tokenizer
from OpenAttack.utils import check_language
from qa.tags import TAG_English, Tag
import numpy as np

import copy

homos = {
    '-': '˗', '9': '৭', '8': 'Ȣ', '7': '𝟕', '6': 'б', '5': 'Ƽ', '4': 'Ꮞ', '3': 'Ʒ', '2': 'ᒿ', '1': 'l', '0': 'O',
    "'": '`', 'a': 'ɑ', 'b': 'Ь', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝚏', 'g': 'ɡ', 'h': 'հ', 'i': 'і', 'j': 'ϳ',
    'k': '𝒌', 'l': 'ⅼ', 'm': 'ｍ', 'n': 'ո', 'o': 'о', 'p': 'р', 'q': 'ԛ', 'r': 'ⲅ', 's': 'ѕ', 't': '𝚝', 'u': 'ս',
    'v': 'ѵ', 'w': 'ԝ', 'x': '×', 'y': 'у', 'z': 'ᴢ'
}


class DeepWordBugAttacker(QuestionAnsweringAttacker):
    @property
    def TAGS(self):
        return {self.__lang_tag, Tag("get_pred", "victim"), Tag("get_prob", "victim")}

    def __init__(self,
                 token_unk="<UNK>",
                 scoring="combined",
                 transform="homoglyph",
                 power=5,
                 tokenizer: Tokenizer = None,
                 max_length=512,
                 truncation=True,
                 padding=True,
                 ):
        """
        Generate an adversarial sentence based on the provided goal.
        Args:
            token_unk: The token id or the token name for out-of-vocabulary words in victim model. **Default:** ``"<UNK>"``
            scoring: Scoring function used to compute word importance, must be one of the following: ``["replaceone", "temporal", "tail", "combined"]``. **Default:** replaceone
            transform: Transform function to modify a word, must be one of the following:  ``["homoglyph", "swap"]``. **Default:** homoglyph
            power: Max words to replace. **Default:** 5
            tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`
            max_length: The maximum length of an input sentence for bert. **Default:** 512
            truncation: Enables tokenizer's truncation.
            padding: Enables tokenizer's padding.

        :Classifier Capacity:
            * get_pred
            * get_prob
        """

        self.token_unk = token_unk
        self.scoring = scoring
        self.transformer = transform
        self.power = power

        if tokenizer is None:
            self.tokenizer = get_default_tokenizer(None)
        else:
            self.tokenizer = tokenizer

        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding

        self.__lang_tag = TAG_English
        check_language([self.tokenizer], self.__lang_tag)

    def attack(self,
               victim: QuestionAnswering,
               input: dict,
               goal: QuestionAnsweringGoal
               ):
        """
        * **input** : one record of squad.
        """
        question = input["question"]
        context = input["context"]

        tokens = self.tokenizer.tokenize(
            question=[question],
            context=[context],
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            pos_tagging=True)
        input["input_ids"] = tokens["input_ids"][0]
        input["token_type_ids"] = tokens["token_type_ids"][0]
        input["attention_mask"] = tokens["attention_mask"][0]
        input["input_tokens"] = tokens["input_tokens"][0]
        input["input_poss"] = tokens["input_poss"][0]
        input["input_stats"] = tokens["input_stats"][0]
        input["tokenized"] = tokens["tokenized"][0]

        qs = input["input_stats"]["question_span"][0]
        qe = input["input_stats"]["question_span"][1]
        tokenized_question = input["input_tokens"][qs:qe + 1]

        input_ = copy.deepcopy(input)

        # determine the important words to change
        losses = self.scorefunc(self.scoring, victim,  goal, input_)
        indices = [k-qs for k, _ in sorted(losses.items(), key=lambda item: item[1][0], reverse=True) ] #Todo check format

        advinputs = tokenized_question[:]
        t = 0
        j = 0

        while j < self.power and t < len(tokens):
            if advinputs[indices[t]] != '' and advinputs[indices[t]] != ' ':
                advinputs[indices[t]] = self.transform(self.transformer, advinputs[indices[t]])

                j += 1
            t += 1

        ret = self.tokenizer.detokenize(advinputs)
        # input_["question"] = ret
        # input_["tokenized"] = False
        input_ = self.dict_for_new_question(input, ret)

        pred, _ = victim.get_ans([input_])
        if goal.check(input_, pred[0]):
            return input_
        return None

    def scorefunc(self, type_, victim, tokens, goal):
        if type_ == "replaceone":
            return self.replaceone(victim, tokens, goal)
        elif type_ == "temporal":
            return self.temporal(victim, tokens, goal)
        elif type_ == "tail":
            return self.temporaltail(victim, tokens, goal)
        elif type_ == "combined":
            return self.combined(victim, tokens, goal)
        else:
            raise ValueError(
                "Unknown score function %s, %s expected" % (type_, ["replaceone", "temporal", "tail", "combined"]))

    def transform(self, type_, word):
        if type_ == "homoglyph":
            return self.homoglyph(word)
        elif type_ == "swap":
            return self.swap(word)
        else:
            raise ValueError("Unknown transform function %s, %s expected" % (type_, ["homoglyph", "swap"]))


    def replaceone(self, victim, goal , input):
        qs = input["input_stats"]["question_span"][0] #question start
        qe = input["input_stats"]["question_span"][1] #question end
        tokenized_question = input["input_tokens"][qs:qe + 1]

        losses = {}

        ans_start, ans_end = victim.get_ans_span([input])[0] 

        for i in range(qs, qe+1):

            input_ = copy.deepcopy(input)
            tempinputs = tokenized_question[:]
            tempinputs[i - qs] = self.token_unk

            x_new = self.tokenizer.detokenize(tempinputs)
            input_["question"] = x_new
            input_["tokenized"] = False

            tempoutput = victim.get_prob([input_])
            qs_prob = tempoutput[0][0]
            qe_prob = tempoutput[1][0]
            
            losses[i] = (qs_prob[ans_start], qe_prob[ans_end])

        return losses

    def temporal(self, victim, goal , input_):
        qs = input_["input_stats"]["question_span"][0]
        qe = input_["input_stats"]["question_span"][1]
        tokenized_question = input_["input_tokens"][qs:qe + 1]
        losses1={}
        dloss={}
        dloss[0]=(0,0)
        losses1[0]=(0,0)
        
        ans_start, ans_end = victim.get_ans_span([input_])[0]
        
        for i in range(qs, qe+1):
            tempinputs = tokenized_question[: i - qs + 1]
            x_new = self.tokenizer.detokenize(tempinputs)
            input_["question"] = x_new
            input_["tokenized"] = False
            tempoutput = victim.get_prob([input_])

            qs_prob = tempoutput[0][0]
            qe_prob = tempoutput[1][0]
            
            losses1[i] = (qs_prob[ans_start], qe_prob[ans_end])

        for i in range(1, len(tokenized_question)):
            dloss[i] = (abs(losses1[i][0] - losses1[i - 1][0]), abs(losses1[i][1] - losses1[i - 1][1]) )
        return dloss

    def temporaltail(self, victim, goal , input_):
        qs = input_["input_stats"]["question_span"][0]
        qe = input_["input_stats"]["question_span"][1]
        tokenized_question = input_["input_tokens"][qs:qe + 1]

        losses1={}
        dloss={}
        dloss[0]=(0,0)
        losses1[0]=(0,0)

        ans_start, ans_end = victim.get_ans_span([input_])[0]
        
        for i in range(qs, qe+1):
            tempinputs = tokenized_question[i - qs:]
            x_new = self.tokenizer.detokenize(tempinputs)
            input_["question"] = x_new
            input_["tokenized"] = False
            tempoutput = victim.get_prob([input_])

            qs_prob = tempoutput[0][0]
            qe_prob = tempoutput[1][0]
            
            losses1[i] = (qs_prob[ans_start], qe_prob[ans_end])

        for i in range(1, len(tokenized_question)):
                dloss[i] =(abs(losses1[i][0] - losses1[i - 1][0]), abs(losses1[i][1] - losses1[i - 1][1]))
        return dloss

    def combined(self, victim,  goal , input_):
        temp = self.temporal(victim, goal, input_)
        temptail = self.temporaltail(victim, goal, input_)
        dloss = {}

        for i in range(1, len(temp)):
                dloss[i] = ((temp[i][0] + temptail[i][0]) / 2, (temp[i][1] + temptail[i][1]) / 2)
        return dloss

    def homoglyph(self, word):
        s = np.random.randint(0, len(word))
        if word[s] in homos:
            rletter = homos[word[s]]
        else:
            rletter = word[s]
        cword = word[:s] + rletter + word[s + 1:]
        return cword

    def swap(self, word):
        if len(word) != 1:
            s = np.random.randint(0, len(word) - 1)
            cword = word[:s] + word[s + 1] + word[s] + word[s + 2:]
        else:
            cword = word
        return cword
    
    def dict_for_new_question(self, input, new_question):
        tokens = self.tokenizer.tokenize(
            question=[new_question],
            context=[input["context"]],
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            pos_tagging=True)

        new_input = input.copy()
        new_input["question"] = new_question
        new_input["input_ids"] = tokens["input_ids"][0]
        new_input["token_type_ids"] = tokens["token_type_ids"][0]
        new_input["attention_mask"] = tokens["attention_mask"][0]
        new_input["input_tokens"] = tokens["input_tokens"][0]
        new_input["input_poss"] = tokens["input_poss"][0]
        new_input["input_stats"] = tokens["input_stats"][0]
        new_input["tokenized"] = tokens["tokenized"][0]

        return new_input