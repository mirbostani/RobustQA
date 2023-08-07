from typing import List, Optional
import numpy as np
from qa.attackers.question_answering import QuestionAnsweringAttacker, QuestionAnswering, QuestionAnsweringGoal
from qa.text_processor.tokenizer import Tokenizer, get_default_tokenizer
from OpenAttack.attack_assist.substitute.word import get_default_substitute, WordSubstitute
from OpenAttack.utils import get_language, check_language, language_by_name
from qa.tags import Tag, TAG_English
from OpenAttack.exceptions import WordNotInDictionaryException
from OpenAttack.attack_assist.filter_words import get_default_filter_words
import copy


class GeneticAttacker(QuestionAnsweringAttacker):

    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim"), Tag("get_prob", "victim"), Tag("get_ans_span", "victim") }

    def __init__(self, 
            pop_size : int = 10, 
            max_iters : int = 10, 
            tokenizer : Optional[Tokenizer] = None, 
            substitute : Optional[WordSubstitute] = None, 
            lang = None,
            max_length=512,
            filter_words : List[str] = None,
            truncation = True,
            padding=True
        ):
        """
        Generating Natural Language Adversarial Examples. Moustafa Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, Kai-Wei Chang. EMNLP 2018.
        pdf: https://www.aclweb.org/anthology/D18-1316.pdf
        repo: https://github.com/nesl/nlp_adversarial_examples
        
        Args:
            pop_size: Genetic algorithm popluation size. **Default:** 20
            max_iter: Maximum generations of genetic algorithm. **Default:** 20
            tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`
            substitute: A substitute that will be used during the attack procedure. Must be an instance of :py:class:`.WordSubstitute`
            lang: The language used in attacker. If is `None` then `attacker` will intelligently select the language based on other parameters.
            max_length: The maximum length of an input sentence for bert. **Default:** 512
            filter_words: A list of words that will be preserved in the attack procesudre.
            truncation: Enables tokenizer's truncation.
            padding: Enables tokenizer's padding.

        :Classifier Capacity:
            * get_pred
            * get_prob
            * get_ans_span
        """
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
        
        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer
        
        if substitute is None:
            substitute = get_default_substitute(self.__lang_tag)
        self.substitute = substitute
        
        self.pop_size = pop_size
        self.max_iters = max_iters

        if filter_words is None:
            filter_words = get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length

        check_language([self.tokenizer, self.substitute], self.__lang_tag)


    def attack(self,
               victim: QuestionAnswering,
               input: dict,
               goal: QuestionAnsweringGoal
               ):

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

        tokenized_question = input["input_tokens"][qs:qe+1]

        num_of_tokens = len(input["input_tokens"])
        ans_start, ans_end = victim.get_ans_span([input])[0]

        neighbours_nums = [
            self.get_neighbour_num(word, pos) if word not in self.filter_words else 0
            for word, pos in zip(tokenized_question, input["input_poss"][qs:qe+1])
        ]       
        
        neighbours = [
            self.get_neighbours(word, pos)
            if word not in self.filter_words
            else []
            for word, pos in zip(tokenized_question, input["input_poss"][qs:qe+1])
        ]

        if np.sum(neighbours_nums) == 0:
            return None
        w_select_probs = neighbours_nums / np.sum(neighbours_nums)

        pop = [  # generate population
            self.perturb(
                victim, input, tokenized_question, tokenized_question, neighbours, w_select_probs, goal
            )
            for _ in range(self.pop_size)
        ]

        for i in range(self.max_iters):
            batched_list = self.make_batch(pop) 
            pop_preds_start =[]

            for question in batched_list:      
                input_ = self.dict_for_new_question(input, question)
                
                diff_with_orig = len(input_["input_tokens"]) - num_of_tokens
                ans_start, ans_end = input["input_stats"]["answer_span"][0] + diff_with_orig, input["input_stats"]["answer_span"][1] + diff_with_orig
                target_start = ans_start
                preds = victim.get_prob([input_])
                pop_preds_start.append(preds[0][0][target_start])
               

            if goal.targeted:
                raise NotImplementedError(
                "Targeted attacks are not implemented for question answering.")
            else:
                top_attack = np.argmin(pop_preds_start)
                input_ = self.dict_for_new_question(input, self.tokenizer.detokenize(pop[top_attack]))
                pred, _ = victim.get_ans([input_])
                if goal.check(input_, pred[0]):
                    return input_

            pop_scores = np.asarray(pop_preds_start)
            
            if not goal.targeted:
                pop_scores = 1.0 - pop_scores

            if np.sum(pop_scores) == 0:
                return None
            pop_scores = pop_scores / np.sum(pop_scores)

            elite = [pop[top_attack]]
            parent_indx_1 = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=pop_scores
            )
            parent_indx_2 = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=pop_scores
            )
            childs = [
                self.crossover(pop[p1], pop[p2])
                for p1, p2 in zip(parent_indx_1, parent_indx_2)
            ]

            childs = [
                self.perturb(
                    victim, input, x_cur, tokenized_question, neighbours, w_select_probs, goal
                )
                for x_cur in childs
            ]

            pop = elite + childs
         

        return None

    def get_neighbour_num(self, word, pos):
        try:
            return len(self.substitute(word, pos))
        except WordNotInDictionaryException:
            return 0

    def get_neighbours(self, word, pos):
        try:
            return list(
                map(
                    lambda x: x[0],
                    self.substitute(word, pos),
                )
            )
        except WordNotInDictionaryException:
            return []

    def select_best_replacements(self, victim, input, indx, neighbours, x_cur, x_orig, goal : QuestionAnsweringGoal):

        def do_replace(word):
            ret = copy.deepcopy(x_cur)
            ret[indx] = word
            return ret
        new_list = []
        rep_words = []
        for word in neighbours:
            if word != x_orig[indx]:
                new_list.append(do_replace(word))
                rep_words.append(word)
        if len(new_list) == 0:
            return x_cur
        new_list.append(x_cur)
        
        batched_list = self.make_batch(new_list)
        
        pred_scores = []
        
        for question in batched_list:
            # making a new dict with the new questoins
            input_ = self.dict_for_new_question(input, question)

            diff_with_orig = len(input_["input_tokens"]) - len(input["input_tokens"])
            ans_start, ans_end = input["input_stats"]["answer_span"][0] + diff_with_orig, input["input_stats"]["answer_span"][1] + diff_with_orig
            target_start = ans_start
            pred_scores.append(victim.get_prob([input_])[0][0][target_start])
    
     
        if goal.targeted:
            raise NotImplementedError(
            "Targeted attacks are not implemented for question answering.")
        else:
            new_scores = pred_scores[-1] - pred_scores[:-1]

        if np.max(new_scores) > 0:
            return new_list[np.argmax(new_scores)]
        else:
            return x_cur

    def make_batch(self, sents):
        return [self.tokenizer.detokenize(sent) for sent in sents]

    def perturb(self, victim, input, x_cur, x_orig, neighbours, w_select_probs, goal : QuestionAnsweringGoal):
        x_len = len(x_cur)
        num_mods = 0
        for i in range(x_len):
            if x_cur[i] != x_orig[i]:
                num_mods += 1
        mod_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        if num_mods < np.sum(
            np.sign(w_select_probs)
        ):  # exists at least one index not modified
            while x_cur[mod_idx] != x_orig[mod_idx]:  # already modified
                mod_idx = np.random.choice(x_len, 1, p=w_select_probs)[
                    0
                ]  # random another index
        return self.select_best_replacements(
            victim, input, mod_idx, neighbours[mod_idx], x_cur, x_orig, goal
        )

    def crossover(self, x1, x2):
        ret = []
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                ret.append(x1[i])
            else:
                ret.append(x2[i])
        return ret

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