from typing import List, Optional
import numpy as np
import copy
from qa.attackers.question_answering import QuestionAnsweringAttacker, QuestionAnswering, QuestionAnsweringGoal
from qa.text_processor.tokenizer import Tokenizer, get_default_tokenizer
from OpenAttack.attack_assist.substitute.word import get_default_substitute, WordSubstitute
from OpenAttack.utils import get_language, check_language, language_by_name
from qa.tags import Tag
from OpenAttack.exceptions import WordNotInDictionaryException
from OpenAttack.attack_assist.filter_words import get_default_filter_words


class PSOAttacker(QuestionAnsweringAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim"), Tag("get_prob", "victim")}

    def __init__(self, 
            pop_size : int = 20,
            max_iters : int = 20,
            tokenizer : Optional[Tokenizer] = None,
            substitute : Optional[WordSubstitute] = None,
            lang = None,
            max_length=512,
            filter_words : List[str] = None,
            truncation = True,
            padding=True
        ):
        """
        Word-level Textual Adversarial Attacking as Combinatorial Optimization. Yuan Zang, Fanchao Qi, Chenghao Yang, Zhiyuan Liu, Meng Zhang, Qun Liu and Maosong Sun. ACL 2020.
        `[pdf] <https://www.aclweb.org/anthology/2020.acl-main.540.pdf>`__
        `[code] <https://github.com/thunlp/SememePSO-Attack>`__
        Args:
            pop_size: Genetic algorithm popluation size. **Default:** 20
            max_iter: Maximum generations of pso algorithm. **Default:** 20
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
        
        if substitute is None:
            substitute = get_default_substitute(self.__lang_tag)
        self.substitute = substitute

        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer

        self.pop_size = pop_size
        self.max_iters = max_iters

        if filter_words is None:
            filter_words = get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length

        check_language([self.tokenizer, self.substitute], self.__lang_tag)

    def attack(self, victim: QuestionAnswering, input, goal: QuestionAnsweringGoal):
        self.invoke_dict = {}
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
        
        x_len = qe+1-qs
        neighbours_nums = [
            min(self.get_neighbour_num(word, pos),10) if word not in self.filter_words else 0
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
        pop = self.generate_population(tokenized_question, neighbours, w_select_probs, x_len)

        part_elites = pop
        if goal.targeted: 
            raise NotImplementedError(
            "Targeted attacks are not implemented for question answering.")
        else:
            all_elite_score = -1
            part_elites_scores = [-1 for _ in range(self.pop_size)]
        all_elite = pop[0]

        Omega_1 = 0.8
        Omega_2 = 0.2
        C1_origin = 0.8
        C2_origin = 0.2
        V = [np.random.uniform(-3, 3) for _ in range(self.pop_size)]
        V_P = [[V[t] for _ in range(x_len)] for t in range(self.pop_size)]
        for i in range(self.max_iters):

            pop_scores = self.predict_batch(victim, pop, input, num_of_tokens, goal) 
            if goal.targeted:
                raise NotImplementedError(
                    "Targeted attacks are not implemented for question answering.")
            else:
                pop_ranks = np.argsort(pop_scores)
                top_attack = pop_ranks[0]
                if np.min(pop_scores) < all_elite_score:
                    all_elite = pop[top_attack]
                    all_elite_score = np.min(pop_scores)
                for k in range(self.pop_size):
                    if pop_scores[k] < part_elites_scores[k]:
                        part_elites[k] = pop[k]
                        part_elites_scores[k] = pop_scores[k]

                input_ = self.dict_for_new_question(input, self.tokenizer.detokenize(pop[top_attack]))
                pred, _ = victim.get_ans([input_])
                
                if goal.check(input_, pred[0]):
                    return input_
            
            Omega = (Omega_1 - Omega_2) * (self.max_iters - i) / self.max_iters + Omega_2
            C1 = C1_origin - i / self.max_iters * (C1_origin - C2_origin)
            C2 = C2_origin + i / self.max_iters * (C1_origin - C2_origin)
            for id in range(self.pop_size):

                for dim in range(x_len):
                    V_P[id][dim] = Omega * V_P[id][dim] + (1 - Omega) * (
                                self.equal(pop[id][dim], part_elites[id][dim]) + self.equal(pop[id][dim],
                                                                                            all_elite[dim]))
                turn_prob = [self.sigmod(V_P[id][d]) for d in range(x_len)]
                P1 = C1
                P2 = C2

                if np.random.uniform() < P1:
                    pop[id] = self.turn(part_elites[id], pop[id], turn_prob, x_len)
                if np.random.uniform() < P2:
                    pop[id] = self.turn(all_elite, pop[id], turn_prob, x_len)

            pop_scores = self.predict_batch(victim, pop, input, num_of_tokens, goal)
            if goal.targeted:
                raise NotImplementedError(
                    "Targeted attacks are not implemented for question answering.")
            else:
                pop_ranks = np.argsort(pop_scores)
                top_attack = pop_ranks[0]
                if np.min(pop_scores) < all_elite_score:
                    all_elite = pop[top_attack]
                    all_elite_score = np.min(pop_scores)
                for k in range(self.pop_size):
                    if pop_scores[k] < part_elites_scores[k]:
                        part_elites[k] = pop[k]
                        part_elites_scores[k] = pop_scores[k]

                input_ = self.dict_for_new_question(input, self.tokenizer.detokenize(pop[top_attack]))
                pred, _ = victim.get_ans([input_])
                if goal.check(input_, pred[0]):
                    return input_

            new_pop = []
            for x in pop:
                change_ratio = self.count_change_ratio(x, tokenized_question, x_len)
                p_change = 1 - 2 * change_ratio
                if np.random.uniform() < p_change:
                    tem = self.mutate( x, tokenized_question, neighbours, w_select_probs)
                    new_pop.append(tem)
                else:
                    new_pop.append(x)
            pop = new_pop

        return None


    def predict_batch(self, victim, sentences, input, num_of_tokens, goal: QuestionAnsweringGoal):
        return np.array([self.predict(victim, s, input, num_of_tokens, goal) for s in sentences])

    def predict(self, victim, sentence, input, num_of_tokens, goal: QuestionAnsweringGoal):
        if tuple(sentence) in self.invoke_dict:
            return self.invoke_dict[tuple(sentence)]

        tokenized_sentence = self.tokenizer.detokenize(sentence)
        input_ = self.dict_for_new_question(input, tokenized_sentence)
        tem = victim.get_prob([input_])    
        diff_with_orig = len(input_["input_tokens"]) - num_of_tokens
        ans_start, ans_end = input["input_stats"]["answer_span"][0] + diff_with_orig, input["input_stats"]["answer_span"][1] + diff_with_orig
        target_start = ans_start
        self.invoke_dict[tuple(sentence)] = tem[0][0][target_start]  # use start probability to rank perturbed question
        return tem[0][0][target_start]

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def generate_population(self, x_orig, neighbours_list, w_select_probs, x_len):
        pop = []
        x_len = w_select_probs.shape[0]
        for i in range(self.pop_size):
            r = np.random.choice(x_len, 1, p=w_select_probs)[0]
            replace_list = neighbours_list[r]
            sub = np.random.choice(replace_list, 1)[0]
            tem = self.do_replace(x_orig, r, sub)
            pop.append(tem)
        return pop

    def turn(self, x1, x2, prob, x_len):
        x_new = copy.deepcopy(x2)
        for i in range(x_len):
            if np.random.uniform() < prob[i]:
                x_new[i] = x1[i]
        return x_new

    def mutate(self, x, x_orig, neigbhours_list, w_select_probs):
        x_len = w_select_probs.shape[0]
        rand_idx = np.random.choice(x_len, 1,p=w_select_probs)[0]
        while x[rand_idx] != x_orig[rand_idx] and self.sum_diff(x_orig,x) < np.sum(np.sign(w_select_probs)):
            rand_idx = np.random.choice(x_len, 1,p=w_select_probs)[0]
        replace_list = neigbhours_list[rand_idx]
        sub_idx= np.random.choice(len(replace_list), 1)[0]
        new_x=copy.deepcopy(x)
        new_x[rand_idx]=replace_list[sub_idx]
        return new_x

    def sum_diff(self, x_orig, x_cur):
        ret = 0
        for wa, wb in zip(x_orig, x_cur):
            if wa != wb:
                ret += 1
        return ret

    def norm(self, n):
        tn = []
        for i in n:
            if i <= 0:
                tn.append(0)
            else:
                tn.append(i)
        s = np.sum(tn)
        if s == 0:
            for i in range(len(tn)):
                tn[i] = 1
            return [t / len(tn) for t in tn]
        new_n = [t / s for t in tn]

        return new_n


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


    def make_batch(self, sents):
        return [self.tokenizer.detokenize(sent) for sent in sents]

    def equal(self, a, b):
        if a == b:
            return -3
        else:
            return 3

    def sigmod(self, n):
        return 1 / (1 + np.exp(-n))

    def count_change_ratio(self, x, x_orig, x_len):
        change_ratio = float(np.sum(np.array(x) != np.array(x_orig))) / float(x_len)
        return change_ratio

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