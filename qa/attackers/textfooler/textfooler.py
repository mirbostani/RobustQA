import numpy as np
from typing import List, Dict, Optional
from OpenAttack.text_process.tokenizer import get_default_tokenizer
from OpenAttack.metric import UniversalSentenceEncoder
from OpenAttack.attack_assist.substitute.word import WordSubstitute, get_default_substitute
from OpenAttack.utils import get_language, check_language, language_by_name
from OpenAttack.exceptions import WordNotInDictionaryException
from OpenAttack.attack_assist.filter_words import get_default_filter_words
from torch import TupleType
from qa.attackers.question_answering import QuestionAnsweringAttacker, QuestionAnswering, QuestionAnsweringGoal
from qa.text_processor.tokenizer import Tokenizer, get_default_tokenizer
from qa.tags import Tag


class TextFoolerAttacker(QuestionAnsweringAttacker):
    @property
    def TAGS(self):
        return {self.__lang_tag,
                Tag("get_pred", "victim"), 
                Tag("get_prob", "victim"),
                Tag("get_ans_span", "victim")
                # Tag("get_f1", "victim"),
                # Tag("get_em", "victim")
                }

    def __init__(self,
            import_score_threshold : float = -1,
            sim_score_threshold : float = 0.5,
            sim_score_window : int = 15,
            tokenizer : Optional[Tokenizer] = None,
            max_length=512,
            truncation=True,
            padding=True,
            substitute : Optional[WordSubstitute] = None,
            filter_words : List[str] = None,
            token_unk = "<UNK>",
            lang = None,
            metric="average",
        ):
        """
        Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment. Di Jin, Zhijing Jin, Joey Tianyi Zhou, Peter Szolovits. AAAI 2020.
        pdf: https://arxiv.org/pdf/1907.11932v4
        repo: https://github.com/jind11/TextFooler
        Args:
            import_score_threshold: Threshold used to choose important word. **Default:** -1.
            sim_score_threshold: Threshold used to choose sentences of high semantic similarity. **Default:** 0.5
            sim_score_window: length used in score module. **Default:** 15
            tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`
            max_length: The maximum length of an input sentence for bert. **Default:** 512
            truncation: Enables tokenizer's truncation.
            padding: Enables tokenizer's padding.v
            substitute: A substitute that will be used during the attack procedure. Must be an instance of :py:class:`.WordSubstitute`
            filter_words: A list of words that will be preserved in the attack procesudre.
            token_unk: The token id or the token name for out-of-vocabulary words in victim model. **Default:** ``"<UNK>"``
            lang: The language used in attacker. If is `None` then `attacker` will intelligently select the language based on other parameters.
            metric: Can be "answer_start", "average", "em_score" and "f1_score"

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
        
        if substitute is None:
            substitute = get_default_substitute(self.__lang_tag)
        self.substitute = substitute

        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer

        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.metric = metric

        self.sim_predictor = UniversalSentenceEncoder()

        check_language([self.tokenizer, self.substitute, self.sim_predictor], self.__lang_tag)

        self.import_score_threshold = import_score_threshold
        self.sim_score_threshold = sim_score_threshold
        self.sim_score_window = sim_score_window

        if filter_words is None:
            filter_words = get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

        self.unk_token = self.tokenizer.unk_token
        self.unk_token_id = self.tokenizer.unk_token_id

    def attack(self,
               victim: QuestionAnswering,
               input: dict,
               goal: QuestionAnsweringGoal
               ):
        """
        Generate an adversarial sentence based on the provided goal.

        Args:
            victim (QuestionAnswering): Victim model
            sentence (str): Sentence to be attacked
            goal (QuestionAnsweringGoal): Goal of the attack

        Returns:
            str: A generated adversarial sentence
        """

        orig_label = tuple(np.squeeze(victim.get_pred([input])))
        orig_prob = tuple(np.squeeze(victim.get_pred_prob([input])))

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

        len_text = qe-qs+1
        if len_text < self.sim_score_window:
            self.sim_score_threshold = 0.1  
        half_sim_score_window = (self.sim_score_window - 1) // 2

        words_perturb = self.get_words_perturb(victim, input)

        synonym_words = [
            self.get_neighbours(word, pos)
            if word not in self.filter_words
            else []
            for idx, word, pos in words_perturb
        ]
      
        synonyms_all = []
        for idx, word, pos in words_perturb:
            # synonyms = array of synonyms for one word
            synonyms = synonym_words.pop(0)
            # if there exists any synonym for the current word:
            if synonyms:   
                synonyms_all.append((idx, synonyms))


        text_prime = input["input_tokens"][qs:qe+1]
        text_cache = text_prime[:]
        
        for idx, synonyms in synonyms_all:

            new_texts, synonyms_pos_ls = self.do_replace(input, idx, synonyms)
            # some words have no acceptable synonyms therefore the new_texts list remains empty
            if(len(new_texts) == 0):
                continue

            new_probs = victim.get_prob(new_texts)
            new_preds = victim.get_pred(new_texts)

            if (idx-qs) >= half_sim_score_window and len_text - (idx-qs) - 1 >= half_sim_score_window:
                text_range_min = (idx-qs) - half_sim_score_window 
                text_range_max = (idx-qs) + half_sim_score_window + 1 
            elif idx < half_sim_score_window and len_text - (idx-qs) - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = self.sim_score_window 
            elif (idx-qs) >= half_sim_score_window and len_text - (idx-qs) - 1 < half_sim_score_window:
                text_range_min = len_text - self.sim_score_window 
                text_range_max = len_text 
            else:
                text_range_min = 0
                text_range_max = len_text 

            texts = []
            for x in new_texts:
                x = x["input_tokens"][qs:qe+1]
                texts.append(self.tokenizer.detokenize(x[text_range_min:text_range_max]))
            
            semantic_sims = np.array([self.sim_predictor.calc_score(self.tokenizer.detokenize(text_cache[text_range_min:text_range_max]), x) for x in texts])
            
            new_preds_tuples =  list(zip(new_preds[0], new_preds[1]))
            new_probs_mask = list(map(lambda pred: pred!=orig_label , new_preds_tuples))
            new_probs_mask *= (semantic_sims >= self.sim_score_threshold)

            pos_mask = np.array(self.pos_filter(input["input_poss"][idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx-qs] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                x_adv = self.tokenizer.detokenize(text_prime) # generated adversarial sentence
                input_ = self.dict_for_new_question(input, x_adv)
                pred, _ = victim.get_ans([input_])
                

                if goal.check(input_, pred[0]):
                    return input_ 
            else:
                new_probs_tuples = list(zip( new_probs[0][:, orig_label[0]], new_probs[1][:, orig_label[1]]))
                new_label_probs = new_probs_tuples + (semantic_sims < self.sim_score_threshold).astype(np.float64).reshape(-1,1) + (1 - pos_mask).astype(np.float64).reshape(-1,1)
                new_label_prob_argmin = np.argmin(list(map(lambda prob: (prob[0] + prob[1])/2, new_label_probs)))
                new_label_prob_min = tuple(np.squeeze(new_label_probs[new_label_prob_argmin]))
                
                if new_label_prob_min < orig_prob:
                    text_prime[idx-qs] = synonyms[new_label_prob_argmin]
            text_cache = text_prime[:]
        return None

    def get_leave_1(self, input):

        """
        Leaves out the ith word of the question in each iteration (replacing the word, its id and pos with unk)
        and keeps the new sentences in leave_1 in the form of a dictionary
        
        Args:
            input (Dict): A single sample of dataset
        
        Returns:
            List: A list of generated dictionaries
        """
        
        leave_1 = []

        qs = input["input_stats"]["question_span"][0]
        qe = input["input_stats"]["question_span"][1]
        for i in range(qs, qe + 1):
            input_ = input.copy()

            left = input["input_ids"][qs:i]
            right = input["input_ids"][i + 1:qe + 1]
            x_i_hat_id = left + [self.unk_token_id] + right
            input_["input_ids"] = input["input_ids"][:qs] + \
                x_i_hat_id + input["input_ids"][qe + 1:]
            
            left = input["input_tokens"][qs:i]
            right = input["input_tokens"][i + 1:qe + 1]
            x_i_hat = left + [self.unk_token] + right
            input_["input_tokens"] = input["input_tokens"][:qs] + \
                x_i_hat + input["input_tokens"][qe + 1:]

            left = input["input_poss"][qs:i]
            right = input["input_poss"][i + 1:qe + 1]
            x_i_hat_pos = left + [self.tokenizer.pos_other] + right
            input_["input_poss"] = input["input_poss"][:qs] + \
                x_i_hat_pos + input["input_poss"][qe + 1:]

            question = self.tokenizer.detokenize(x_i_hat)
            input_["question"] = question
            input_["tokenized"] = True
            leave_1.append(input_)

        return leave_1
        
    def do_replace(self, input, idx, synonyms):

        """
        Replaces the idxth word of the input question with its synonyms 
        and keeps the new sentences in new_texts in the form of a dictionary
        
        Args:
            input (Dict): A single sample of dataset
            idx (int): index of the word to be replaced
            synonyms (List): A List of synonyms of the word to be replaced
        
        Returns:
            new_texts (List): A list of generated dictionaries
            synonyms_pos_ls (List): A list of synonyms's POS
        """

        new_texts = []
        synonyms_pos_ls = []

        qs = input["input_stats"]["question_span"][0]
        qe = input["input_stats"]["question_span"][1]

        for synonym in synonyms:

            input_ = input.copy()

            tokenized_synonym = self.tokenizer.tokenize([synonym]) 
            tokenized_length = len(tokenized_synonym["input_ids"][0])
            
            # ignore the words that are tokenized to more than one tokens
            # due to sub-words creation by the BERT tokenizer
            # since words after tokenizing turn to [ [CLS], word, [SEP] ], any word
            # tokenized to more than one tokens will have a tokenized_length > 3
            if tokenized_length > 3 :
                continue

            synonym_ids = tokenized_synonym["input_ids"][0][1]
            synonym_poss = tokenized_synonym["input_poss"][0][1]
            synonyms_pos_ls.append(synonym_poss)

            left = input["input_tokens"][qs:idx]
            right = input["input_tokens"][idx + 1:qe + 1]
            x_i_hat = left + [synonym] + right
            input_["input_tokens"] = input["input_tokens"][:qs] + \
                x_i_hat + input["input_tokens"][qe + 1:] 

            left = input["input_ids"][qs:idx]
            right = input["input_ids"][idx + 1:qe + 1]
            x_i_hat_id = left + [synonym_ids] + right
            input_["input_ids"] = input["input_ids"][:qs] + \
                x_i_hat_id + input["input_ids"][qe + 1:]

            left = input["input_poss"][qs:idx]
            right = input["input_poss"][idx + 1:qe + 1]
            x_i_hat_pos = left + [synonym_poss] + right
            input_["input_poss"] = input["input_poss"][:qs] + \
                x_i_hat_pos + input["input_poss"][qe + 1:]
            

            question = self.tokenizer.detokenize(x_i_hat)
            input_["question"] = question
            input["tokenized"] = True
            new_texts.append(input_)

        return new_texts, synonyms_pos_ls
            

    def get_neighbours(self, word, pos):
        try:
            return list(
                filter(
                    lambda x: x != word,
                    map(
                        lambda x: x[0],
                        self.substitute(word, pos),
                    )
                )
            )
        except WordNotInDictionaryException:
            return []

    def max_tuple(ls:List[TupleType]):
        return sorted(ls, key=lambda x: (x[0] + x[1])/2, reverse=True)[0]

    def pos_filter(self, ori_pos, new_pos_list):
        same = [True if ori_pos == new_pos or (set([ori_pos, new_pos]) <= set(['noun', 'verb']))
                else False for new_pos in new_pos_list]
        return same

    def get_words_perturb(self, victim, input: Dict):
        qs = input["input_stats"]["question_span"][0]

        if self.metric == 'f1_score':
            f1_score = self.f1_score(victim, input)
            words_perturb = []
            for idx, score in sorted(enumerate(f1_score), key=lambda x: x[1], reverse=True):
                if score > self.import_score_threshold and input["input_tokens"][qs+idx] not in self.filter_words:
                    words_perturb.append((qs+idx, input["input_tokens"][qs+idx], input["input_poss"][qs+idx]))

        if self.metric == 'em_score':
            em_score = self.em_score(victim, input)
            words_perturb = []
            for idx, score in sorted(enumerate(em_score), key=lambda x: x[1], reverse=True):
                if score > self.import_score_threshold and input["input_tokens"][qs+idx] not in self.filter_words:
                    words_perturb.append((qs+idx, input["input_tokens"][qs+idx], input["input_poss"][qs+idx]))
        
        if self.metric == 'average':
            import_scores = self.importance_score(victim, input)
            words_perturb = []
            for idx, score in sorted(enumerate(import_scores), key=lambda x: (x[1][0] + x[1][1])/2, reverse=True):
                if (score[0]+score[1])/2 > self.import_score_threshold and input["input_tokens"][qs+idx] not in self.filter_words:
                    words_perturb.append((qs+idx, input["input_tokens"][qs+idx], input["input_poss"][qs+idx]))   

        if self.metric == 'answer_start':
            import_scores = self.importance_score(victim, input)
            words_perturb = []
            for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1][0], reverse=True):
                if score[0] > self.import_score_threshold and input["input_tokens"][qs+idx] not in self.filter_words:
                    words_perturb.append((qs+idx, input["input_tokens"][qs+idx], input["input_poss"][qs+idx]))

        return words_perturb

    def importance_score(self, victim, input: Dict):
        
        orig_probs = victim.get_prob([input])

        ans_start, ans_end = victim.get_ans_span([input])[0]
        orig_label = (ans_start, ans_end)

        orig_prob = (orig_probs[0][0][ans_start], orig_probs[1][0][ans_end]) 

        leave_1 = self.get_leave_1(input)
        
        leave_1_probs = victim.get_prob(leave_1)
        leave_1_orig_probs =list(zip( leave_1_probs[0][:, orig_label[0]], leave_1_probs[1][:, orig_label[1]]))

        leave_1_probs_argmax = victim.get_pred(leave_1)
        leave_1_probs_argmax_tuples = list(zip(leave_1_probs_argmax[0], leave_1_probs_argmax[1]))

        leave_1_pred_probs = victim.get_pred_prob(leave_1)
        leave_1_pred_probs_tuples = list(zip(leave_1_pred_probs[0], leave_1_pred_probs[1]))

        orig_probs_argmax = list(zip(orig_probs[0][0][leave_1_probs_argmax[0]], orig_probs[1][0][leave_1_probs_argmax[1]]))
    
        mask = np.array(list(map(lambda x: np.not_equal(x, orig_label).astype(float) , leave_1_probs_argmax_tuples)))
        orig_prob = np.array(orig_prob)
        orig_probs_argmax = np.array(orig_probs_argmax)
        leave_1_pred_probs_tuples= np.array(leave_1_pred_probs_tuples)

        import_scores = orig_prob - leave_1_orig_probs + mask * (leave_1_pred_probs_tuples - orig_probs_argmax)

        return import_scores

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

    