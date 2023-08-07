import copy
from typing import List, Optional, Union
import numpy as np
from transformers import BertConfig, BertTokenizerFast, BertForMaskedLM
import torch
from qa.attackers.question_answering import QuestionAnsweringAttacker, QuestionAnswering, QuestionAnsweringGoal
from qa.tags import Tag, TAG_English
from OpenAttack.exceptions import WordNotInDictionaryException
from OpenAttack.attack_assist.filter_words import get_default_filter_words
from OpenAttack.attack_assist.substitute.word import get_default_substitute, WordSubstitute
from qa.text_processor.tokenizer import Tokenizer, get_default_tokenizer
import string
from qa.victim.question_answering.transformers import TransformersQuestionAnswering

class Feature(object):
    def __init__(self, input, label):
        self.label = label
        self.input = input
        self.seq = input['question']
        self.final_adverse = input['question']
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []

class BERTAttacker(QuestionAnsweringAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim"), Tag("get_prob", "victim") }

    def __init__(self, 
            mlm_path : str = 'bert-base-uncased',
            k : int = 36,
            use_bpe : bool = True,
            sim_mat : Union[None, bool, WordSubstitute] = None,
            tokenizer : Optional[Tokenizer] = None,
            threshold_pred_score : float = 0.3,
            max_length : int = 512,
            device : Optional[torch.device] = None,
            filter_words : List[str] = None,
            truncation = True,
            padding = True
        ):
        """
        BERT-ATTACK: Adversarial Attack Against BERT Using BERT, Linyang Li, Ruotian Ma, Qipeng Guo, Xiangyang Xue, Xipeng Qiu, EMNLP2020
        pdf: https://arxiv.org/abs/2004.09984
        repo: https://github.com/LinyangLee/BERT-Attack

        Args:
            mlm_path: The path to the masked language model. **Default:** 'bert-base-uncased'
            k: The k most important words / sub-words to substitute for. **Default:** 36
            use_bpe: Whether use bpe. **Default:** `True`
            sim_mat: Whether use cosine_similarity to filter out atonyms. Keep `None` for not using a sim_mat.
            tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`
            threshold_pred_score: Threshold used in substitute module. **Default:** 0.3
            max_length: The maximum length of an input sentence for bert. **Default:** 512
            device: A computing device for bert.
            filter_words: A list of words that will be preserved in the attack procesudre.
            truncation: Enables tokenizer's truncation.
            padding: Enables tokenizer's padding.

        :Classifier Capacity:
            * get_pred
            * get_prob
        """


        self.tokenizer_mlm = BertTokenizerFast.from_pretrained(mlm_path, do_lower_case=True)

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer
        config_atk = BertConfig.from_pretrained(mlm_path)
        self.mlm_model = BertForMaskedLM.from_pretrained(mlm_path, config=config_atk).to(self.device)
        self.k = k
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        

        self.__lang_tag = TAG_English
        if filter_words is None:
            filter_words = get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

        if sim_mat is None or sim_mat is False:
            self.use_sim_mat = False
        else:
            self.use_sim_mat = True
            if sim_mat is True:
                self.substitute = get_default_substitute(self.__lang_tag)
            else:
                self.substitute = sim_mat


    def attack(self,
               victim: QuestionAnswering,
               input: dict,
               goal: QuestionAnsweringGoal
               ):

        input["question"] = input["question"].translate(str.maketrans('', '', string.punctuation)) # removing punctuation from question
        input["question"] = input["question"] + " ?"
        question = input["question"]  
        context = input["context"]

        mlm_tokenizer = self.tokenizer_mlm
        tokenizer = self.tokenizer

        tokens = tokenizer.tokenize(
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

        tokenized_target = self.tokenizer.tokenize(question=[goal.target])['input_tokens'][0]
        tokenized_target = tokenized_target[1:len(tokenized_target)-1]
        
        # MLM-process
        feature = Feature(input, goal.target)
        words, sub_words, keys = self._tokenize(feature.seq, mlm_tokenizer)

        max_length = self.max_length
        inputs = mlm_tokenizer.encode_plus(feature.seq, None, add_special_tokens=True, max_length=max_length, truncation=self.truncation)
        input_ids, _ = torch.tensor(inputs["input_ids"]), torch.tensor(inputs["token_type_ids"])

        if "answer_span" in input["input_stats"]:
            ans_start, ans_end = input["input_stats"]["answer_span"]
        else:
            ans_start, ans_end = victim.get_ans_span([input])[0]

        current_prob = victim.get_prob([feature.input])
        current_prob_S = current_prob[0][0][ans_start]

        sub_words = ['[CLS]'] + sub_words[:2] + sub_words[2:max_length - 2] + ['[SEP]']   

        input_ids_ = torch.tensor([mlm_tokenizer.convert_tokens_to_ids(sub_words)])        
        word_predictions = self.mlm_model(input_ids_.to(self.device))[0].squeeze()  # seq-len(sub) vocab
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, self.k, -1)  # seq-len k

        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

        important_scores = self.get_important_scores(input, words, victim)
        feature.query += int(len(words))
        list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1][0], reverse=True)
        final_words = copy.deepcopy(words)

        for top_index in list_of_index:

            if feature.change > int(0.2 * (len(words))):
                feature.success = 1
                return None

            tgt_word = words[top_index[0]]
            if tgt_word in self.filter_words:
                continue
            if keys[top_index[0]][0] > max_length - 2:
                continue

            substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
            word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]

            substitutes = self.get_substitues(substitutes, mlm_tokenizer, self.mlm_model, self.use_bpe, word_pred_scores, self.threshold_pred_score)

            if self.use_sim_mat:
                try:
                    cfs_output = self.substitute(tgt_word)
                    cos_sim_subtitutes = [elem[0] for elem in cfs_output]
                    substitutes = list(set(substitutes) & set(cos_sim_subtitutes))
                except WordNotInDictionaryException:
                    pass
                    
            most_gap_S = 0.0
            candidate = None
            
            for substitute in substitutes:               
                if substitute == tgt_word:
                    continue  # filter out original word
                if '##' in substitute:
                    continue  # filter out sub-word

                if substitute in self.filter_words:
                    continue
                # if substitute in self.w2i and tgt_word in self.w2i:
                #     if self.cos_mat[self.w2i[substitute]][self.w2i[tgt_word]] < 0.4:
                #         continue
                
                temp_replace = final_words
                temp_replace[top_index[0]] = substitute
                temp_text = mlm_tokenizer.convert_tokens_to_string(temp_replace)
                inputs = mlm_tokenizer.encode_plus(temp_text, None, add_special_tokens=True, max_length=max_length, truncation=True)
                input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to(self.device)
                seq_len = input_ids.size(1)
    
                # input_ = copy.deepcopy(input)
                input_ = self.dict_for_new_question(input, temp_text)
                # input_["question"] = temp_text
                # input_["tokenized"] = False

                temp_prob = victim.get_prob([input_])
                feature.query += 1
                
                pred, _ = victim.get_ans([input_])
                if goal.check(input_, pred[0]):
                    feature.change += 1
                    final_words[top_index[0]] = substitute
                    feature.changes.append([keys[top_index[0]][0], substitute, tgt_word])
                    feature.final_adverse = temp_text
                    feature.success = 4
                    return input_
                else:
                    label_prob = (temp_prob[0][0][ans_start] , temp_prob[1][0][ans_end]) 

                    gap_S = current_prob_S - label_prob[0]
                    if gap_S > most_gap_S:
                        most_gap_S = gap_S
                        candidate = substitute

            if most_gap_S > 0:
                feature.change += 1
                feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
                current_prob_S = current_prob_S - most_gap_S
                final_words[top_index[0]] = candidate

        feature.final_adverse = (mlm_tokenizer.convert_tokens_to_string(final_words))
        feature.success = 2
        return None


    def _tokenize(self, seq, tokenizer):
        seq = seq.replace('\n', '').lower()
        words = seq.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, words):
        len_text = max(len(words), 2)
        masked_words = []
        for i in range(len_text-1):

            # breaking the masked word into tokens
            tokenized_word = self.tokenizer.tokenize([words[i]]) 
            tokenized_length = len(tokenized_word["input_ids"][0])

            # replacing each token with UNK 
            masked = words[0:i]
            for j in range (tokenized_length-2):
                masked = (masked + ['[UNK]'])
            masked = masked +  words[i + 1:]

            joined = " ".join(masked)
            masked_words.append(joined)

        # list of words
        return masked_words

    def get_leave_1(self, input, words):
        """
        Leaves out the ith word of the question in each iteration (replacing the word, its id and pos with unk)
        and keeps the new sentences in leave_1 in the form of a dictionary
        
        Args:
            input (Dict): A single sample of dataset
        
        Returns:
            List: A list of generated dictionaries
        """
        masked_words = self._get_masked(words)
        leave_1 = []
        for masked in masked_words:
            input_ = input.copy()
            input_["question"] = masked
            input_["tokenized"] = False
            leave_1.append(input_)

        return leave_1

    
    def get_important_scores(self, input, words, tgt_model):

        orig_probs = np.squeeze(tgt_model.get_prob([input]))    
 
        if "answer_span" in input["input_stats"]:
            ans_start, ans_end = input["input_stats"]["answer_span"]
        else:
            ans_start, ans_end = tgt_model.get_ans_span([input])[0]

        orig_label = (ans_start, ans_end)
        orig_prob = [orig_probs[0][ans_start], orig_probs[1][ans_end]]

        leave_1_words = self.get_leave_1(input, words)

        # sending each leave_1 sentence for the model separately (sentences may have different number of tokens)
        leave_1_orig_probs = []
        leave_1_probs_argmax_tuples = []
        leave_1_pred_probs_list = []
        orig_probs_argmax = []
        
        for leave_1 in leave_1_words:
            leave_1_prob = np.squeeze(tgt_model.get_prob([leave_1]))
            leave_1_orig_probs.append([leave_1_prob[0][orig_label[0]], leave_1_prob[1][orig_label[1]]])

            leave_1_probs_argmax_tuple = tuple(np.squeeze(tgt_model.get_pred([leave_1])))
            leave_1_probs_argmax_tuples.append(leave_1_probs_argmax_tuple)

            leave_1_pred_probs_list.append(np.squeeze(tgt_model.get_pred_prob([leave_1])))

            orig_probs_argmax.append(
                [orig_probs[0][leave_1_probs_argmax_tuple[0]], orig_probs[1][leave_1_probs_argmax_tuple[1]]])

        mask = np.array(list(map(lambda x: np.not_equal(x, orig_label).astype(float) , leave_1_probs_argmax_tuples)))
        leave_1_orig_probs = np.array(leave_1_orig_probs)
        leave_1_pred_probs_list = np.array(leave_1_pred_probs_list)
        import_scores = orig_prob - leave_1_orig_probs + mask * (leave_1_pred_probs_list - orig_probs_argmax)

        return import_scores


    def get_bpe_substitues(self, substitutes, tokenizer, mlm_model):
        # substitutes L, k

        substitutes = substitutes[0:12, 0:4] # maximum BPE candidates

        # find all possible candidates 
        all_substitutes = []
        for i in range(substitutes.size(0)):
            if len(all_substitutes) == 0:
                lev_i = substitutes[i]
                all_substitutes = [[int(c)] for c in lev_i]
            else:
                lev_i = []
                for all_sub in all_substitutes:
                    for j in substitutes[i]:
                        lev_i.append(all_sub + [int(j)])
                all_substitutes = lev_i

        # all substitutes  list of list of token-id (all candidates)
        c_loss = torch.nn.CrossEntropyLoss(reduction='none')
        word_list = []
        # all_substitutes = all_substitutes[:24]
        all_substitutes = torch.tensor(all_substitutes) # [ N, L ]
        all_substitutes = all_substitutes[:24].to(self.device)

        N, L = all_substitutes.size()
        word_predictions = mlm_model(all_substitutes)[0] # N L vocab-size

        ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1)) # [ N*L ] 
        ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1)) # N  
        _, word_list = torch.sort(ppl)
        word_list = [all_substitutes[i] for i in word_list]
        final_words = []
        for word in word_list:
            tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
            text = tokenizer.convert_tokens_to_string(tokens)
            final_words.append(text)
        return final_words

    def get_substitues(self, substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
        # substitues L,k
        # from this matrix to recover a word
        words = []
        sub_len, k = substitutes.size()  # sub-len, k

        if sub_len == 0:
            return words
            
        elif sub_len == 1:
            for (i,j) in zip(substitutes[0], substitutes_score[0]):
                if threshold != 0 and j < threshold:
                    break
                words.append(tokenizer._convert_id_to_token(int(i)))
        else:
            if use_bpe == 1:
                words = self.get_bpe_substitues(substitutes, tokenizer, mlm_model)
            else:
                return words
        return words
    
    def get_sim_embed(self, embed_path, sim_path):
        id2word = {}
        word2id = {}

        with open(embed_path, 'r', encoding='utf-8') as ifile:
            for line in ifile:
                word = line.split()[0]
                if word not in id2word:
                    id2word[len(id2word)] = word
                    word2id[word] = len(id2word) - 1

        cos_sim = np.load(sim_path)
        return cos_sim, word2id, id2word

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