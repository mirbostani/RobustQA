from typing import List, Dict, Tuple, Union
from OpenAttack.data_manager import DataManager
from qa.tags.tags import *
from .base import Tokenizer
from tqdm import tqdm
from collections import Counter, defaultdict
import numpy as np
import transformers
import re
import string

_POS_MAPPING = {
    "JJ": "adj",
    "VB": "verb",
    "NN": "noun",
    "RB": "adv"
}


class TransformersTokenizer(Tokenizer):
    r"""
    Pretrained Tokenizer from transformers.
    """

    @property
    def TAGS(self):
        return {self.__lang_tag}

    @property
    def unk_token(self):
        return self.__tokenizer.unk_token

    @property
    def unk_token_id(self):
        return self.__tokenizer.unk_token_id

    @property
    def pos_other(self):
        return "other"

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 lang_tag):

        self.__tokenizer = tokenizer
        self.__lang_tag = lang_tag
        self.__pos_tagger = DataManager.load(
            "TProcess.NLTKPerceptronPosTagger")

        import tensorflow_hub as hub
        self.__sim = hub.load(DataManager.load(
            "AttackAssist.UniversalSentenceEncoder"))

    def __tokenize(self,
                   questionList: List[str],
                   contextList: Union[List[str], None],
                   answersList: Union[List[Dict], None],
                   max_length: int,
                   truncation: bool,
                   padding: bool,
                   pos_tagging: bool):
        r"""Tokenize `text` and `text_pair` using Transformers' tokenizer

        Example

            ```python
            question = 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?'
            context = 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'
            tokens = self.__tokenize([question], [context], 512, True, True, True)
            # {
            #  'input_ids': [[101, 2000, 3183, 2106, 1996, 6261, 2984, 9382, 3711, 1999, 8517, 1999, 10223, 26371, 2605, 1029, 102, 6549, 2135, 1010, 1996, 2082, 2038, 1037, 3234, 2839, 1012, 10234, 1996, 2364, 2311, 1005, 1055, 2751, 8514, 2003, 1037, 3585, 6231, 1997, 1996, 6261, 2984, 1012, 3202, 1999, 2392, 1997, 1996, 2364, 2311, 1998, 5307, 2009, 1010, 2003, 1037, 6967, 6231, 1997, 4828, 2007, 2608, 2039, 14995, 6924, 2007, 1996, 5722, 1000, 2310, 3490, 2618, 4748, 2033, 18168, 5267, 1000, 1012, 2279, 2000, 1996, 2364, 2311, 2003, 1996, 13546, 1997, 1996, 6730, 2540, 1012, 3202, 2369, 1996, 13546, 2003, 1996, 24665, 23052, 1010, 1037, 14042, 2173, 1997, 7083, 1998, 9185, 1012, 2009, 2003, 1037, 15059, 1997, 1996, 24665, 23052, 2012, 10223, 26371, 1010, 2605, 2073, 1996, 6261, 2984, 22353, 2135, 2596, 2000, 3002, 16595, 9648, 4674, 2061, 12083, 9711, 2271, 1999, 8517, 1012, 2012, 1996, 2203, 1997, 1996, 2364, 3298, 1006, 1998, 1999, 1037, 3622, 2240, 2008, 8539, 2083, 1017, 11342, 1998, 1996, 2751, 8514, 1007, 1010, 2003, 1037, 3722, 1010, 2715, 2962, 6231, 1997, 2984, 1012, 102]],
            #  'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
            #  'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
            #  'input_tokens': [['[CLS]', 'to', 'whom', 'did', 'the', 'virgin', 'mary', 'allegedly', 'appear', 'in', '1858', 'in', 'lou', '##rdes', 'france', '?', '[SEP]', 'architectural', '##ly', ',', 'the', 'school', 'has', 'a', 'catholic', 'character', '.', 'atop', 'the', 'main', 'building', "'", 's', 'gold', 'dome', 'is', 'a', 'golden', 'statue', 'of', 'the', 'virgin', 'mary', '.', 'immediately', 'in', 'front', 'of', 'the', 'main', 'building', 'and', 'facing', 'it', ',', 'is', 'a', 'copper', 'statue', 'of', 'christ', 'with', 'arms', 'up', '##rai', '##sed', 'with', 'the', 'legend', '"', 've', '##ni', '##te', 'ad', 'me', 'om', '##nes', '"', '.', 'next', 'to', 'the', 'main', 'building', 'is', 'the', 'basilica', 'of', 'the', 'sacred', 'heart', '.', 'immediately', 'behind', 'the', 'basilica', 'is', 'the', 'gr', '##otto', ',', 'a', 'marian', 'place', 'of', 'prayer', 'and', 'reflection', '.', 'it', 'is', 'a', 'replica', 'of', 'the', 'gr', '##otto', 'at', 'lou', '##rdes', ',', 'france', 'where', 'the', 'virgin', 'mary', 'reputed', '##ly', 'appeared', 'to', 'saint', 'bern', '##ade', '##tte', 'so', '##ub', '##iro', '##us', 'in', '1858', '.', 'at', 'the', 'end', 'of', 'the', 'main', 'drive', '(', 'and', 'in', 'a', 'direct', 'line', 'that', 'connects', 'through', '3', 'statues', 'and', 'the', 'gold', 'dome', ')', ',', 'is', 'a', 'simple', ',', 'modern', 'stone', 'statue', 'of', 'mary', '.', '[SEP]']],
            #  'input_poss': [['noun', 'other', 'other', 'verb', 'other', 'noun', 'adj', 'adv', 'verb', 'other', 'other', 'other', 'adj', 'noun', 'noun', 'other', 'adj', 'adj', 'noun', 'other', 'other', 'noun', 'verb', 'other', 'adj', 'noun', 'other', 'verb', 'other', 'adj', 'noun', 'other', 'adj', 'noun', 'noun', 'verb', 'other', 'adj', 'noun', 'other', 'other', 'noun', 'noun', 'other', 'adv', 'other', 'noun', 'other', 'other', 'adj', 'noun', 'other', 'verb', 'other', 'other', 'verb', 'other', 'noun', 'noun', 'other', 'noun', 'other', 'noun', 'other', 'noun', 'verb', 'other', 'other', 'noun', 'noun', 'noun', 'noun', 'noun', 'noun', 'other', 'adj', 'noun', 'adv', 'other', 'adj', 'other', 'other', 'adj', 'noun', 'verb', 'other', 'noun', 'other', 'other', 'adj', 'noun', 'other', 'adv', 'other', 'other', 'noun', 'verb', 'other', 'noun', 'noun', 'other', 'other', 'adj', 'noun', 'other', 'noun', 'other', 'noun', 'other', 'other', 'verb', 'other', 'noun', 'other', 'other', 'noun', 'noun', 'other', 'noun', 'noun', 'other', 'noun', 'other', 'other', 'noun', 'noun', 'verb', 'adv', 'verb', 'other', 'verb', 'adj', 'noun', 'noun', 'adv', 'adj', 'noun', 'noun', 'other', 'other', 'other', 'other', 'other', 'noun', 'other', 'other', 'adj', 'noun', 'other', 'other', 'other', 'other', 'adj', 'noun', 'other', 'verb', 'other', 'other', 'noun', 'other', 'other', 'adj', 'noun', 'other', 'other', 'verb', 'other', 'adj', 'other', 'adj', 'noun', 'noun', 'other', 'adj', 'other', 'noun']],
            #  'input_stats': [{'question_span': (1, 15), 'context_span': (17, 174), 'answer_span': (x, x)}],
            #  'tokenized': [True]
            # }
            ```
        """

        tokens = self.__tokenizer(text=questionList,
                                  text_pair=contextList,
                                  max_length=max_length,
                                  truncation=truncation,
                                  padding=padding)

        input_tokens = []
        input_poss = []
        input_stats = []
        tokenized = []

        for k, v in enumerate(tokens["input_ids"]):
            input_ids = tokens["input_ids"][k]

            curr_tokens = self.__tokenizer.convert_ids_to_tokens(input_ids)
            input_tokens.append(curr_tokens)

            if pos_tagging:
                ret = []
                for word, pos in self.__pos_tagger(curr_tokens):
                    if pos[:2] in _POS_MAPPING:
                        mapped_pos = _POS_MAPPING[pos[:2]]
                    else:
                        mapped_pos = self.pos_other
                    ret.append(mapped_pos)
                input_poss.append(ret)

            # [CLS]question[SEP]context[SEP][PAD]...[PAD]
            stat = {}
            question_start_idx = input_ids.index(
                self.__tokenizer.cls_token_id) + 1
            question_end_idx = input_ids.index(
                self.__tokenizer.sep_token_id) - 1
            stat["question_span"] = (question_start_idx, question_end_idx)
            if contextList:
                context_start_idx = question_end_idx + 2
                context_end_idx = input_ids.index(
                    self.__tokenizer.sep_token_id, context_start_idx) - 1
                stat["context_span"] = (context_start_idx, context_end_idx)
            if answersList:
                answer_span = self.get_answer_span(
                    input_tokens=curr_tokens,
                    context_start=context_start_idx,
                    context_end=context_end_idx,
                    answers_text=answersList[k]["text"])
                stat["answer_span"] = answer_span
            input_stats.append(stat)
            tokenized.append(True)

        tokens["input_tokens"] = input_tokens
        tokens["input_poss"] = input_poss
        tokens["input_stats"] = input_stats
        tokens["tokenized"] = tokenized

        return tokens

    def do_tokenize(self,
                    question: Union[str, List[str], Tuple[str]],
                    context: Union[str, List[str], Tuple[str], None],
                    answers: Union[Dict, List[Dict], Tuple[Dict], None],
                    max_length: int,
                    truncation: bool,
                    padding: bool,
                    pos_tagging: bool):
        questionList = question
        contextList = context
        answersList = answers

        if isinstance(question, str):
            questionList = [question]
        if isinstance(context, str):
            contextList = [context]
        if isinstance(answers, dict):
            answersList = [answers]

        tokens = self.__tokenize(questionList=questionList,
                                 contextList=contextList,
                                 answersList=answersList,
                                 max_length=max_length,
                                 truncation=truncation,
                                 padding=padding,
                                 pos_tagging=pos_tagging)

        if isinstance(question, str):
            return dict([(k, v[0]) for k, v in tokens.items()])

        return tokens

    def do_detokenize(self, tokens: List[str]) -> str:
        return self.__tokenizer.convert_tokens_to_string(tokens)

    def ids_to_tokens(self, token_ids: List[int]):
        return self.__tokenizer.convert_ids_to_tokens(token_ids)

    def normalize(self, raw: str):
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

    def exact_match(self, prediction, ground_truth):
        return (self.normalize(prediction) == self.normalize(ground_truth))

    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize(prediction).split()
        ground_truth_tokens = self.normalize(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def similarity(self, sentA, sentB):
        sentA = sentA.lower()
        sentB = sentB.lower()
        ret = self.__sim([sentA, sentB]).numpy()
        return ret[0].dot(ret[1]) / (np.linalg.norm(ret[0]) * np.linalg.norm(ret[1]))

    def get_answer_span(self,
                        input_tokens: List[str],
                        context_start: int,
                        context_end: int,
                        answers_text: List[str]):
        """
        Calculates the start and end token indexes based on the answer text
        """

        answers = set([self.normalize(answer)
                       for answer in answers_text])
        limit1 = int(np.ceil(max([len(answer.split(" ")) for answer in answers]) * 3.5))
        limit2 = int(np.ceil(max([len(self.__tokenizer(text=[answer])["input_ids"][0]) for answer in answers]))) - 0 # for [CLS] and [SEP]
        answer_limit = max(limit1, limit2)
        # answer_limit = max(limit1, 30)
        context_tokens = input_tokens[context_start:context_end + 1]
     
        results = []
        desc = "answer_span: " + list(answers)[0]
        # lengthRange = tqdm(range(1, min(answer_limit, len(context_tokens) + 1)), desc=desc)
        lengthRange = range(1, min(answer_limit, len(context_tokens) + 1))
        for length in lengthRange:
            for index in range(len(context_tokens)):

                span_text = self.detokenize(context_tokens[index:index+length])
                span_text = self.normalize(span_text)
                score = max([self.similarity(span_text, answer)
                             for answer in answers])
                span = (context_start + index,
                        context_start + index+length-1)
                results.append({
                    "score": score,
                    "span_text": span_text,
                    "span": span,
                    "length": length
                })

                if score == 1.0:
                    return span


        results_sorted = sorted(results, key=lambda x: -x["score"])
        max_span = results_sorted[0]["span"]

        return max_span

