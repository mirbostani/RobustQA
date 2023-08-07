from OpenAttack.utils.auto_lang import language_by_name
from OpenAttack.utils.transformers_hook import HookCloser
from OpenAttack.attack_assist.word_embedding import WordEmbedding
from qa.victim.question_answering.base import QuestionAnswering
from qa.text_processor.tokenizer.transformers_tokenizer import TransformersTokenizer
from typing import List, Dict, Optional
import transformers
import torch
import numpy as np


class TransformersQuestionAnswering(QuestionAnswering):

    def __init__(self,
                 model: transformers.PreTrainedModel,
                 tokenizer: transformers.PreTrainedTokenizer,
                 embedding_layer,
                 device: str = None,
                 max_length: int = 512,
                 truncation: bool = True,
                 padding: bool = True,
                 batch_size: int = 8,
                 lang=None,
                 ) -> None:
        """
        Args:
            model: HuggingFace model for question answering
            tokenizer: HuggingFace tokenizer for question answering.
            embedding_layer: The module of embedding layer used in transformers model, e.g., `BertModel.bert.embeddings.word_embeddings`.
            device: Device of PyTorch model (e.g., "cpu", "cuda").
            max_length: Maximum length of input tokens. Tokens will be truncated if exceed this threshold.
            truncation: Truncate question-context tokens
            padding: Add paddings if tokens list is less than max_length
            batch_size: Maximum batch size.
            lang: Language of the question answering model.
        """

        super().__init__()

        self.model = model

        if lang is not None:
            self.__lang_tag = language_by_name(lang)
        else:
            self.__lang_tag = None

        if device.startswith("cuda") and torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        self.to(device)


        self.word2id = dict()
        for i in range(tokenizer.vocab_size):
            self.word2id[tokenizer.convert_ids_to_tokens(i)] = i
        self.__tokenizer = tokenizer
        self.__transformers_tokenizer = None
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.embedding = embedding_layer.weight.detach().cpu().numpy()
        self.batch_size = batch_size

    @property
    def TAGS(self):
        if self.__lang_tag is None:
            return super().TAGS
        return super().TAGS.union({self.__lang_tag})

    @property
    def tokenizer(self):
        if self.__transformers_tokenizer is None:
            self.__transformers_tokenizer = TransformersTokenizer(
                self.__tokenizer, 
                self.__lang_tag
            )
        return self.__transformers_tokenizer

    def to(self, device: torch.device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def get_f1(self, input_: List[Dict]):
        f1_scores = []
        for k, v in enumerate(input_):
            ground_truth = v["answers"]["text"][0]
            preds, _ = self.get_ans([v])
            pred = preds[0]
            f1_score = self.f1_score(pred, ground_truth)
            f1_scores.append(f1_score * 100)

        return np.array(f1_scores)

    def get_em(self, input_: List[Dict]):
        ems = []
        for k, v in enumerate(input_):
            ground_truth = v["answers"]["text"][0]
            preds, _ = self.get_ans([v])
            pred = preds[0]
            em = self.exact_match(pred, ground_truth)
            ems.append(em * 100)

        return np.array(ems)

    def get_ans(self, input_: List[Dict]):
        r"""
        Get predicted answers
        """
        v = self.predict(input_)
        token_ids = v["token_ids"]

        # When end < start, answer will be "" (empty)
        answers = []
        for i, _ in enumerate(input_):
            curr_tokens = self.tokenizer.ids_to_tokens(token_ids[i])
            answer_start_token_idx = v["start_probs"][i].argmax(axis=0)
            answer_end_token_idx = v["end_probs"][i].argmax(axis=0)
            if answer_start_token_idx >= answer_end_token_idx:
                answer_end_token_idx = v["end_probs"][i].argpartition(
                    -2, axis=0)[-2:].max()
            answer = self.tokenizer.detokenize(
                curr_tokens[answer_start_token_idx:answer_end_token_idx + 1])
            answers.append(answer)

        return answers, None

    def get_ans_span(self, input_: List[Dict]):
        answer_spans = []
        for v in input_:
            if ("tokenized" in v) and v["tokenized"]:
                input_tokens = v["input_tokens"]
                cs, ce = v["input_stats"]["context_span"]
            else:
                tokens = self.tokenizer.tokenize(question=v["question"],
                                                context=v["context"],
                                                max_length=self.max_length,
                                                truncation=self.truncation,
                                                padding=self.padding,
                                                pos_tagging=True)
                input_tokens = tokens["input_tokens"]
                cs, ce = tokens["input_stats"]["context_span"]

            answers_text = v["answers"]["text"]

            answer_span = self.tokenizer.get_answer_span(
                input_tokens=input_tokens,
                context_start=cs,
                context_end=ce,
                answers_text=answers_text
            )
            answer_spans.append(answer_span)

            v["input_stats"]["answer_span"] = answer_span

        return answer_spans
            

    def get_pred_prob(self, input_: List[Dict]):
        r"""
        Get predicted probability start and end tokens index
        """
        v = self.predict(input_)
        return v["start_probs"].max(axis=1), v["end_probs"].max(axis=1)

    def get_pred(self, input_: List[Dict]):
        r"""
        Get predicted start and end tokens index
        """
        v = self.predict(input_)
        return v["start_probs"].argmax(axis=1), v["end_probs"].argmax(axis=1)

    def get_prob(self, input_: List[Dict]):
        r"""
        Get all probabilities of sentence tokens (both start and end sets)
        """
        v = self.predict(input_)
        return v["start_probs"], v["end_probs"]

    def get_grad(self, input_: List[Dict], labels):
        v = self.predict(input_, labels)
        return v["start_probs"], v["end_probs"], v["grads"]

    def predict(self, input_: List[Dict]):
        context_list = []
        question_list = []
        answers_list = []
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        # input_tokens_list = []
        # input_poss_list = []
        # input_stats_list = []
        for v in input_:
            context_list.append(v["context"])
            question_list.append(v["question"])
            answers_list.append(v["answers"])

            # Use tokens if available; otherwise, tokenize again
            # Tokens might have `##` sub-words.
            if ("tokenized" in v) and v["tokenized"]:
                input_ids_list.append(v["input_ids"])
                token_type_ids_list.append(v["token_type_ids"])
                attention_mask_list.append(v["attention_mask"])
            else:
                tokens = self.tokenizer.tokenize(question=[v["question"]],
                                                 context=[v["context"]],
                                                 answers=None,
                                                 max_length=self.max_length,
                                                 truncation=self.truncation,
                                                 padding=self.padding,
                                                 pos_tagging=True)
                input_ids_list.append(tokens["input_ids"][0])
                token_type_ids_list.append(tokens["token_type_ids"][0])
                attention_mask_list.append(tokens["attention_mask"][0])

        # Requires padding
        input_ids = np.array(input_ids_list, dtype=np.int64)
        token_type_ids = np.array(token_type_ids_list, dtype=np.short)
        attention_mask = np.array(attention_mask_list, dtype=np.short)

        result_start = None
        result_end = None
        result_grad = None
        all_hidden_states = None

        requires_grad = False
        output_attentions = False
        output_hidden_states = False

        # Iterate over batches
        for i in range((len(question_list) + self.batch_size - 1) // self.batch_size):
            curr_input_ids = input_ids[i *
                                       self.batch_size: (i + 1) * self.batch_size]
            curr_token_type_ids = token_type_ids[i *
                                                 self.batch_size: (i + 1) * self.batch_size]
            curr_attention_mask = attention_mask[i *
                                                 self.batch_size: (i + 1) * self.batch_size]

            curr_input_ids_ts = torch.from_numpy(
                curr_input_ids).long().to(self.device)
            curr_input_ids_ts.requires_grad = requires_grad

            curr_token_type_ids_ts = torch.from_numpy(
                curr_token_type_ids).long().to(self.device)
            curr_token_type_ids_ts.requires_grad = requires_grad

            curr_attention_mask_ts = torch.from_numpy(
                curr_attention_mask).float().to(self.device)
            curr_attention_mask_ts.requires_grad = requires_grad

            with torch.no_grad():
                self.model.config.output_attentions = output_attentions
                self.model.config.output_hidden_states = output_hidden_states
                outputs = self.model(input_ids=curr_input_ids_ts,
                                     token_type_ids=curr_token_type_ids_ts,
                                     attention_mask=curr_attention_mask_ts)
                # output_hidden_states=output_hidden_states)

            if i == 0:
                start_logits = outputs.start_logits
                start_logits = torch.nn.functional.softmax(
                    start_logits, dim=-1)
                end_logits = outputs.end_logits
                end_logits = torch.nn.functional.softmax(end_logits, dim=-1)

                result_start = start_logits.detach().cpu()
                result_end = end_logits.detach().cpu()
            else:
                start_logits = outputs.start_logits
                start_logits = torch.nn.functional.softmax(
                    start_logits, dim=-1)
                end_logits = outputs.end_logits
                end_logits = torch.nn.functional.softmax(end_logits, dim=-1)

                result_start = torch.cat(
                    (result_start, start_logits.detach().cpu()))
                result_end = torch.cat((result_end, end_logits.detach().cpu()))

        result_start = result_start.numpy()
        result_end = result_end.numpy()

        return {
            "start_probs": result_start,
            "end_probs": result_end,
            "grads": result_grad,
            "hidden_states": all_hidden_states,
            "token_ids": input_ids.tolist(),
        }

    def convert_idx(self, text: str, tokens: List[str]):
        current = 0
        spans = []
        for token in tokens:
            current = text.find(token, current)
            if current < 0:
                raise RuntimeError("Token {} not found.".format(token))
            spans.append((current, current + len(token)))
            current += len(token)
        return spans

    def get_hidden_states(self, input_, labels=None):
        return self.predict(input_, labels)[2]

    def get_embedding(self):
        return WordEmbedding(self.word2id, self.embedding)

    def exact_match(self, prediction, ground_truth):
        return self.tokenizer.exact_match(prediction, ground_truth)

    def f1_score(self, prediction, ground_truth):
        return self.tokenizer.f1_score(prediction, ground_truth)

    def normalize_answer(self, raw):
        return self.tokenizer.normalize(raw)
