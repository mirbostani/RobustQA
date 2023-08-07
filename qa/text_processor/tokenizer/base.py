from typing import List, Dict, Tuple, Union

class Tokenizer:
    """
    Tokenizer is the base class of all tokenizers.
    """

    def tokenize(self,
                 question: Union[str, List[str], Tuple[str]],
                 context: Union[str, List[str], Tuple[str], None] = None,
                 answers: Union[Dict, List[Dict], Tuple[Dict], None] = None,
                 max_length: int = 512,
                 truncation: bool = True,
                 padding: bool = True,
                 pos_tagging: bool = True):

        if (not isinstance(question, str) and
            not isinstance(question, list) and
                not isinstance(question, tuple)):
            raise TypeError(
                f"`question` must be a string or list/tuple of strings")
        if (not isinstance(context, str) and
            not isinstance(context, list) and
                not isinstance(context, tuple) and
                context is not None):
            raise TypeError(
                f"`context` must be a string or list/tuple of strings or None")

        if (not isinstance(answers, dict) and
            not isinstance(answers, List) and
                not isinstance(answers, Tuple) and
                answers is not None):
            raise TypeError(
                f"`answers` must be a dict or list/tuple of dicts or None")

        if context is not None and type(question) != type(context):
            raise TypeError(f"`question` and `context` must be \
                of the same type")

        if isinstance(question, (list, tuple)):
            for q in question:
                if not isinstance(q, str):
                    raise TypeError(
                        f"`question` must be a list/tuple of strings")
        if isinstance(context, (list, tuple)):
            for c in context:
                if not isinstance(c, str):
                    raise TypeError(
                        f"`context` must be a list/tuple of strings")

        if isinstance(answers, (list, tuple)):
            for a in answers:
                if not isinstance(a, dict):
                    raise TypeError(f"`answers` must be a list/tuple of dicts")
                if "text" not in a or not isinstance(a["text"], list):
                    raise TypeError(
                        f"`answers[x].text` must be a list of strings")

        return self.do_tokenize(question,
                                context,
                                answers,
                                max_length,
                                truncation,
                                padding,
                                pos_tagging)

    def detokenize(self, tokens: List[str]) -> str:
        if not isinstance(tokens, list):
            raise TypeError(f"`tokens` must be a list of tokens")
        if len(tokens) == 0:
            return ""
        return self.do_detokenize(tokens)

    def do_tokenize(self,
                    question: Union[str, List[str], Tuple[str]],
                    context: Union[str, List[str], Tuple[str], None],
                    answers: Union[Dict, List[Dict], Tuple[Dict], None],
                    max_length: int,
                    truncation: bool,
                    padding: bool,
                    pos_tagging: bool):
        raise NotImplementedError()

    def do_detokenize(self, tokens: List[str]) -> str:
        raise NotImplementedError()
