from typing import Dict
from .base import DataPoint
from ...text_processor.tokenizer.base import Tokenizer


class QADataPoint(DataPoint):

    @property
    def is_tokenized(self):
        "Checks whether current data point is tokenized or not"

        if hasattr(self, "tokenized"):
            return self.tokenized
        return False

    def __init__(self, data: Dict, tokenizer: Tokenizer):
        super().__init__(data, tokenizer)

        fields = ["context", "question", "answers"]
        for field in fields:
            if field not in data:
                raise ValueError(f"The `{field}` is not available")
            setattr(self, field, data[field])

        self.tokenize()

    def tokenize(self):
        "Tokenizes current data point"

        tokens = self.tokenizer.tokenize(question=self.question,
                                         context=self.context)
        for key in tokens.keys():
            setattr(self, key, tokens[key])

    def to_dict(self):
        "Returns current data point as a dictionary"

        d = {}
        for k in self.__dict__.keys():
            if k in ("data", "tokenizer"):
                continue
            d[k] = getattr(self, k)
        return d
