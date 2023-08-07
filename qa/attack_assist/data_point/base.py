class DataPoint:
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def tokenize(self):
        raise NotImplementedError()