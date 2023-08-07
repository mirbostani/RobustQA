from OpenAttack.victim.method import VictimMethod
from OpenAttack.attack_assist.word_embedding import WordEmbedding


class GetF1(VictimMethod):
    def before_call(self, input_):
        if not isinstance(input_, list):
            raise TypeError(
                "get_f1: `input` must be a list of dictionaries, but got %s" % type(input_))
        if len(input_) == 0:
            raise ValueError("empty `input` list")
        for i, it in enumerate(input_):
            if not isinstance(it, dict):
                raise TypeError(
                    "get_f1: `input[%d]` must be a list of dictionaries, but got %s" % (i, type(it)))

    def invoke_count(self, input_):
        return len(input_)


class GetEM(VictimMethod):
    def before_call(self, input_):
        if not isinstance(input_, list):
            raise TypeError(
                "get_em: `input` must be a list of dictionaries, but got %s" % type(input_))
        if len(input_) == 0:
            raise ValueError("empty `input` list")
        for i, it in enumerate(input_):
            if not isinstance(it, dict):
                raise TypeError(
                    "get_em: `input[%d]` must be a list of dictionaries, but got %s" % (i, type(it)))

    def invoke_count(self, input_):
        return len(input_)


class GetAnswer(VictimMethod):
    def before_call(self, input_):
        if not isinstance(input_, list):
            raise TypeError(
                "get_ans: `input` must be a list of dictionaries, but got %s" % type(input_))
        if len(input_) == 0:
            raise ValueError("empty `input` list")
        for i, it in enumerate(input_):
            if not isinstance(it, dict):
                raise TypeError(
                    "get_ans: `input[%d]` must be a list of dictionaries, but got %s" % (i, type(it)))

    def invoke_count(self, input_):
        return len(input_)


class GetAnswerSpan(VictimMethod):
    def before_call(self, input_):
        if not isinstance(input_, list):
            raise TypeError(
                "get_ans_span: `input` must be a list of dictionaries, but got %s" % type(input_))
        if len(input_) == 0:
            raise ValueError("empty `input` list")
        for i, it in enumerate(input_):
            if not isinstance(it, dict):
                raise TypeError(
                    "get_ans_span: `input[%d]` must be a list of dictionaries, but got %s" % (i, type(it)))

    def invoke_count(self, input_):
        return len(input_)


class GetPredict(VictimMethod):
    def before_call(self, input_):
        if not isinstance(input_, list):
            raise TypeError(
                "get_pred: `input` must be a list of dictionaries, but got %s" % type(input_))
        if len(input_) == 0:
            raise ValueError("empty `input` list")
        for i, it in enumerate(input_):
            if not isinstance(it, dict):
                raise TypeError(
                    "get_pred: `input[%d]` must be a list of dictionaries, but got %s" % (i, type(it)))

    def invoke_count(self, input_):
        return len(input_)


class GetProbability(VictimMethod):
    def before_call(self, input_):
        if not isinstance(input_, list):
            raise TypeError(
                "get_prob: `input` must be a list of dictionaries, but got %s" % type(input_))
        if len(input_) == 0:
            raise ValueError("empty `input` list")
        for i, it in enumerate(input_):
            if not isinstance(it, dict):
                raise TypeError(
                    "get_prob: `input[%d]` must be a dictionaries, but got %s" % (i, type(it)))

    def invoke_count(self, input_):
        return len(input_)


class GetGradient(VictimMethod):
    def before_call(self, input_, labels):
        if not isinstance(input_, list):
            raise TypeError(
                "get_prob: `input` must be a list of dictionaries, but got %s" % type(input_))
        if len(input_) == 0:
            raise ValueError("empty `input` list")
        for i, it in enumerate(input_):
            if not isinstance(it, dict):
                raise TypeError(
                    "get_prob: `input[%d]` must be a dictionaries, but got %s" % (i, type(it)))

    def invoke_count(self, input_, labels):
        return len(input_)


class GetEmbedding(VictimMethod):
    def after_call(self, ret):
        if not isinstance(ret, WordEmbedding):
            raise TypeError(
                "`get_embedding`: must return a `WordEmbedding` object")
