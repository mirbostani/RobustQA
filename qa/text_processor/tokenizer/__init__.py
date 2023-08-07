from .base import Tokenizer
from .punct_tokenizer import PunctTokenizer
from .transformers_tokenizer import TransformersTokenizer

def get_default_tokenizer(lang):
    from qa.tags.tags import TAG_English, TAG_Persian
    if lang == TAG_English:
        return PunctTokenizer() # Used for metrics. Transformers tokenizer adds subwords ##?
        # return TransformersTokenizer()

    if lang == TAG_Persian:
        raise NotImplementedError("Persian tokenizer is not implemeneted.")

    return PunctTokenizer()