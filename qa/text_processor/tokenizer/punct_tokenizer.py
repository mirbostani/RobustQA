from OpenAttack.data_manager import DataManager
from .base import Tokenizer
from ...tags import *
import nltk

_POS_MAPPING = {
    "JJ": "adj",
    "VB": "verb",
    "NN": "noun",
    "RB": "adv"
}
class PunctTokenizer(object):
    """
    Tokenizer based on NLTK word_tokenizer.
    """


    TAGS = { TAG_English }

    def __init__(self) -> None:
        self.sent_tokenizer = DataManager.load("TProcess.NLTKSentTokenizer")
        self.word_tokenizer = nltk.WordPunctTokenizer().tokenize
        self.pos_tagger = DataManager.load("TProcess.NLTKPerceptronPosTagger")

    def tokenize(self, x, pos_tagging=True):
        return self.do_tokenize(x, pos_tagging)

    def detokenize(self, x):
        return self.do_detokenize(x)
        
    def do_tokenize(self, x, pos_tagging=True):
        sentences = self.sent_tokenizer(x)
        tokens = []
        for sent in sentences:
            tokens.extend( self.word_tokenizer(sent) )

        if not pos_tagging:
            return tokens
        ret = []
        for word, pos in self.pos_tagger(tokens):
            if pos[:2] in _POS_MAPPING:
                mapped_pos = _POS_MAPPING[pos[:2]]
            else:
                mapped_pos = "other"
            ret.append( (word, mapped_pos) )
        return ret

    def do_detokenize(self, x):
        return " ".join(x)