from typing import List, Optional
from qa.text_processor.tokenizer import Tokenizer, get_default_tokenizer
from OpenAttack.text_process.constituency_parser import ConstituencyParser, get_default_constituency_parser
from OpenAttack.utils import check_language
from qa.tags import TAG_English, Tag
from qa.attackers.question_answering import QuestionAnsweringAttacker, QuestionAnswering, QuestionAnsweringGoal
from OpenAttack.data_manager import DataManager
import numpy as np
import pickle
import torch
import copy

DEFAULT_TEMPLATES = [
    '( ROOT ( S ( NP ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( S ( VP ) ( . ) ) ) EOP',
    '( ROOT ( NP ( NP ) ( . ) ) ) EOP',
    '( ROOT ( FRAG ( SBAR ) ( . ) ) ) EOP',
    '( ROOT ( S ( S ) ( , ) ( CC ) ( S ) ( . ) ) ) EOP',
    '( ROOT ( S ( LST ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( SBARQ ( WHADVP ) ( SQ ) ( . ) ) ) EOP',
    '( ROOT ( S ( PP ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( S ( ADVP ) ( NP ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP'
]


def reverse_bpe(sent):
    x = []
    cache = ''
    for w in sent:
        if w.endswith('@@'):
            cache += w.replace('@@', '')
        elif cache != '':
            x.append(cache + w)
            cache = ''
        else:
            x.append(w)
    return ' '.join(x)


class SCPNAttacker(QuestionAnsweringAttacker):

    @property
    def TAGS(self):
        return {Tag("get_pred", "victim"), self.__lang_tag}

    def __init__(self,
                 templates: List[str] = DEFAULT_TEMPLATES,
                 device: Optional[torch.device] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 parser: Optional[ConstituencyParser] = None,
                 max_length=512,
                 truncation=True,
                 padding=True,
                 ):
        """
        Adversarial Example Generation with Syntactically Controlled Paraphrase Networks. Mohit Iyyer, John Wieting, Kevin Gimpel, Luke Zettlemoyer. NAACL-HLT 2018.
        pdf: https://www.aclweb.org/anthology/N18-1170.pdf
        repo: https://github.com/miyyer/scpn

        Args:
            templates: A list of templates used in SCPNAttacker. **Default:** ten manually selected templates.
            device: The device to load SCPN models (pytorch). **Default:** Use "cpu" if cuda is not available else "cuda".
            tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`
            parser: A constituency parser.
            max_length: The maximum length of an input sentence for bert. **Default:** 512
            truncation: Enables tokenizer's truncation.
            padding: Enables tokenizer's padding.

        :Language: english

        The default templates are:

        .. code-block:: python

            DEFAULT_TEMPLATES = [
                '( ROOT ( S ( NP ) ( VP ) ( . ) ) ) EOP',
                '( ROOT ( S ( VP ) ( . ) ) ) EOP',
                '( ROOT ( NP ( NP ) ( . ) ) ) EOP',
                '( ROOT ( FRAG ( SBAR ) ( . ) ) ) EOP',
                '( ROOT ( S ( S ) ( , ) ( CC ) ( S ) ( . ) ) ) EOP',
                '( ROOT ( S ( LST ) ( VP ) ( . ) ) ) EOP',
                '( ROOT ( SBARQ ( WHADVP ) ( SQ ) ( . ) ) ) EOP',
                '( ROOT ( S ( PP ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP',
                '( ROOT ( S ( ADVP ) ( NP ) ( VP ) ( . ) ) ) EOP',
                '( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP'
            ]

        """
        from . import models
        from . import subword

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.__lang_tag = TAG_English

        if tokenizer is None:
            self.tokenizer = get_default_tokenizer(self.__lang_tag)
        else:
            self.tokenizer = tokenizer

        if parser is None:
            self.parser = get_default_constituency_parser(self.__lang_tag)
        else:
            self.parser = parser

        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding

        check_language([self.parser, self.tokenizer], self.__lang_tag)

        self.templates = templates

        # Use DataManager Here
        model_path = DataManager.load("AttackAssist.SCPN")
        pp_model = torch.load(model_path["scpn.pt"], map_location=self.device)
        parse_model = torch.load(model_path["parse_generator.pt"], map_location=self.device)
        pp_vocab, rev_pp_vocab = pickle.load(open(model_path["parse_vocab.pkl"], 'rb'))
        bpe_codes = open(model_path["bpe.codes"], "r", encoding="utf-8")
        bpe_vocab = open(model_path["vocab.txt"], "r", encoding="utf-8")
        self.parse_gen_voc = pickle.load(open(model_path["ptb_tagset.pkl"], "rb"))

        self.pp_vocab = pp_vocab
        self.rev_pp_vocab = rev_pp_vocab
        self.rev_label_voc = dict((v, k) for (k, v) in self.parse_gen_voc.items())

        # load paraphrase network
        pp_args = pp_model['config_args']
        self.net = models.SCPN(pp_args.d_word, pp_args.d_hid, pp_args.d_nt, pp_args.d_trans, len(self.pp_vocab),
                               len(self.parse_gen_voc) - 1, pp_args.use_input_parse)
        self.net.load_state_dict(pp_model['state_dict'])
        self.net = self.net.to(self.device).eval()

        # load parse generator network
        parse_args = parse_model['config_args']
        self.parse_net = models.ParseNet(parse_args.d_nt, parse_args.d_hid, len(self.parse_gen_voc))
        self.parse_net.load_state_dict(parse_model['state_dict'])
        self.parse_net = self.parse_net.to(self.device).eval()

        # instantiate BPE segmenter

        bpe_vocab = subword.read_vocabulary(bpe_vocab, 50)
        self.bpe = subword.BPE(bpe_codes, '@@', bpe_vocab, None)

    # in classification input is just a string but in question answering its a record from squad
    def gen_paraphrase(self, question, templates):
        template_lens = [len(x.split()) for x in templates]
        np_templates = np.zeros((len(templates), max(template_lens)), dtype='int32')
        for z, template in enumerate(templates):
            np_templates[z, :template_lens[z]] = [self.parse_gen_voc[w] for w in templates[z].split()]
        tp_templates = torch.from_numpy(np_templates).long().to(self.device)
        tp_template_lens = torch.LongTensor(template_lens).to(self.device)


        seg_sent = self.bpe.segment(question.lower()).split()

        # encode sentence using pp_vocab, leave one word for EOS
        seg_sent = [self.pp_vocab[w] for w in seg_sent if w in self.pp_vocab]

        # add EOS
        seg_sent.append(self.pp_vocab['EOS'])

        torch_sent = torch.LongTensor(seg_sent).to(self.device)

        torch_sent_len = torch.LongTensor([len(seg_sent)]).to(self.device)

        # encode parse using parse vocab
        # Stanford Parser
        parse_tree = self.parser(question)
        parse_tree = " ".join(parse_tree.replace("\n", " ").split()).replace("(", "( ").replace(")", " )")
        parse_tree = parse_tree.split()

        for i in range(len(parse_tree) - 1):
            if (parse_tree[i] not in "()") and (parse_tree[i + 1] not in "()"):
                parse_tree[i + 1] = ""
        parse_tree = " ".join(parse_tree).split() + ["EOP"]

        torch_parse = torch.LongTensor([self.parse_gen_voc[w] for w in parse_tree]).to(self.device)
        torch_parse_len = torch.LongTensor([len(parse_tree)]).to(self.device)

        # generate full parses from templates
        beam_dict = self.parse_net.batch_beam_search(torch_parse.unsqueeze(0), tp_templates, torch_parse_len[:],
                                                     tp_template_lens, self.parse_gen_voc['EOP'], beam_size=3,
                                                     max_steps=150)
        seq_lens = []
        seqs = []
        for b_idx in beam_dict:
            prob, _, _, seq = beam_dict[b_idx][0]
            seq = seq[:-1]  # chop off EOP
            seq_lens.append(len(seq))
            seqs.append(seq)
        np_parses = np.zeros((len(seqs), max(seq_lens)), dtype='int32')
        for z, seq in enumerate(seqs):
            np_parses[z, :seq_lens[z]] = seq
        tp_parses = torch.from_numpy(np_parses).long().to(self.device)
        tp_len = torch.LongTensor(seq_lens).to(self.device)

        # generate paraphrases from parses
        ret = []
        beam_dict = self.net.batch_beam_search(torch_sent.unsqueeze(0), tp_parses, torch_sent_len[:], tp_len,
                                               self.pp_vocab['EOS'], beam_size=3, max_steps=40)
        for b_idx in beam_dict:
            prob, _, _, seq = beam_dict[b_idx][0]
            gen_parse = ' '.join([self.rev_label_voc[z] for z in seqs[b_idx]])
            gen_sent = ' '.join([self.rev_pp_vocab[w] for w in seq[:-1]])
            ret.append(reverse_bpe(gen_sent.split()))
        return ret

    def attack(self,
               victim: QuestionAnswering,
               input: dict,
               goal: QuestionAnsweringGoal
               ):
        r"""
        Generate an adversarial sentence based on the provided goal.

        Args:
            victim (QuestionAnswering): Victim model
            input (dict): one record of squad
            goal (QuestionAnsweringGoal): Goal of the attack

        Returns:
            dict : input record with generated adversarial sentence
        """
        question = input["question"]
        context = input["context"]

        tokens = self.tokenizer.tokenize(
            question=[question],
            context=[context],
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            pos_tagging=False)

        input["input_ids"] = tokens["input_ids"][0]
        input["token_type_ids"] = tokens["token_type_ids"][0]
        input["attention_mask"] = tokens["attention_mask"][0]
        input["input_tokens"] = tokens["input_tokens"][0]
        input["input_stats"] = tokens["input_stats"][0]
        input["tokenized"] = tokens["tokenized"][0]

        qs = input["input_stats"]["question_span"][0]  # question start
        qe = input["input_stats"]["question_span"][1]  # question end

        sent = input["input_tokens"][qs:qe + 1].copy()

        ssent = ' '.join(sent)

        try:
            pps = self.gen_paraphrase(ssent, self.templates)
        except KeyError as e:
            return None
        for pp in pps:
            input_ = self.dict_for_new_question(input, pp)
            pred, _ = victim.get_ans([input_])

            if goal.check(input_, pred[0]):
                return input_
        return None

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
