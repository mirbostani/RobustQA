import argparse
import json
from distutils.util import strtobool
from qa import VERSION


class ArgumentParser():

    def __init__(self):
        self.parser = argparse.ArgumentParser("RobustQA", add_help=True)

        self.parser.add_argument(
            "-m",
            "--victim_model_or_path",
            type=str,
            default="bert-large-uncased-whole-word-masking-finetuned-squad",
            help="A victim model's name or a trained/finetuned model's local path"
        )
        self.parser.add_argument(
            "-t",
            "--victim_tokenizer_or_path",
            type=str,
            default="bert-large-uncased-whole-word-masking-finetuned-squad",
            help="Victim model's tokenizer"
        )
        self.parser.add_argument(
            "--truncation_max_length",
            type=int,
            default=512,
            help="Maximum number of tokens after which truncation is to occur"
        )
        self.parser.add_argument(
            "-r",
            "--attack_recipe",
            type=str,
            default="pwws",
            choices=["pwws", "textfooler", "genetic", "pso", "bertattack", "deepwordbug", "scpn"],
            help="Attack recipe"
        )
        self.parser.add_argument(
            "-b",
            "--batch_size",
            type=int,
            default=32,
            help="Number of batches"
        )
        self.parser.add_argument(
            "-l",
            "--language",
            type=str,
            default="english",
            choices=["english"]
        )
        self.parser.add_argument(
            "-d",
            "--dataset",
            type=str,
            default="squad",
            # choices=["squad"]
            help="Dataset to be used for attack"
        )
        self.parser.add_argument(
            "--dataset_split",
            type=str,
            default=None,
            help="Splitted dataset to be used for attack (e.g., \
                'validation[0:10]'). Default is set to use all datapoints."
        )
        self.parser.add_argument(
            "--uncased",
            type=lambda v: bool(strtobool(str(v))),
            default=True,
            help="Consider samples as uncased"
        )
        self.parser.add_argument(
            "--use_metric_f1_score",
            default=False,
            action="store_true",
            help="Calculate F1 score metric"
        )
        self.parser.add_argument(
            "--use_metric_exact_match",
            default=False,
            action="store_true",
            help="Calculate EM metric"
        )
        self.parser.add_argument(
            "--use_metric_edit_distance",
            default=False,
            action="store_true",
            help="Calculate Edit Distance metric"
        )
        self.parser.add_argument(
            "--use_metric_fluency",
            default=False,
            action="store_true",
            help="Calculate Fluency metric"
        )
        self.parser.add_argument(
            "--use_metric_grammatical_errors",
            default=False,
            action="store_true",
            help="Calculate Grammer metric"
        )
        self.parser.add_argument(
            "--use_metric_modification_rate",
            default=False,
            action="store_true",
            help="Calculate Modification Rate metric"
        )
        self.parser.add_argument(
            "--use_metric_semantic_similarity",
            default=False,
            action="store_true",
            help="Calculate Semantic Similarity metric"
        )
        self.parser.add_argument(
            "--use_metric_jaccard_char_similarity",
            default=False,
            action="store_true",
            help="Calculate Jaccard Character Similarity metric"
        )
        self.parser.add_argument(
            "--use_metric_jaccard_word_similarity",
            default=False,
            action="store_true",
            help="Calculate Jaccard Word Similarity metric"
        )
        self.parser.add_argument(
            "--visualization",
            type=lambda v: bool(strtobool(str(v))),
            default=True,
            help="Visualize the attack"
        )
        self.parser.add_argument(
            "--progress_bar",
            type=lambda v: bool(strtobool(str(v))),
            default=True,
            help="Show progress of the attack with progress bars"
        )
        self.parser.add_argument(
            "--use_cuda",
            default=False,
            action="store_true",
            help="Enable CUDA"
        )
        self.parser.add_argument(
            "-v",
            "--version",
            action="version",
            version="%(prog)s v{}".format(VERSION),
            help="Display current version of the package."
        )

        self.args = self.parser.parse_args()

    def summary(self):
        print(json.dumps(vars(self.args), indent=4))
