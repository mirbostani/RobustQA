from qa.attack_eval import AttackEval
from qa.victim.question_answering.transformers import TransformersQuestionAnswering
from qa.attackers.pwws import PWWSAttacker
from qa.attackers.textfooler import TextFoolerAttacker
from qa.attackers.genetic import GeneticAttacker
from qa.attackers.pso import PSOAttacker
from qa.attackers.bertattack import BERTAttacker
from qa.attackers.deepwordbug import DeepWordBugAttacker
from qa.attackers.scpn import SCPNAttacker
import qa.metric as metric

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset


class AttackSelector():
    def __init__(self, args):
        self.args = args

        # Victim Model
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.victim_tokenizer_or_path
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            self.args.victim_model_or_path
        )
        self.victim = TransformersQuestionAnswering(
            model=model,
            tokenizer=tokenizer,
            embedding_layer=model.bert.embeddings.word_embeddings,
            device="cuda" if self.args.use_cuda else "cpu",
            max_length=self.args.truncation_max_length,
            truncation=True,
            padding=True,
            batch_size=self.args.batch_size,
            lang=self.args.language
        )

        # Dataset
        self.dataset = load_dataset(
            self.args.dataset,
            split=self.args.dataset_split
        ).map(lambda v: {
            "context": v["context"],
            "question": v["question"],
            "answers": v["answers"]
        })

        # Attach Recipe
        if self.args.attack_recipe == "pwws":
            self.attacker = PWWSAttacker(
                tokenizer=self.victim.tokenizer,
                max_length=self.args.truncation_max_length
            )
        elif self.args.attack_recipe == "textfooler":
            self.attacker = TextFoolerAttacker(
                tokenizer=self.victim.tokenizer,
                max_length=self.args.truncation_max_length
            )
        elif self.args.attack_recipe == "genetic":
            self.attacker = GeneticAttacker(
                tokenizer=self.victim.tokenizer,
                max_length=self.args.truncation_max_length
            )
        elif self.args.attack_recipe == "pso":
            self.attacker = PSOAttacker(
                tokenizer=self.victim.tokenizer,
                max_length=self.args.truncation_max_length
            )
        elif self.args.attack_recipe == "bertattack":
            self.attacker = BERTAttacker(
                tokenizer=self.victim.tokenizer,
                max_length=self.args.truncation_max_length
            )
        elif self.args.attack_recipe == "deepwordbug":
            self.attacker = DeepWordBugAttacker(
                tokenizer=self.victim.tokenizer,
                max_length=self.args.truncation_max_length
            )
        elif self.args.attack_recipe == "scpn":
            self.attacker = SCPNAttacker(
                tokenizer=self.victim.tokenizer,
                max_length=self.args.truncation_max_length
            )


        # Attack Evaluator
        metrics = []
        if self.args.use_metric_exact_match:
            metrics.append(metric.QAScore(
                victim=self.victim, metric="em", sample_type="orig"))
            metrics.append(metric.QAScore(
                victim=self.victim, metric="em", sample_type="adv"))
        if self.args.use_metric_f1_score:
            metrics.append(metric.QAScore(
                victim=self.victim, metric="f1", sample_type="orig"))
            metrics.append(metric.QAScore(
                victim=self.victim, metric="f1", sample_type="adv"))
        if self.args.use_metric_edit_distance:
            metrics.append(metric.EditDistance(uncased=self.args.uncased))
        if self.args.use_metric_fluency:
            metrics.append(metric.Fluency())
        if self.args.use_metric_grammatical_errors:
            metrics.append(metric.GrammaticalErrors())
        if self.args.use_metric_modification_rate:
            metrics.append(metric.ModificationRate(uncased=self.args.uncased))
        if self.args.use_metric_semantic_similarity:
            metrics.append(metric.SemanticSimilarity(uncased=self.args.uncased))
        if self.args.use_metric_jaccard_char_similarity:
            metrics.append(metric.JaccardChar(self.args.uncased))
        if self.args.use_metric_jaccard_word_similarity:
            metrics.append(metric.JaccardWordSimilarity(uncased=self.args.uncased))

        self.attack_evaluator = AttackEval(
            attacker=self.attacker,
            victim=self.victim,
            metrics=metrics
        )

    def attack(self):
        self.attack_evaluator.eval(
            self.dataset,
            visualize=self.args.visualization,
            progress_bar=self.args.progress_bar
        )
