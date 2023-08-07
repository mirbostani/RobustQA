from qa.victim.question_answering.transformers import TransformersQuestionAnswering
from qa.attackers.textfooler import TextFoolerAttacker
from qa.metric import QAScore, EditDistance
from qa.attack_eval import AttackEval
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
victim = TransformersQuestionAnswering(
  model=model, 
  tokenizer=tokenizer,
  embedding_layer=model.bert.embeddings.word_embeddings,
  device="cuda",
  max_length=512,
  truncation=True,
  padding=True,
  batch_size=8,
  lang="english"
)
dataset = load_dataset("squad", split="validation[0:100]")
attacker = TextFoolerAttacker(tokenizer=victim.tokenizer, max_length=512)
metrics = [
  QAScore(victim, "f1", "orig"),
  QAScore(victim, "f1", "adv"),
  QAScore(victim, "em", "orig"),
  QAScore(victim, "em", "adv"),
  EditDistance()
]
evaluator = AttackEval(attacker, victim, metrics=metrics)
evaluator.eval(dataset, visualize=True, progress_bar=True)
