# RobustQA

A Framework for Adversarial Text Generation Analysis on Question Answering Systems

## Abstract

Question answering (QA) systems have reached human-level accuracy; however, these systems are not robust enough and are vulnerable to adversarial examples. Recently, adversarial attacks have been widely investigated in text classification. However, there have been few research efforts on this topic in QA. In this article, we have modified the attack algorithms widely used in text classification to fit those algorithms for QA systems. We have evaluated the impact of various attack methods on QA systems at character, word, and sentence levels. Furthermore, we have developed a new framework, named RobustQA, as the first open-source toolkit for investigating textual adversarial attacks in QA systems. RobustQA consists of seven modules: Tokenizer, Victim Model, Goals, Metrics, Attacker, Attack Selector, and Evaluator. It currently supports six different attack algorithms. Furthermore, the framework simplifies the development of new attack algorithms in QA.

## Download PDF

- [https://aclanthology.org/2023.emnlp-demo.24/](https://aclanthology.org/2023.emnlp-demo.24/)
- [https://aclanthology.org/2023.emnlp-demo.24.pdf](https://aclanthology.org/2023.emnlp-demo.24.pdf)

## Cite

You can cite our paper in your work using the following reference formats:

**ACL Anthology**

```
Yasaman Boreshban, Seyed Morteza Mirbostani, Seyedeh Fatemeh Ahmadi, Gita Shojaee, Fatemeh Kamani, Gholamreza Ghassem-Sani, and Seyed Abolghasem Mirroshandel. 2023. RobustQA: A Framework for Adversarial Text Generation Analysis on Question Answering Systems. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 274–285, Singapore. Association for Computational Linguistics.
```

**BibTeX**

```
@inproceedings{boreshban-etal-2023-robustqa,
    title = "{R}obust{QA}: A Framework for Adversarial Text Generation Analysis on Question Answering Systems",
    author = "Boreshban, Yasaman  and
      Mirbostani, Seyed Morteza  and
      Ahmadi, Seyedeh Fatemeh  and
      Shojaee, Gita  and
      Kamani, Fatemeh  and
      Ghassem-Sani, Gholamreza  and
      Mirroshandel, Seyed Abolghasem",
    editor = "Feng, Yansong  and
      Lefever, Els",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-demo.24",
    pages = "274--285",
    abstract = "Question answering (QA) systems have reached human-level accuracy; however, these systems are not robust enough and are vulnerable to adversarial examples. Recently, adversarial attacks have been widely investigated in text classification. However, there have been few research efforts on this topic in QA. In this article, we have modified the attack algorithms widely used in text classification to fit those algorithms for QA systems. We have evaluated the impact of various attack methods on QA systems at character, word, and sentence levels. Furthermore, we have developed a new framework, named RobustQA, as the first open-source toolkit for investigating textual adversarial attacks in QA systems. RobustQA consists of seven modules: Tokenizer, Victim Model, Goals, Metrics, Attacker, Attack Selector, and Evaluator. It currently supports six different attack algorithms. Furthermore, the framework simplifies the development of new attack algorithms in QA. The source code and documentation of RobustQA are available at https://github.com/mirbostani/RobustQA.",
}
```

## Presentations

**RobustQA Presentation (4 min)**

[![RobustQA](https://img.youtube.com/vi/rC9sFV7n8N8/0.jpg)](https://www.youtube.com/watch?v=rC9sFV7n8N8)

**RobustQA Installation & Usage (2 min 20 sec)**

[![RobustQA Demo](https://img.youtube.com/vi/VHPe5DVXdhw/0.jpg)](https://www.youtube.com/watch?v=VHPe5DVXdhw)

## Installation

Clone the following repository to your local system:

```shell
$ git clone https://github.com/mirbostani/RobustQA
$ cd RobustQA
```

Create an environment based on the provided `environment.yml` file to install the dependencies:

```shell
$ conda env create -f environment.yml
$ conda activate robustqa
```

Verify RobustQA:

```shell
$ python robustqa.py --version
RobustQA 1.0.0
```

Download required datasets:

```shell
$ ./download.sh
```

## Usage Examples

### Built-in Attacks & Victim Models via CLI

Run the TextFooler attack on the BERT Large model using the command-line interface.

```shell
$ cd RobustQA
$ python robustqa.py \
    --use_cuda \
    --victim_model_or_path "bert-base-uncased" \
    --victim_tokenizer_or_path "bert-base-uncased" \
    --dataset squad \
    --dataset_split "validation[0:1000]" \
    --truncation_max_length 512 \
    --attack_recipe textfooler \
    --batch_size 8 \
    --language english \
    --use_metric_f1_score \
    --use_metric_exact_match \
    --use_metric_edit_distance \
    --use_metric_fluency \
    --use_metric_grammatical_errors \
    --use_metric_modification_rate \
    --use_metric_semantic_similarity \
    --use_metric_jaccard_char_similarity \
    --use_metric_jaccard_word_similarity \
    --visualization True
```

Attack scenarios in Bash scripts are available in `./examples/*.sh`:

```shell
$ ./examples/textfooler_on_bert_base_uncased.sh
```

### Built-in Attacks & Victim Models via Python

Run the TextFooler attack on the BERT Large model using Python.

```python
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
```

```shell
$ export PYTHONPATH="."
$ python ./examples/textfooler_on_bert_large_finetuned_squad.py

+===========================================+
|                  Summary                  |
+===========================================+
| Total Attacked Instances:       | 100     |
| Successful Instances:           | 16      |
| Attack Success Rate:            | 0.16    |
| Avg. Running Time:              | 0.40168 |
| Total Query Exceeded:           | 0       |
| Avg. Victim Model Queries:      | 76.3    |
| Avg. F1 Score (Orig):           | 74.896  |
| Avg. F1 Score (Adv):            | 65.351  |
| Avg. Exact Match (Orig):        | 61      |
| Avg. Exact Match (Adv):         | 54      |
| Avg. Levenshtein Edit Distance: | 1.375   |
+===========================================+
```

## Help

```shell
$ python robustqa.py --help
usage: RobustQA [-h] [-m VICTIM_MODEL_OR_PATH] [-t VICTIM_TOKENIZER_OR_PATH]
                [--truncation_max_length TRUNCATION_MAX_LENGTH]
                [-r {pwws,textfooler,genetic,pso,bertattack,deepwordbug,scpn}]
                [-b BATCH_SIZE] [-l {english}] [-d DATASET]
                [--dataset_split DATASET_SPLIT] [--uncased UNCASED]
                [--use_metric_f1_score] [--use_metric_exact_match]
                [--use_metric_edit_distance] [--use_metric_fluency]
                [--use_metric_grammatical_errors]
                [--use_metric_modification_rate]
                [--use_metric_semantic_similarity]
                [--use_metric_jaccard_char_similarity]
                [--use_metric_jaccard_word_similarity]
                [--visualization VISUALIZATION] [--progress_bar PROGRESS_BAR]
                [--use_cuda] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -m VICTIM_MODEL_OR_PATH, --victim_model_or_path VICTIM_MODEL_OR_PATH
                        A victim model's name or a trained/finetuned model's
                        local path
  -t VICTIM_TOKENIZER_OR_PATH, --victim_tokenizer_or_path VICTIM_TOKENIZER_OR_PATH
                        Victim model's tokenizer
  --truncation_max_length TRUNCATION_MAX_LENGTH
                        Maximum number of tokens after which truncation is to
                        occur
  -r {pwws,textfooler,genetic,pso,bertattack,deepwordbug,scpn}, --attack_recipe {pwws,textfooler,genetic,pso,bertattack,deepwordbug,scpn}
                        Attack recipe
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of batches
  -l {english}, --language {english}
  -d DATASET, --dataset DATASET
                        Dataset to be used for attack
  --dataset_split DATASET_SPLIT
                        Splitted dataset to be used for attack (e.g.,
                        'validation[0:10]'). Default is set to use all
                        datapoints.
  --uncased UNCASED     Consider samples as uncased
  --use_metric_f1_score
                        Calculate F1 score metric
  --use_metric_exact_match
                        Calculate EM metric
  --use_metric_edit_distance
                        Calculate Edit Distance metric
  --use_metric_fluency  Calculate Fluency metric
  --use_metric_grammatical_errors
                        Calculate Grammer metric
  --use_metric_modification_rate
                        Calculate Modification Rate metric
  --use_metric_semantic_similarity
                        Calculate Semantic Similarity metric
  --use_metric_jaccard_char_similarity
                        Calculate Jaccard Character Similarity metric
  --use_metric_jaccard_word_similarity
                        Calculate Jaccard Word Similarity metric
  --visualization VISUALIZATION
                        Visualize the attack
  --progress_bar PROGRESS_BAR
                        Show progress of the attack with progress bars
  --use_cuda            Enable CUDA
  -v, --version         Display current version of the package.
```
