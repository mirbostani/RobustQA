#!/usr/bin/env bash

time python robustqa.py \
    --use_cuda \
    --victim_model_or_path "~/transformers/examples/legacy/question-answering/bert_base_uncased_finetuned_squad_textfooler_adv10" \
    --victim_tokenizer_or_path "~/transformers/examples/legacy/question-answering/bert_base_uncased_finetuned_squad_textfooler_adv10" \
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