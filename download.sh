#!/usr/bin/env bash

# SQuAD v1.1
echo "Downloading SQuAD v1.1 dataset..."
SQUAD_DIR=./scripts/squad_adv_attack
mkdir -p "$SQUAD_DIR"
wget -c "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json" -O "$SQUAD_DIR/train-v1.1.json"
wget -c "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json" -O "$SQUAD_DIR/dev-v1.1.json"