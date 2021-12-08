#!/bin/bash
conda activate  IFT6285-NLP
for confidence in 2 5 7 10 15 20 50 100 500 1000 1500 2000 3000; do
  log_file="confidence-"$confidence".json"
  echo "-------$confidence----------------------------------------------------------------"
  sacrebleu -t wmt14 -l en-fr -i translations.txt -m bleu --confidence-n $confidence --confidence  -f json  --short
done
# sacrebleu -t wmt14 -l en-fr -i translations.txt -m bleu  --confidence  -f json  --short