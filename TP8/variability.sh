#!/bin/bash

for confidence in 2 5 7 10 15 20 50 100 500 1000 1500 2000 3000; do
  log_file="confidence-"$confidence".json"
  echo "-------$confidence----------------------------------------------------------------"
  {
    time sacrebleu -t wmt14 -l en-fr -i translations.txt -m bleu --confidence-n $confidence --confidence  -f json  -short  2> logs/"$log_file"
  } 2> logs/time_"$log_file".log
done
