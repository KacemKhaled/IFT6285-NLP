#!/bin/bash

#Jaro_Winkler Levenshtein Jaccard Cosine Hamming LCSS Damerau_Levenshtein Needleman_Wunsch Soundex

for DISTANCE in Jaro_Winkler Levenshtein Jaccard Cosine Hamming LCSS Damerau_Levenshtein Needleman_Wunsch Soundex
do
  for ORDER in distance unigram comb_d_u comb_u_d
  do
    OUTFILENAME="out/sortie-100-${DISTANCE}-${ORDER}.txt"
    echo "Spell checking voc-1bwc.txt -n 100 -w devoir3-train.txt -d $DISTANCE -o $ORDER > $OUTFILENAME"
    python corrige.py -v voc-1bwc.txt -n 100 -w devoir3-train.txt -d $DISTANCE -o $ORDER > $OUTFILENAME
    echo "Evaluating -f $OUTFILENAME -r devoir3-train.txt -e no_order"
    python eval.py -f $OUTFILENAME -r devoir3-train.txt -e no_order
    echo "Evaluating -f $OUTFILENAME -r devoir3-train.txt -e by_order"
    python eval.py -f $OUTFILENAME -r devoir3-train.txt -e by_order
  done
done
python plots.py

