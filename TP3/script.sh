#!/bin/bash


python corrige.py -v voc-1bwc.txt -n 100 -w devoir3-train.txt -d Jaro_Winkler -o distance > out/devoir3-sortie-100-JW-u.txt
python corrige.py -v voc-1bwc.txt -n 10 -w devoir3-train.txt -d Jaro_Winkler -o distance > out/devoir3-sortie-10-JW.txt
python corrige.py -v voc-1bwc.txt -n 100 -w devoir3-train.txt -d Levenshtein -o distance > out/devoir3-sortie-100-Lev.txt
python corrige.py -v voc-1bwc.txt -n 10 -w devoir3-train.txt -d Levenshtein -o distance > out/devoir3-sortie-10-Lev.txt
python corrige.py -v voc-1bwc.txt -n 100 -w devoir3-train.txt -d Jaccard -o unigram > out/devoir3-sortie-100-Jc.txt
python corrige.py -v voc-1bwc.txt -n 100 -w devoir3-train.txt -d Cosine -o unigram > out/devoir3-sortie-100-Co.txt
python corrige.py -v voc-1bwc.txt -n 100 -w devoir3-train.txt -d Hamming -o unigram > out/devoir3-sortie-100-Hm.txt
python corrige.py -v voc-1bwc.txt -n 100 -w devoir3-train.txt -d LCSS -o unigram > out/devoir3-sortie-100-LCSS.txt
python corrige.py -v voc-1bwc.txt -n 100 -w devoir3-train.txt -d Needleman_Wunsch -o unigram > out/devoir3-sortie-100-NW.txt
python corrige.py -v voc-1bwc.txt -n 100 -w devoir3-train.txt -d Soundex -o unigram > out/devoir3-sortie-100-Sd.txt



