#import jaro
import heapq as hq
import editdistance
#import nltk
import soundex
import textdistance
from soundex import Soundex
import pandas as pd
import argparse
import os.path as osp
from os.path import dirname, abspath
import logging

logging.basicConfig(format='%(message)s', level=logging.DEBUG)

def get_frequency_table(file_name):
    frequency_table = {}
    # create a dict from lexique
    with open(file_name ,'r',encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            s = line.rstrip('\n').lstrip()
            frequency = int(s.split(' ')[0])
            word = s.split(' ')[1]
            frequency_table[word] = frequency
    f.close()
    return frequency_table

def affichage(wrong_word, corrections):
    if len(corrections) == 0 : 
        result =  wrong_word + '\t' + wrong_word
    else:
        result =  wrong_word + '\t' + '\t'.join(corrections)
    print(result)
    logging.info(result)

def find_best_candidates(wrong_word, vocab_scores, nb_best_scores, largest, n=5, order_by= 'unigram'):
    scores = list(set(a[0] for a in vocab_scores.values())) 
    if largest:
     # Find three biggest scores (sorted) --> three best possible corrections
        best_scores= hq.nlargest(nb_best_scores, scores)
    else:
        best_scores= hq.nsmallest(nb_best_scores, scores)
            
    best_candidates = {v:vocab_scores[v] for v in vocab_scores if vocab_scores[v][0] in best_scores}
    ordered_list=[]
    if order_by == 'distance':
        ordered_list = sorted(best_candidates, key=lambda k: (best_candidates[k][0]), reverse=largest)
    elif order_by == 'unigram': # negative number in purpose to have descending order by frequency
        ordered_list = sorted(best_candidates, key=lambda k: (-best_candidates[k][1]), reverse=False)
    elif order_by == 'comb_d_u':
        ordered_list = sorted(best_candidates, key=lambda k: (best_candidates[k][0],-best_candidates[k][1]), reverse=largest)
    elif order_by == 'comb_u_d':
        ordered_list = sorted(best_candidates, key=lambda k: (-best_candidates[k][1],best_candidates[k][0]), reverse=largest)
    best_candidates_ordered = {candidate:best_candidates[candidate] for candidate in ordered_list}
    logging.info(f'TOP 10 best candidates : {list(best_candidates_ordered.items())[:10]}')
    corrections = [c for c in best_candidates_ordered][:n]

    affichage(wrong_word, corrections)
    
    return corrections, best_candidates

def process(dispatcher,distance_name,wrong_word,frequency_table,vocab,ordre):
    
    if distance_name=="Soundex":
        instance = Soundex()
        # We want Englidh corrections --> remove non english corrections (-1)
        vocab = [ v for v in vocab if instance.compare(v, wrong_word) >= 0 ]
    distance_scores = { v: [dispatcher[distance_name][0](v, wrong_word), frequency_table[v]] for v in vocab }
    logging.info(f'Corrections using {distance_name} distance :')
    find_best_candidates(wrong_word, distance_scores, 1, dispatcher[distance_name][1],10, order_by= ordre) # 'distance' , 'unigram' or 'comb_d_u' or 'comb_u_d'

def corrige(wrong_words, n, lexique_file, distances=['Jaro_Winkler','Levenshtein'], order_by='unigram'):

    frequency_table = get_frequency_table(lexique_file)
    vocab = list(frequency_table.keys())
    instance = Soundex()

    # https://pypi.org/project/textdistance/
    dispatcher = {
        'Jaro_Winkler' : [textdistance.jaro_winkler.distance,False],
        'Levenshtein' : [textdistance.levenshtein.distance,False],
        'Jaccard': [textdistance.jaccard.distance,False],
        'Cosine': [textdistance.cosine.distance,False],
        'Hamming': [textdistance.hamming.distance,False],
        'LCSS': [textdistance.lcsstr.distance,False],
        'Damerau_Levenshtein': [textdistance.damerau_levenshtein.distance,False],
        'Needleman_Wunsch': [textdistance.needleman_wunsch.distance,False],
        'Soundex': [instance.compare,False]
                 }
    

    # lit une liste de mots a corriger, un par ligne
    with open(wrong_words, 'r',encoding="utf8") as f:
        i=0
        a = len(list(f))
        logging.info(f'Correcting {min(n,a)} words')
        f.seek(0)
        for line in f:
            if i<n:
                i+=1
                wrong_word = line.rstrip().split()[0]
                logging.info(f'{i}: ****************** {wrong_word}')
                # calculate the distance for all the vocabulary
                # for each word in vocabulary, create a dictionary of candidates with 'nb_best_scores': {cand1:score1,cand2:score2,...}
                # get the 'n' best candidates and order them according to a criteria: 'distance' , 'unigram' or 'comb_d_u' or 'comb_u_d'

                for d in distances:
                    process(dispatcher,d,wrong_word,frequency_table,vocab,order_by)
            else:
                break

    f.close()

vocab = 'voc-1bwc.txt'
wrong_words = 'devoir3-train-short.txt'
reference = 'devoir3-train.txt'

CONFIG_PATH = abspath(__file__)
SRC_ROOT = dirname(CONFIG_PATH)
VOCAB_ROOT = osp.join(SRC_ROOT, vocab)
TRAIN_SHORT_ROOT = osp.join(SRC_ROOT, wrong_words)
TRAIN_ROOT = osp.join(SRC_ROOT, reference)

distances = ['Jaro_Winkler', 'Levenshtein', 'Jaccard', 'Cosine', 'Hamming', 'LCSS', 'Damerau_Levenshtein', 'Needleman_Wunsch', 'Soundex']
order_by = ['distance','unigram','comb_d_u','comb_u_d']

parser = argparse.ArgumentParser(description='Words correction')
parser.add_argument('-v','--vocab', default=VOCAB_ROOT,
                    help='Path to vocab file: (default: ./voc-1bwc.txt)')

parser.add_argument('-n','--nb_of_lines', default=10,type=int,
                    help='Nb of lines to read from file: (default: 10)')

parser.add_argument('-w','--wrong_words', default=TRAIN_SHORT_ROOT,
                    help='Path to wrong words file: (default: ./devoir3-train.txt)')

parser.add_argument('-d', '--distance', nargs='+', default=['Jaro_Winkler'],
                    choices=distances,
                    help='Distances: ' +
                        ' | '.join(distances) +
                        ' (default: Jaro_Winkler)')

parser.add_argument('-o', '--order', default='unigram',
                    choices=order_by,
                    help='Order by: ' +
                        ' | '.join(order_by) +
                        ' (default: unigram)')


def main():
    args = parser.parse_args()
    vocab = osp.join(SRC_ROOT, args.vocab)
    wrong_words = osp.join(SRC_ROOT, args.wrong_words)
    corrige(wrong_words, args.nb_of_lines, vocab, args.distance, args.order)
    # corrige(wrong_words, 10, vocab, ['Soundex'], 'unigram')

if __name__ == '__main__':
    main()

# c:\Users\kacem\Workspace\IFT6285\github-nlp\IFT6285-NLP\TP3\

#   python corrige.py -v voc-1bwc.txt -n 100 -w devoir3-train.txt -d Jaro_Winkler -o unigram > devoir3-sortie.txt