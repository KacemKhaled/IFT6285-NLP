import pandas as pd
import argparse
import os.path as osp
from os.path import dirname, abspath
import logging

logging.basicConfig(format='%(message)s', level=logging.DEBUG)

def evaluate(file_to_evaluate, reference, evaluation_measure):

    #read file_to_evaluate
    with open(file_to_evaluate, 'r', encoding="utf8") as f:
       corrections = {}
       for line in f:
            wrong_word = line.rstrip().split('\t')[0]
            suggestions = line.rstrip().split('\t')[1:]

            corrections[wrong_word] = suggestions
    f.close()

    print('corrections : ', corrections)


    #read_the_reference
    with open(reference, 'r', encoding="utf8") as r:
        references = {}
        for line in r:
            wrong_word = line.rstrip().split('\t')[0]
            correction = line.rstrip().split('\t')[1]

            references[wrong_word] = correction
    r.close()
    print('references : ', references)


    #depending on the evaluation metric , calculate the result
    score = 0
    if evaluation_measure == 0 : # if references exists among the 5 suggestations +1
        for wrong_word in corrections.keys():
            if references[wrong_word] in corrections[wrong_word]:
                score = score +1
        score = ( score / len(corrections.keys()) ) * 100

    else:
        # depending on the order
        for wrong_word in corrections.keys():
            if references[wrong_word] in corrections[wrong_word]:
                indice = corrections[wrong_word].index( references[wrong_word] )
                score = score +  ( 5 - indice )  # 5 est le nombre de suggestations  que notre prog donne, au plus
        score =  ( score /  (len(corrections.keys()) * 5)  ) *100


    # print + log the results
    print(score)
    message = f"score is = {score:.2f} %"
    logging.info( message )

reference = 'devoir3-train.txt'

CONFIG_PATH = abspath(__file__)
SRC_ROOT = dirname(CONFIG_PATH)
TRAIN_ROOT = osp.join(SRC_ROOT, reference)


parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('-f','--file_to_evaluate',
                    help='Path to the file to evaluate:')

parser.add_argument('-r','--reference', default=TRAIN_ROOT,
                    help='Path to reference file: (default: ./devoir3-train.txt)')

parser.add_argument('-e','--evaluation_measure', default=0,type=int,
                    help=': Evaluation measure to use (default: 0)')


def main():
    args = parser.parse_args()
    evaluate('test-mouna.txt', args.reference, 0 )
    # eval(args.file_to_evaluate, args.reference, args.evaluation_measure)



if __name__ == '__main__':
    main()


#   python eval.py -f test-mouna.txt  -r devoir3-train.txt    -e 0   >   devoir3-eval.txt