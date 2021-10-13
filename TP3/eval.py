#import pandas as pd
import argparse
import os.path as osp
from os.path import dirname, abspath
import logging
import time


def evaluate(file_to_evaluate, reference, evaluation_measure):
    # read file_to_evaluate
    with open(file_to_evaluate, 'r', encoding="utf8") as f:
        corrections = {}
        for line in f:
            wrong_word = line.rstrip().split('\t')[0]
            suggestions = line.rstrip().split('\t')[1:]

            corrections[wrong_word] = suggestions
    f.close()
    # print('corrections : ', corrections)

    # read_the_reference
    with open(reference, 'r', encoding="utf8") as r:
        references = {}
        for line in r:
            wrong_word = line.rstrip().split('\t')[0]
            correction = line.rstrip().split('\t')[1]

            references[wrong_word] = correction
    r.close()
    # print('references : ', references)

    # depending on the evaluation metric , calculate the result
    score = 0
    if evaluation_measure == 'no_order':  # if references exists among the 5 suggestions +1
        for wrong_word in corrections.keys():
            if references[wrong_word] in corrections[wrong_word]:
                score = score + 1
        score = (score / len(corrections.keys())) * 100

    else: #if evaluation_measure == 'by_order'
        # depending on the order
        for wrong_word in corrections.keys():
            if references[wrong_word] in corrections[wrong_word]:
                indice = corrections[wrong_word].index(references[wrong_word])
                score = score + (5 - indice)  # 5 est le nombre de suggestions  que notre prog donne, au plus
        score = (score / (len(corrections.keys()) * 5)) * 100

    # print + log the results
    print(score)
    message = f"score is = {score:.2f} %"
    logging.info(message)
    logging.info(score)


reference = 'devoir3-train.txt'

CONFIG_PATH = abspath(__file__)
SRC_ROOT = dirname(CONFIG_PATH)
TRAIN_ROOT = osp.join(SRC_ROOT, reference)

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('-f', '--file_to_evaluate',
                    help='Path to the file to evaluate:')

parser.add_argument('-r', '--reference', default=TRAIN_ROOT,
                    help='Path to reference file: (default: ./devoir3-train.txt)')

parser.add_argument('-e', '--evaluation_measure', default='no_order', type=str,
                    choices=['no_order','by_order'],
                    help=': Evaluation measure to use (default: no_order)')


def main():
    args = parser.parse_args()
    logfilename = f"eval/logs-f_{args.file_to_evaluate[4:-4]}-r_{args.reference[:-4]}-e_{args.evaluation_measure}-{time.time()}.txt"
    logging.basicConfig(filename=logfilename, format='%(message)s', level=logging.DEBUG)
    logging.info(f"file_to_evaluate: {args.file_to_evaluate}")
    logging.info(f"reference file: {args.reference}")
    logging.info(f"evaluation_measure:{args.evaluation_measure}")
    #evaluate('test-mouna.txt', args.reference, 'no_order')
    evaluate(args.file_to_evaluate, args.reference, args.evaluation_measure)


if __name__ == '__main__':
    main()


#  python eval.py -f devoir3-sortie.txt -r devoir3-train.txt -e no_order