import gensim
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.phrases import Phrases, Phraser
import argparse
import multiprocessing
from time import time
from os import listdir
from tqdm import tqdm
import matplotlib.pyplot as plt

folder = 'training-monolingual.tokenized.shuffled/'
short_folder = 'training-monolingual.tokenized.shuffled_short/'


times = []


def get_args():

    parser = argparse.ArgumentParser(description="train a gensim word2vec model from text read on stdin")

    parser.add_argument("-v", '--verbosity', type=int, help="increase output verbosity", default=0)
    parser.add_argument("-n", '--nb', type=int, help="# of input lines to read (per file I think)", default=None)
    parser.add_argument("-d", '--datapath', type=str, help="directory where txt files can be found", default=None)
    parser.add_argument("-f", '--name', type=str, help="basename modelname", default="genw2v")
    parser.add_argument("-g", '--negative', type=int, help="# neg samples", default=100)
    parser.add_argument("-c", '--mincount', type=int, help="min count of a word", default=1)
    parser.add_argument("-w", '--window', type=int, help="window size", default=2)
    return parser.parse_args()


def plot(nb_of_tranches = 100):
    print('Creating the figure')
    tranches = list(range(1, nb_of_tranches))

    plt.figure(figsize=(9, 6))
    plt.plot(tranches, times, 'r--')
    plt.title("Le temps mis pour entrainer les modeles en fonction du nombre de tranches considérées")
    plt.xlabel("Nombre de tranches considérées")
    plt.ylabel("Le temps mis pour entrainer les modeles (en secondes)")
    plt.savefig("courbe.svg",format="svg")
    plt.savefig("courbe.png", format="png")


def train(args,  save_model = False , data = folder):
    # Utilisez gensim pour entrainer des representations vectorielles sur tout ou partie du 1BWC.
    #train by tranche + save the time for each tranche
    print('Starting the training')
    files = listdir(data)
    sents = []

    for fn in tqdm(files):
        with open(data + fn, 'r', encoding="utf8") as f:
            corpus = f.read()  # TRANCHE
            sentences = corpus.split('\n')  #phrases in the tranche

            sents.append(sentences) # train accumulative

            phrases = Phrases(sents, min_count=10, progress_per=1000)
            bigram = Phraser(phrases)
            sents_b = bigram[sents]

            start_time = time()
            w2v_model = Word2Vec(min_count=args.mincount, window=args.window, sample=6e-5, alpha=0.03,
                                     min_alpha=0.0007, negative=args.negative, workers= multiprocessing.cpu_count() -1)
            w2v_model.build_vocab(sents_b, progress_per=10000)
            w2v_model.train(sents_b, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
            times.append( time() - start_time )

            print('times: '+ str(times))


    if save_model:
        w2v_model.init_sims(replace=True)
        w2v_model.save('word2vec.model')



def main():
    args = get_args()
    train(args, data = folder)  # remove the data argument for all tranches
    plot() # remove the data argument for all  tranches  nb_of_tranches = 11

if __name__ == '__main__':
    main()
