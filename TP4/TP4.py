# Based on:
# http://www-labs.iro.umontreal.ca/~felipe/IFT6285-Automne2020/word2vec-gensim-train.py
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html

import gensim
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.test.utils import datapath
import argparse
import multiprocessing
from time import time
from os import listdir
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import logging
import numpy as np
from gensim.models.word2vec import  PathLineSentences, LineSentence
from gensim import utils



folder = 'training-monolingual.tokenized.shuffled/'
short_folder = 'training-monolingual.tokenized.shuffled_short/'
words_list_file= 'liste_mots_devoir4.txt'

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __init__(self,path,files) -> None:
        self.path = path
        self.files = files

    def __iter__(self):
        for f in self.files:
            for line in open(self.path+f, 'r', encoding="utf8"):
                # assume there's one document per line, tokens separated by whitespace
                yield utils.simple_preprocess(line)

def get_args():
    """
    -vector_size : int, optional
    Dimensionality of the word vectors.
    -window : int, optional
        Maximum distance between the current and predicted word within a sentence.
    -min_count : int, optional
        Ignores all words with total frequency lower than this.
    -negative : int, optional
    If > 0, negative sampling will be used, the int for negative specifies how many "noise words" 
    should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
    """

    parser = argparse.ArgumentParser(description="train a gensim word2vec model from text read on stdin")

    parser.add_argument("-v", '--verbosity', type=int, help="increase output verbosity", default=0)
    parser.add_argument("-n", '--nb', type=int, help="# of input lines to read", default=None)
    parser.add_argument("-d", '--datapath', type=str, help="directory where txt files can be found", default=None)
    parser.add_argument("-f", '--name', type=str, help="basename modelname", default="genw2v")
    parser.add_argument("-s", '--size', type=int, help="dim of vectors", default=100)
    parser.add_argument("-g", '--negative', type=int, help="# neg samples", default=5)
    parser.add_argument("-c", '--mincount', type=int, help="min count of a word", default=1)
    parser.add_argument("-w", '--window', type=int, help="window size", default=5)
    parser.add_argument("-t", '--tranches', type=int, help="nb of tranches", default=5)

    return parser.parse_args()

def plot_from_file(folder,ext):
    times,_,sent_len = np.loadtxt(folder+ext+'.csv',delimiter=',')

    print('Creating the figure')
    plt.tight_layout() 
    plt.figure(figsize=(5, 4))
    plt.plot(sent_len, times, 'r-')
    plt.title(f"Le temps d'entrainement en\nfonction des phrases considérées")
    plt.xlabel("Nombre de phrases considérées")
    plt.ylabel("Le temps d'entrainement (en sec)")
    plt.savefig(f"plots/courbe-{ext}.svg",format="svg")
    plt.savefig(f"plots/courbe-{ext}.png", format="png")
    plt.savefig(f"plots/courbe-{ext}.eps", format="eps")

def plot(ext,times, sent_len):

    print('Creating the figure')

    plt.figure(figsize=(9, 6))
    plt.plot(sent_len, times, 'r-')
    
    plt.tight_layout() 
    plt.title(f"{ext}\nLe temps d'entrainement en fonction des phrases considérées")
    plt.xlabel("Nombre de phrases considérées")
    plt.ylabel("Le temps d'entrainement (en sec)")
    plt.savefig(f"plots/courbe-{ext}.svg",format="svg")
    plt.savefig(f"plots/courbe-{ext}.png", format="png")
    plt.savefig(f"plots/courbe-{ext}.eps", format="eps")


def similar_words(file,modelname,ext):
    model = Word2Vec.load(modelname)
    unrecognized_words = []
    with open(file, 'r', encoding="utf8") as in_file, \
     open('outputs/out_'+ext+'_'+file, mode='w') as out_file:
        lines = in_file.readlines()
        print(f"Looking for similar words for {len(lines)} words")
        for line in tqdm(lines):
            word = line.strip()
            try:
                liste = model.wv.most_similar(positive=[word])[:10]
                line_out = ""
                for x in liste:
                    line_out+= f"{x[0]} [{x[1]:.2f}] " 
                out_file.write(f"{word}\t{line_out}\n")
            except:
                unrecognized_words.append(word)
                print(f"The word: '{word}' is not in vocab")
    np.savetxt('outputs/out_'+ext+'_unrecognized_words.txt', unrecognized_words ,delimiter ="\n", fmt ='% s')    
    in_file.close()
    out_file.close()

def checkpoint(ext,save_model,w2v_model,times,sizes,sent_len):
    
    modelname = f"models/{ext}.w2v"
    csv_times_sizes= f"outputs/{ext}.csv"

    if save_model:
        # stream the model
        #w2v_model.init_sims(replace=True) # deprecated: calls to init_sims() are unnecessary
        w2v_model.save(modelname)
        logging.info(f"saved {modelname}")
 
    np.savetxt(csv_times_sizes, [times,sizes,sent_len],delimiter =", ", fmt ='% s')
    logging.info(f"saved {csv_times_sizes}")
    

def train(args,  save_model = True , data = folder,nb_tranches=10,option=1,ext='name'):
    # Utilisez gensim pour entrainer des representations vectorielles sur tout ou partie du 1BWC.
    #train by tranche + save the time for each tranche
    print('Starting the training ',option)
    files = listdir(data)[:nb_tranches]
    sents = []
    times = []
    sizes = []
    sent_len = []
    nb_sentences = 0
    # change implementation to stream corpus directly from disk to not overload the RAM
    # option 1: load through RAM
    # option 2: load through disk
    option = 2 if nb_tranches>10 else option
    if nb_tranches==99:
        allfiles = True
        print('all files')
    if option==2:
        training_sample = "training_sample.txt" 
        myfile = open("training_sample.txt", "w")
        myfile.close()

    for i,fn in enumerate(files):
        a = 99 if allfiles else i+1
        print(f"\n{'#-'*10}-Files 1..{a}:")
        logging.info(f"\n{'#-'*10}-Files 1..{a}:")
        t = time()
        if option==1:
            sents = list(MyCorpus(data,files[:])) if allfiles else list(MyCorpus(data,files[:i+1])) # or LineSentence(data+fn) where fn is file containing whole corpus
            print(f"- Temps de lecture de sentences (en secondes): {(time() - t)}\n")
            print(f"Senetences found  {len(sents)} phrases")
            logging.info(f"- Temps de lecture de sentences (en secondes): {(time() - t)}\n")
            logging.info(f"Senetences found  {len(sents)} phrases")
            sent_len.append(len(sents))
            
        elif option==2:
            # concatenate corpus files because LineSentence() takes only a single file
            if not allfiles: 
                os.system("cat "+data+fn+ " >> "+ training_sample)
            t = time()
            sents = PathLineSentences(data) if allfiles else LineSentence(training_sample)
            print(f"- Temps de lecture de sentences (en secondes): {(time() - t)}\n")
            logging.info(f"- Temps de lecture de sentences (en secondes): {(time() - t)}\n")
            l = len(open(data+fn, 'r', encoding="utf8").readlines())
            print(f"Sentences found : {l}")
            logging.info(f"Sentences found : {l}")
            nb_sentences+=l
            print(f"Total Sentences processed : {nb_sentences}")
            logging.info(f"Total Sentences processed : {nb_sentences}")
            sent_len.append(nb_sentences)

            
        # Time to build model 
        t = time()
        w2v_model = Word2Vec(
            sentences = sents,
            min_count=args.mincount, window=args.window,  vector_size=args.size,negative=args.negative,
            min_alpha=1e-4, workers=(os.cpu_count()*2 - 1), sample=0.01, alpha=1e-2, epochs=5
            )
        end = round((time() - t),2)
        times.append(end)
        print(f"- Temps d'entrainement (en secondes): {end}\n")

        # Size of the model
        size = round(w2v_model.estimate_memory()['total']/(1024*1024),2)
        sizes.append(size)

        print(f"- Taille du modele sur disque (en octets)): {w2v_model.estimate_memory()}\n\
                Total en MB: {size}")
        logging.info(f"- Taille du modele sur disque (en octets)): {w2v_model.estimate_memory()}\n\
                Total en MB: {size}")

        print("\n- Nombre de mots encodés (= taille du vocab): %d\n" % len(w2v_model.wv.vectors))
        logging.info("\n- Nombre de mots encodés (= taille du vocab): %d\n" % len(w2v_model.wv.vectors))
        #print(w2v_model.most_similar(positive=['abnormalities'], topn = 10))
        #w2v_model.init_sims(replace=True)
        print(w2v_model.wv.most_similar(positive=['abnormalities'])[:10])
        logging.info(w2v_model.wv.most_similar(positive=['abnormalities'])[:10])
        
        if allfiles: 
            break

    checkpoint(ext,save_model,w2v_model,times,sizes,sent_len)

    return times, sent_len, sizes

def main():
    st = time()
    args = get_args()
    save_model = True 
    option = 2
    nb_tranches=args.tranches

    ext = "{}-t{}-size{}-window{}-neg{}-mincount{}".format(args.name, args.tranches, args.size, args.window, args.negative, args.mincount)
    logname  = f"logs/{ext}.log"
    
    logging.basicConfig(filename=logname,datefmt= '%H:%M:%S', format='%(message)s', level=logging.DEBUG)

    times, sent_len, sizes = train(args,save_model, folder, nb_tranches, option,ext)
    print(f"Total execution time: {round((time()-st)/60,2)} - program: {option}")

    similar_words(words_list_file,f"models/{ext}.w2v",ext)

    try:
        plot(ext,times, sent_len)
    except:
        print("could not plot from list variables, plotting from file: "+ext+'.csv')
        plot_from_file('outputs/',ext)

if __name__ == '__main__':
    main()
