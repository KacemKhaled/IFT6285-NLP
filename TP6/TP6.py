import spacy
from spacy import displacy
import os
import time
import matplotlib.pyplot as plt
import random

# pip install -U spacy
# python -m spacy download en_core_web_sm
#  python -m spacy download en_core_web_md
# python -m spacy download en_core_web_lg

BNC_folder = '1bshort/'


def random_sentences_question2(min_length, max_length, nb_phrases):
    rand_sentences = []

    filename = os.listdir(BNC_folder)[0] #filrst file
    print(filename)

    with open(BNC_folder + filename, 'r', encoding="utf8") as f:

        corpus = f.read()  # TRANCHE
        sentences = corpus.split('\n')  # phrases in the tranche

        while( len(rand_sentences) < nb_phrases) :

            s = random.choice(sentences)
            while (len ( s.split() ) < min_length or  len(s.split()) > max_length   ):
                s = random.choice(sentences)

            rand_sentences.append(s)

    return rand_sentences


def analysis_question1(model_name):

    times = []
    i = 0 #  nombre des phrases analysees
    times.append(0) # when analyse 0 phrases, time is 0

    #load the model
    nlp = spacy.load(model_name)

    #go through all 1bshort folder
    for filename in os.listdir(BNC_folder):
        if i == 50000: break  # stop at 50000 phrases 1/2

        print(filename)

        with open(BNC_folder + filename, 'r', encoding="utf8") as f:

            corpus = f.read()  # TRANCHE
            sentences = corpus.split('\n')  # phrases in the tranche

            print(len(sentences))

            for s in sentences:
                i = i +1
                start_time = time.time()
                doc = nlp(s)
                t = time.time() - start_time
                times.append( times[i-1] + t )
                print(i)

                if i== 50000: break  # stop at 50000 phrases 2/2

    print(f"nb des phrases analysee : { i }  for model {model_name}" )
    return times, i


def plot(times_sm, times_md, times_lg, nb_phrases):
    print('Creating the figure')

    plt.figure(figsize=(9, 6))
    plt.plot(range(nb_phrases +1 ), times_sm, label='en_core_web_sm')
    plt.plot(range(nb_phrases + 1), times_md, label='en_core_web_md')
    plt.plot(range(nb_phrases + 1), times_lg, label='en_core_web_lg')


    plt.title(f"Le temps d'amalyse en fonction du nombre de phrases considérées")
    plt.xlabel("Nombre de phrases considérées")
    plt.ylabel("Le temps d'analyse (en sec)")
    plt.legend()

    plt.savefig(f"plots/courbe-analyse_temps.svg", format="svg")
    plt.savefig(f"plots/courbe-analyse_temps.png", format="png")
    plt.savefig(f"plots/courbe-analyse_temps.eps", format="eps")


def main():
            ###### Question 1
    # times_sm , nb_phrases_sm = analysis_question1('en_core_web_sm')
    # times_md , nb_phrases_md = analysis_question1('en_core_web_md')
    # times_lg , nb_phrases_lg = analysis_question1('en_core_web_lg')
    #
    # assert nb_phrases_md == nb_phrases_lg
    # assert nb_phrases_sm == nb_phrases_md
    #
    # assert len(times_sm) == len(times_lg)
    # assert len(times_lg) == len(times_md)
    #
    # assert nb_phrases_sm + 1 == len(times_sm)
    #
    # plot(times_sm, times_md, times_lg, nb_phrases_sm)


            ###### Question 2
    random_sentences = random_sentences_question2(min_length=5, max_length=30, nb_phrases=5 )
    print(len(random_sentences))
    print(random_sentences)
    # test these on displacy and see
    # some potential wrong sentences #todo: I need  5
    problematic_sentences = [
         'We mourn his loss and our thoughts and prayers are with his family and friends at this very sad time .'
          'We soon discovered that Quest was no Fred Astaire .',
            ]

    print( spacy.explain('NP') )
    # doc = nlp(s)
    # spacy.displacy.serve(doc, style='dep')


if __name__ == '__main__':
    main()