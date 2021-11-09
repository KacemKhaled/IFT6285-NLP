import spacy
from spacy import displacy
import os
import time
import matplotlib.pyplot as plt
import random
from pathlib import Path
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
import textacy

# pip install -U spacy
# python -m spacy download en_core_web_sm
#  python -m spacy download en_core_web_md
# python -m spacy download en_core_web_lg

# pip install svglib

BNC_folder = '1bshort/'


def analysis_question3(model_name):
    nb_triplets_acquis = []
    i = 0  # nombre des phrases analysees
    nb_triplets_acquis.append(0)  # when analyse 0 phrases, nb_triplets_acquis is 0

    all_tuples = [] # <--- need this for Q4

    # load the model
    nlp = spacy.load(model_name)

    # go through all 1bshort folder
    for filename in os.listdir(BNC_folder):
        if i == 50000: break  # stop at 50000 phrases 1/2

        print(filename)

        with open(BNC_folder + filename, 'r', encoding="utf8") as f:

            corpus = f.read()  # TRANCHE
            sentences = corpus.split('\n')  # phrases in the tranche

            print(len(sentences))

            for s in sentences:
                i = i + 1
                selected_tuples = []

                doc = nlp(s)
                tuples = textacy.extract.subject_verb_object_triples(doc)

                for t in tuples:
                    if (t.subject[0].pos_ == 'NOUN' or t.subject[0].pos_ == 'PROPN') and (
                            t.object[0].dep_ == 'dobj') and (t.subject[0].dep_ == 'nsubj'):
                        subject_lemma = t.subject[0].lemma_ if t.subject[0].pos_ != 'PROPN' else 'PROPN'
                        verb_lemma = t.verb[0].lemma_
                        object_lemma = t.object[0].lemma_ if t.object[0].pos_ != 'PROPN' else 'PROPN'
                        new_tuple = (subject_lemma, verb_lemma, object_lemma)
                        selected_tuples.append(new_tuple)

                all_tuples.extend(selected_tuples)
                nb_triplets_acquis.append(nb_triplets_acquis[i - 1] + len(selected_tuples))
                print(i)

                if i == 50000: break  # stop at 50000 phrases 2/2

    print(f"nb des tuples in {i} phrases analysee for model {model_name}")
    return nb_triplets_acquis, i , all_tuples


def render_and_save_parses_pictures_question2(sentences):

    # for each sentence , we parsed with the 2 models and we save it as PDF
    for sentence in sentences:
        nlp = spacy.load('en_core_web_sm') # <------ MODEL 1
        doc = nlp(sentence)

        print('doc', str(doc))
        # spacy.displacy.serve(doc, style='dep', host='localhost', port=5000) <---on the web
        svg = displacy.render(doc, style="dep")
        output_path = Path("./parses/" + str(doc) + "_sm.svg")
        output_path.open("w", encoding="utf-8").write(svg)
        # make it pdf to use in report
        drawing = svg2rlg("./parses/" + str(doc) + "_sm.svg")
        renderPDF.drawToFile(drawing, "./parses/" + str(doc) + "_sm.pdf")


        nlp2 = spacy.load('en_core_web_lg')   # <------ MODEL2
        doc2 = nlp2(sentence)
        print('doc2', doc2)

        # spacy.displacy.serve(doc2, style='dep', host='localhost', port=5001)

        svg = displacy.render(doc2, style="dep")
        output_path = Path("./parses/" + str(doc2) + "_lg.svg")
        output_path.open("w", encoding="utf-8").write(svg)

        drawing = svg2rlg("./parses/" + str(doc2) + "_lg.svg")
        renderPDF.drawToFile(drawing, "./parses/" + str(doc2) + "_lg.pdf")


def random_sentences_question2(min_length, max_length, nb_phrases ):
    rand_sentences = []

    filename = os.listdir(BNC_folder)[0] #first file
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


def plot(times_sm, times_md, times_lg, nb_phrases, title, xlabel, ylabel, name_fig):
    print('Creating the figure')

    plt.figure(figsize=(9, 6))
    plt.plot(range(nb_phrases +1 ), times_sm, label='en_core_web_sm')
    plt.plot(range(nb_phrases + 1), times_md, label='en_core_web_md')
    plt.plot(range(nb_phrases + 1), times_lg, label='en_core_web_lg')


    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.savefig(f"plots/"+name_fig+".svg", format="svg")
    plt.savefig(f"plots/"+name_fig+".png", format="png")
    plt.savefig(f"plots/"+name_fig+".eps", format="eps")


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
    # plot(times_sm, times_md, times_lg, nb_phrases_sm, "Le temps d'amalyse en fonction du nombre de phrases considérées", \
            # "Nombre de phrases considérées" , "Le temps d'analyse (en sec)" , "courbe-analyse_temps" )


            ###### Question 2
    # random_sentences = random_sentences_question2(min_length=5, max_length=8, nb_phrases=5 )
    # print(len(random_sentences))
    # print(random_sentences)
    #
    # problematic_sentences = [
    #      'We mourn his loss and our thoughts and prayers are with his family and friends at this very sad time .', #cNot correctd in lg _ mourn _ are
    #         'During the study , 478 people developed dementia and 376 people developed cancer .', #corrected in lg
    #         'America has endured many such false hopes in Pakistan.' , #  <--- corrected in lg
    #        'They then forced open the main gates to make their escape .',# <--- corrected in lg
    #        'The key point came early in the afternoon.' # <--- corrected in lg
    #
    # ]
    #
    # render_and_save_parses_pictures_question2(problematic_sentences)


            ###### Question 3

    nb_tuples_sm , nb_phrases_sm , all_tuples_sm = analysis_question3('en_core_web_sm')
    nb_tuples_md , nb_phrases_md , all_tuples_md = analysis_question3('en_core_web_md')
    nb_tuples_lg , nb_phrases_lg , all_tuples_lg = analysis_question3('en_core_web_lg')

    assert nb_phrases_md == nb_phrases_lg
    assert nb_phrases_sm == nb_phrases_md

    assert len(nb_tuples_sm) == len(nb_tuples_md)
    assert len(nb_tuples_md) == len(nb_tuples_lg)

    assert nb_phrases_sm + 1 == len(nb_tuples_sm)

    plot(nb_tuples_sm, nb_tuples_md, nb_tuples_lg, nb_phrases_sm, "Le nombre de triplets acquis en fonction du nombre de phrases considérées", \
             "Nombre de phrases considérées" , "Le nombre de triplets acquis" , "courbe-analyse_triplets" )


                ####### Question4
    print('length of the tuples list:')
    print(len(all_tuples_sm))
    print(len(all_tuples_md))
    print(len(all_tuples_lg))

    print('Is all_tuples_sm inclus in all_tuples_md ?' )
    print( all(x in all_tuples_md for x in all_tuples_sm)  )

    print('Is all_tuples_md inclus in all_tuples_lg ?')
    print(all(x in all_tuples_lg for x in all_tuples_md))

    print('Intersection between all_tuples_sm and all_tuples_md :')
    print( len( [x for x in all_tuples_sm if x in all_tuples_md ]) )

    print('Intersection between all_tuples_md and all_tuples_lg :')
    print(  len([x for x in all_tuples_md if x in all_tuples_lg ]))

    print('Intersection between all_tuples_sm and all_tuples_lg :')
    print(len([ x for x in all_tuples_sm if x in all_tuples_lg ]))

    print('Intersection between 3 models :')
    print(len([ x  for x in all_tuples_sm if ( x in all_tuples_lg and x in all_tuples_md ) ] ))

    # save the tuples in a files:
    with open('tuples_sm.txt', 'w', encoding='utf-8') as f1:
            for tuple in all_tuples_sm:
                f1.write('%s\n' % str(tuple))
    f1.close()

    with open('tuples_md.txt', 'w', encoding='utf-8') as f2:
        for tuple in all_tuples_md:
            f2.write('%s\n' % str(tuple))
    f2.close()

    with open('tuples_lg.txt', 'w', encoding='utf-8') as f3:
        for tuple in all_tuples_lg:
            f3.write('%s\n' % str(tuple))
    f3.close()


    #todo: maybe a viz,
            # https://towardsdatascience.com/mining-an-economic-news-article-using-pre-trained-language-models-f75af041ecf0
            #
                 ####### Others

    # nlp = spacy.load('en_core_web_sm')
    #
    # doc = nlp('Hello my friend. How are you? I am Mouna. I go shopping, John ate two green apples this morning,')
    # print(type(doc))
    #
    # print('sentences', list(doc.sents))
    # print('Noun chunks', list(doc.noun_chunks))

    # print('Noun chunks')
    # for nc in doc.noun_chunks:
    #     print(nc)
    #     print([(token.text, token.tag_, token.pos_) for token in nc])

    # for token in doc:
    #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #                   token.shape_, token.is_alpha, token.is_stop)

    #
    # selected_tupes = []
    # tuples = textacy.extract.subject_verb_object_triples(doc)
    # print('tuples')
    # for t in tuples:
    #     print(t)
    #     print(t.subject)
    #     print(t.subject[0])
    #     print(t.subject[0].pos_)
    #     print(t.subject[0].tag_)
    #
    #
    #     if ( t.subject[0].pos_ == 'NOUN' or t.subject[0].pos_ == 'PROPN' ) and ( t.object[0].dep_ == 'dobj' ) and ( t.subject[0].dep_ =='nsubj') :
    #         subject_lemma = t.subject[0].lemma_ if t.subject[0].pos_ != 'PROPN' else 'PROPN'
    #         verb_lemma = t.verb[0].lemma_
    #         object_lemma = t.object[0].lemma_ if t.object[0].pos_ != 'PROPN' else 'PROPN'
    #         new_tuple = (subject_lemma, verb_lemma, object_lemma )
    #         selected_tupes.append( new_tuple)

    # print('selected_tuples: ', selected_tupes)


    print('NP:', spacy.explain('NP'))
    print('NNP:', spacy.explain('NNP'))
    print('PRP:', spacy.explain('PRP'))
    print('PROPN', spacy.explain('PROPN'))


if __name__ == '__main__':
    main()