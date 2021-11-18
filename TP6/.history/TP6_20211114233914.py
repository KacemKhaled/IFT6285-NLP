"""
Readme:
Installation:

pip install -U pip setuptools wheel

# GPU 
pip install -U spacy[cuda101]

# CPU
pip install -U spacy

python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_trf

pip install svglib
pip install wordcloud
pip install textacy

"""

__author__ = "Kacem Khaled and Mouna Dhaouadi"
__status__ = "Development"

import spacy
from spacy import displacy
import os
import time
import matplotlib.pyplot as plt
import random
from pathlib import Path
from svglib.svglib import svg2rlg
# from reportlab.graphics import renderPDF, renderPM
import textacy
from collections import Counter
from tqdm import tqdm
import torch
import pandas as pd
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns
from venn import venn


MAX_SENTENCES = 100000

print(f"GPU Available: {torch.cuda.is_available()}")




BNC_folder = '1bshort/'

def analysis_question3(model_name,gpu=False):
    nb_triplets_acquis = []
    i = 0  # nombre des phrases analysees
    nb_triplets_acquis.append(0)  # when analyse 0 phrases, nb_triplets_acquis is 0

    all_tuples = [] # <--- need this for Q4

    # load the model
    
    activated = False
    if gpu: activated =  spacy.prefer_gpu()
    print(f"GPU Used: {activated}")
    # spacy.require_gpu()

    nlp = spacy.load(model_name)

    # go through all 1bshort folder
    for filename in sorted(os.listdir(BNC_folder)):
        max_sentences = MAX_SENTENCES
        if i == max_sentences: break  # stop at 50000 phrases 1/2

        print(filename)

        with open(BNC_folder + filename, 'r', encoding="utf8") as f:

            corpus = f.read()  # TRANCHE
            sentences = corpus.split('\n')  # phrases in the tranche

            print(f"File: {filename}\thas: {len(sentences)} sentences")
            if len(sentences) > max_sentences-i: sentences=sentences[:max_sentences-i]

            for s in tqdm(sentences):
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
                # print(i)
                if i%1000==0 : pd.DataFrame(all_tuples, columns=['subject_lemma', 'verb_lemma', 'object_lemma']).to_csv(f'all_tuples{model_name[11:]}_3.csv',encoding='utf-8',index=False)
                if i%1000==0 : pd.DataFrame(nb_triplets_acquis, columns=['nb_triplets']).to_csv(f'nb_triplet_{model_name[11:]}_3.csv',encoding='utf-8',index=False)
                
                if i == max_sentences: break  # stop at 50000 phrases 2/2

    pd.DataFrame(all_tuples, columns=['subject_lemma', 'verb_lemma', 'object_lemma']).to_csv(f'all_tuples{model_name[11:]}_3.csv',encoding='utf-8',index=False)
    pd.DataFrame(nb_triplets_acquis, columns=['nb_triplets']).to_csv(f'nb_triplet_{model_name[11:]}_3.csv',encoding='utf-8',index=False)
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
        # renderPDF.drawToFile(drawing, "./parses/" + str(doc) + "_sm.pdf")


        nlp2 = spacy.load('en_core_web_lg')   # <------ MODEL2
        doc2 = nlp2(sentence)
        print('doc2', doc2)

        # spacy.displacy.serve(doc2, style='dep', host='localhost', port=5001)

        svg = displacy.render(doc2, style="dep")
        output_path = Path("./parses/" + str(doc2) + "_lg.svg")
        output_path.open("w", encoding="utf-8").write(svg)

        # THEse lines RENDER pdf, but without annotations on the arcs, so I used an online tool for the conversion to pdf
        # drawing = svg2rlg("./parses/" + str(doc2) + "_lg.svg")
        # renderPDF.drawToFile(drawing, "./parses/" + str(doc2) + "_lg.pdf")


def random_sentences_question2(min_length, max_length, nb_phrases ):
    rand_sentences = []

    filename = os.listdir(BNC_folder)[0] #first file
    print(filename)

    with open(BNC_folder + filename, 'r', encoding="utf8") as f:

        corpus = f.read()  # TRANCHE
        sentences = corpus.split('\n')  # phrases in the tranche

        while( len(rand_sentences) < nb_phrases) :

            s = random.choice(sentences)
            while (len ( s.split() ) < min_length or  len(s.split()) > max_length  or " 's " not in s or " in " not in s ) :
                s = random.choice(sentences)

            rand_sentences.append(s)

    return rand_sentences



def analysis_question1(model_name, gpu=False):
    max_sentences = MAX_SENTENCES
    times = []
    i = 0 #  nombre des phrases analysees
    times.append(0) # when analyse 0 phrases, time is 0

    # load the model
    activated = False
    if gpu: activated =  spacy.prefer_gpu()
    print(f"GPU Used: {activated}")
    # spacy.require_gpu()

    nlp = spacy.load(model_name)

    #go through all 1bshort folder
    for filename in os.listdir(BNC_folder):
        if i == max_sentences: break  # stop at 50000 phrases 1/2

        print(filename)

        with open(BNC_folder + filename, 'r', encoding="utf8") as f:

            corpus = f.read()  # TRANCHE
            sentences = corpus.split('\n')  # phrases in the tranche

            print(f"File: {filename}\thas: {len(sentences)} sentences")
            if len(sentences) > max_sentences-i: sentences=sentences[:max_sentences-i]

            for s in tqdm(sentences):
                i = i +1
                start_time = time.time()
                doc = nlp(s)
                t = time.time() - start_time
                times.append( times[i-1] + t )
                # if i%500 ==0 : print(i)
                if i%1000 ==0: pd.DataFrame(times, columns=[f"times_{model_name[11:]}"]).to_csv(f"times_{model_name[11:]}.csv",encoding='utf-8')
                

                if i== max_sentences: break  # stop at 50000 phrases 2/2
    pd.DataFrame(times, columns=[f"times_{model_name[11:]}"]).to_csv(f"times_{model_name[11:]}.csv",encoding='utf-8')
    print(f"nb des phrases analysee : { i }  for model {model_name}" )
    return times, i


def plot(times_sm, times_md, times_lg, times_trf, nb_phrases, title, xlabel, ylabel, name_fig):
    print('Creating the figure')

    plt.figure(figsize=(9, 6))
    plt.plot(range(nb_phrases +1 ), times_sm, label='en_core_web_sm')
    plt.plot(range(nb_phrases + 1), times_md, label='en_core_web_md')
    plt.plot(range(nb_phrases + 1), times_lg, label='en_core_web_lg')
    plt.plot(range(nb_phrases + 1), times_trf, label='en_core_web_trf')


    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.savefig(f"plots/"+name_fig+".svg", format="svg")
    plt.savefig(f"plots/"+name_fig+".png", format="png")
    plt.savefig(f"plots/"+name_fig+".eps", format="eps")


def plot_from_csv(nb_phrases,title, xlabel, ylabel, name_fig):
    print('Creating the figure')
    times_sm = list(pd.read_csv('times__sm.csv'))
    times_md = list(pd.read_csv('times__md.csv'))
    times_lg = list(pd.read_csv('times__lg.csv'))
    times_trf = list(pd.read_csv('times__trf.csv'))
    plt.figure(figsize=(9, 6))
    plt.plot(range(nb_phrases +1 ), times_sm, label='en_core_web_sm')
    plt.plot(range(nb_phrases + 1), times_md, label='en_core_web_md')
    plt.plot(range(nb_phrases + 1), times_lg, label='en_core_web_lg')
    plt.plot(range(nb_phrases + 1), times_trf, label='en_core_web_trf')


    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.savefig(f"plots/"+name_fig+".svg", format="svg")
    plt.savefig(f"plots/"+name_fig+".png", format="png")
    plt.savefig(f"plots/"+name_fig+".eps", format="eps")


def plot1():
    sns.set(rc={'figure.figsize':(10,4)})

    files = ('times__sm','times__md','times__lg','times__trf')
    def plot_from_csv(args):
        sns.set_theme(style="darkgrid")

        fig, ax = plt.subplots()
        
        for y in args:
            df = pd.read_csv(f'{y}.csv')
            sns.lineplot(x='Unnamed: 0',y=y, label = f'en_core_web{y[6:]}', ax=ax,data=df)

        ax.set_xlabel('Nb. de phrases considerées')
        ax.set_ylabel("Temps d'analyse (en s)")
        ax.set_title("Le temps d’analyse (en secondes) en fonction du nombre de phrases analysées")
        ax2 = plt.axes([0.4, 0.65, .15, .15], facecolor='w')
        for y in args:
            df = pd.read_csv(f'{y}.csv')
            sns.lineplot(x='Unnamed: 0',y=y, ax=ax2,data=df)

        ax2.set_xlabel('')
        ax2.set_ylabel('')
        ax2.set_title('zoom')
        ax2.set_ylim([575,590])
        ax2.set_xlim([99000,100000])
        plt.savefig('plots/times.png')
        plt.savefig('plots/times.eps')
    plot_from_csv(files)


def plot3():
    sns.set(rc={'figure.figsize':(8,4)})

    files = ('nb_triplet__sm','nb_triplet__md','nb_triplet__lg','nb_triplet__trf')
    def plot_from_csv(args):
        sns.set_theme(style="darkgrid")

        fig, ax = plt.subplots()
        
        for y in args:
            df = pd.read_csv(f'{y}_3.csv').reset_index()
            sns.lineplot(x='index',y='nb_triplets', label = f'en_core_web{y[11:]}', ax=ax,data=df)

        ax.set_xlabel('Nb. de phrases considerées')
        ax.set_ylabel("Le nombre des triplets acquis")
        ax.set_title("Le nombre des triplets acquis en fonction du nombre de phrases analysées.")
        ax2 = plt.axes([0.7, 0.2, .2, .4], facecolor='w')
        for y in args:
            df = pd.read_csv(f'{y}_3.csv').reset_index()
            print(len(df))
            sns.lineplot(x='index',y='nb_triplets', ax=ax2,data=df)

        ax2.set_xlabel('')
        ax2.set_ylabel('')
        ax2.set_title('zoom')
        ax2.set_ylim([24930,25040])
        ax2.set_xlim([99970,99999])
        plt.savefig('plots/triplets.png')
        plt.savefig('plots/triplets.eps')
    plot_from_csv(files)

def plot3_1():
    files = ('all_tuples_sm','all_tuples_md','all_tuples_lg','all_tuples_trf')#,'all_tuples_lg','tuples_trf')
    names= ('sm','md','lg','trf')
    def plot_from_csv_venn(args):
        # sns.set_theme(style="darkgrid")

        # fig, ax = plt.subplots()
        sets = {}
        for y in args:
            df = pd.read_csv(f'{y}_3.csv')
            print('all: ',len(df))
            tuples = set(zip(df['subject_lemma'],df['verb_lemma'],df['object_lemma']))
            print('unique: ',len(tuples))
            sets.update({f"Triplets générés par le modèle: {names[files.index(y)]}":tuples})
            # df['len tuples'] = tuples
            # sns.lineplot(x='index',y=y, label = f'en_core_web{y[6:]}', ax=ax,data=df)
        # print(sets)
        fig, ax = plt.subplots(1, figsize=(7,7))
        venn(sets, ax=ax)
        
        plt.savefig('plots/tuples4.png',transparent=True)
        plt.savefig('plots/tuples4.eps',transparent=True)
    plot_from_csv_venn(files)


def create_word_cloud(text , figure_name):

    # Create and generate a word cloud image:
    wordcloud = WordCloud(background_color="white").generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(f"plots/cloud_"+figure_name+".png", format='png',transparent=True)
    plt.savefig(f"plots/cloud_"+figure_name+".eps", format='eps',transparent=True)

def q1():
    times_sm , nb_phrases_sm = analysis_question1('en_core_web_sm')
    times_md , nb_phrases_md = analysis_question1('en_core_web_md')
    times_lg , nb_phrases_lg = analysis_question1('en_core_web_lg')
    times_trf, nb_phrases_lg = analysis_question1('en_core_web_trf',gpu=True)

    assert nb_phrases_md == nb_phrases_lg
    assert nb_phrases_sm == nb_phrases_md

    assert len(times_sm) == len(times_lg)
    assert len(times_lg) == len(times_md)

    assert nb_phrases_sm + 1 == len(times_sm)

    plot(times_sm, times_md, times_lg,times_trf, nb_phrases_sm, "Le temps d'amalyse en fonction du nombre de phrases considérées", \
    "Nombre de phrases considérées" , "Le temps d'analyse (en sec)" , "courbe-analyse_temps" )
    plot_from_csv(nb_phrases_sm, "Le temps d'amalyse en fonction du nombre de phrases considérées", \
    "Nombre de phrases considérées" , "Le temps d'analyse (en sec)" , "courbe-analyse_temps" )

def q2():
    random_sentences = random_sentences_question2(min_length=5, max_length=8, nb_phrases=5)
    print(len(random_sentences))
    print(random_sentences)

    problematic_sentences = [
        "LeBron 's can go in summer storage .", "That 's what people come in for .",
        "It 's in it 's fifth printing .",
        'We mourn his loss and our thoughts and prayers are with his family and friends at this very sad time .',
        # cNot correctd in lg _ mourn _ are
        'During the study , 478 people developed dementia and 376 people developed cancer .',  # corrected in lg
        'America has endured many such false hopes in Pakistan.',  # <--- corrected in lg
        'They then forced open the main gates to make their escape .',  # <--- corrected in lg
        'The key point came early in the afternoon.'  # <--- corrected in lg

    ]

    render_and_save_parses_pictures_question2(problematic_sentences)

def q3():
    nb_tuples_sm, nb_phrases_sm, all_tuples_sm = analysis_question3('en_core_web_sm')
    # save the tuples in a files:
    with open('tuples_sm.txt', 'w', encoding='utf-8') as f1:
        for tuple in all_tuples_sm:
            f1.write('%s\n' % str(tuple))
    f1.close()

    nb_tuples_md, nb_phrases_md, all_tuples_md = analysis_question3('en_core_web_md')
    with open('tuples_md.txt', 'w', encoding='utf-8') as f2:
        for tuple in all_tuples_md:
            f2.write('%s\n' % str(tuple))
    f2.close()

    nb_tuples_lg, nb_phrases_lg, all_tuples_lg = analysis_question3('en_core_web_lg')
    with open('tuples_lg.txt', 'w', encoding='utf-8') as f3:
        for tuple in all_tuples_lg:
            f3.write('%s\n' % str(tuple))
    f3.close()
    nb_tuples_trf, nb_phrases_trf, all_tuples_trf = analysis_question3('en_core_web_trf',gpu=True)

    with open('tuples_trf.txt', 'w', encoding='utf-8') as f4:
        for tuple in all_tuples_trf:
            f4.write('%s\n' % str(tuple))
    f4.close()

    assert nb_phrases_md == nb_phrases_lg
    assert nb_phrases_sm == nb_phrases_md

    assert len(nb_tuples_sm) == len(nb_tuples_md)
    assert len(nb_tuples_md) == len(nb_tuples_lg)

    assert nb_phrases_sm + 1 == len(nb_tuples_sm)

    plot(nb_tuples_sm, nb_tuples_md, nb_tuples_lg,nb_tuples_trf, nb_phrases_sm,
         "Le nombre de triplets acquis en fonction du nombre de phrases considérées", \
         "Nombre de phrases considérées", "Le nombre de triplets acquis", "courbe-analyse_triplets")

    print('length of the tuples list:')
    print(len(all_tuples_sm))
    print(len(all_tuples_md))
    print(len(all_tuples_lg))
    print(len(all_tuples_trf))

    print('Is all_tuples_sm inclus in all_tuples_md ?')
    print(all(x in all_tuples_md for x in all_tuples_sm))

    print('Is all_tuples_md inclus in all_tuples_lg ?')
    print(all(x in all_tuples_lg for x in all_tuples_md))

    print('Intersection between all_tuples_sm and all_tuples_md :')
    print(len([x for x in all_tuples_sm if x in all_tuples_md]))

    print('Intersection between all_tuples_md and all_tuples_lg :')
    print(len([x for x in all_tuples_md if x in all_tuples_lg]))

    print('Intersection between all_tuples_sm and all_tuples_lg :')
    print(len([x for x in all_tuples_sm if x in all_tuples_lg]))

    print('Intersection between 3 models :')
    print(len([x for x in all_tuples_sm if (x in all_tuples_lg and x in all_tuples_md)]))

    

    

    
    

def q4():
    # load les triplets md
    with open('tuples_md.txt', 'r', encoding='utf-8') as f1:
        sentences = f1.readlines()
    f1.close()
    print(len(sentences))

    # extract some informations
    infos = {'man': [], 'woman': [], 'teacher': [], 'student': [], 'girl': [], 'boy': [], 'police': []}
    for s in sentences:
        s = s.replace('(', '').replace(')', '').replace(',', '').replace("'", "").replace('\n', '')
        print(s)
        if s.split(' ')[0] == 'man':
            infos['man'].append(s.split(' ')[1] + " " + s.split(' ')[2])

        if s.split(' ')[0] == 'woman':
            infos['woman'].append(s.split(' ')[1] + " " + s.split(' ')[2])

        if s.split(' ')[0] == 'teacher':
            infos['teacher'].append(s.split(' ')[1] + " " + s.split(' ')[2])

        if s.split(' ')[0] == 'student':
            infos['student'].append(s.split(' ')[1] + " " + s.split(' ')[2])

        if s.split(' ')[0] == 'girl':
            infos['girl'].append(s.split(' ')[1] + " " + s.split(' ')[2])



        if s.split(' ')[0] == 'boy':
            infos['boy'].append(s.split(' ')[1] + " " + s.split(' ')[2])

        if s.split(' ')[0] == 'police':
            infos['police'].append(s.split(' ')[1] + "_" + s.split(' ')[2])  # police, consider bigrams

    print(infos)
    print('Man:', Counter(infos['man']))
    print('Woman:', Counter(infos['woman']))
    print('teacher:', Counter(infos['teacher']))
    print('Student:', Counter(infos['student']))
    print('Girl:', Counter(infos['girl']))
    print('Boy:', Counter(infos['boy']))
    print('Police:', Counter(infos['police']))

    #### create word clouds

    text_man = " ".join(infos['man']).replace('PROPN', '')  # remove PROPN
    create_word_cloud(text_man, 'man')

    text_woman = " ".join(infos['woman']).replace('PROPN', '')  # remove PROPN
    create_word_cloud(text_woman, 'woman')

    text_teacher = " ".join(infos['teacher']).replace('PROPN', '')  # remove PROPN
    create_word_cloud(text_teacher, 'teacher')

    text_student = " ".join(infos['student']).replace('PROPN', '')  # remove PROPN
    create_word_cloud(text_student, 'student')

    text_girl = " ".join(infos['girl']).replace('PROPN', '')  # remove PROPN
    create_word_cloud(text_girl, 'girl')

    text_boy = " ".join(infos['boy']).replace('PROPN', '')  # remove PROPN
    create_word_cloud(text_boy, 'boy')



    text_police = " ".join(infos['police']).replace('PROPN', '')  # remove PROPN
    print(text_police.__contains__('help'))
    print(text_police.__contains__('rescue'))
    print(text_police.__contains__('first aid'))
    print(text_police.__contains__('CPR'))
    print(text_police.__contains__('suspect'))

    create_word_cloud(text_police, 'police')

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

def explain(word):
    print(word, spacy.explain(word))

def main():
    ###### Question 1
    # q1()
    ###### Question 2
    # q2()


    ###### Question 3
    # q3()
    ####### Question4 : file md
    q4()
    # print('conj', spacy.explain('conj'))
    explain('AUX')
    explain('NP')
    explain('NNP')
    explain('PRP')
    explain('PROPN')
    explain('ADP')
    explain('AUX')
    explain('parataxis')
    explain('ccomp')
    explain('advmod')
    explain('prt')



if __name__ == '__main__':
    main()