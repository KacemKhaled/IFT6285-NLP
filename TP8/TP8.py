from transformers import pipeline
import spacy
import time
from tqdm import tqdm
from sacrebleu.metrics import BLEU
import matplotlib.pyplot as plt
import numpy as np
import math

from statistics import mean
import plotly.express as px
import pandas as pd


# pip install transformers
# pip install sacrebleu

# sacrebleu -t wmt14 -l en-fr --echo src
# sacrebleu -t wmt14 -l en-fr --echo ref

src_data = 'wmt14/wmt14.en'
tgt_data = 'wmt14/wmt14.fr'
translations_file = 'translations.txt'


def translate(file):
    # utiliser transformers ( un code similaire a https://huggingface.co/transformers/task_summary.html#translation
    # pour traduire EN---> FR au moins 1000 phrases du corpus WMT.
    translator = pipeline("translation_en_to_fr",device=0)
    translations = []
    total_translation_time = 0
    with open(src_data, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print('nb of sentences, ', len(lines))

        for i in tqdm(range(len(lines))):
            # print('source: ', lines[i].rstrip())

            start_time = time.time()
            translation = translator(lines[i])
            finish_time = time.time() - start_time
            total_translation_time += finish_time

            # print(f'translation: {translation} \n')
            # print(f'time for the translation: {finish_time} \n')
            translations.extend(translation)

        assert len(translations) == len(lines)

    print(f'total translation time is {total_translation_time}  seconds for {len(lines)} sentences')
    # print(translations)
    translations_text = [t['translation_text'] for t in translations]
    # print(translations_text)

    with open(file, 'w',encoding='utf-8') as f:
        for tr in translations_text:
            f.write('%s\n' % tr)


def evaluate(file):
    # # Vous evaluerez avec  BLEU -- > use library scra bleu
    # refs = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.' ,'My name is Mouna']] # list of a list
    # sys = ['The dog bit the man.'  , "It wasn't surprising.", 'The man had just bitten him.']
    refs, sys = get_data_for_eval(file)

    bleu = BLEU()
    print(bleu.corpus_score(sys, refs))
    print(bleu.get_signature())
    print(bleu.get_signature().format(short=True))
   # print(bleu.sentence_score("my name is mona", ['my name is mouna'])) #todo: confidence score


def get_data_for_eval(file):
    # read the translations file into a list
    sys = []
    with open(file, "r",encoding='utf-8') as f:
        translations = f.readlines()
        assert len(translations) == 3003
        sys = [t.rstrip() for t in translations]

    # read the reference file into a list of a list
    refs = []
    with open(tgt_data, "r", encoding='utf-8') as f:
        fr_sentences = f.readlines()
        assert len(fr_sentences) == 3003
        refs.append([s.rstrip() for s in fr_sentences])

    # print(sys[:2])
    # print(refs[0][:2])

    return refs, sys


def variability_nb_sentences(refs, sys, min_len_sentences=None,max_len_sentences=None):
    """
    sacrebleu -t wmt14 -l en-fr -i translations.txt -m bleu --confidence -f json --short
    """
    # nb_phrases = [20,50,100,200,300] + list(range(400, 3001,200)) #[500, 1000, 1500, 2000, 2500, 3000]
    nb_phrases = list(range(200, 3001, 200))
    blue_scores = []
    print(nb_phrases)

    bleu = BLEU()
    fig, ax1 = plt.subplots(figsize=(6, 5))
    if min_len_sentences or max_len_sentences:
        print(f"min_len_sentences: {min_len_sentences}")
        print(f"max_len_sentences: {max_len_sentences}")
        idxs = []
        for i,sent in enumerate(sys):
            if min_len_sentences <= len(sent.split(" ")) <= max_len_sentences:
                idxs.append(i)

        print(idxs)
        print(f"new num sentences: {len(idxs)}")
        print(f"new avg len sentences: {average_lengths(list(map(sys.__getitem__, idxs))):.2f}")
        nb_phrases = list(range( len(idxs)//5,len(idxs), len(idxs)//5))
        ax1.set_xticks(nb_phrases[::])
        ax1.set_ylim(37,42)
    else:
        idxs = None
        ax1.set_xticks(nb_phrases[::2])
        ax1.set_ylim(37,39)

    for nb in nb_phrases:
        print(f"----------{nb}-------------")
        # idxs = get_index(sys)
        if idxs is None:
            idxs = list(range(len(sys)))
            print(len(idxs))

        if len(idxs) < nb:
            exit("Not enough sentences")
        sys_mapping = list(map(sys.__getitem__, idxs[:nb]))
        refs_mapping = list(map(refs[0].__getitem__, idxs[:nb]))
        s = bleu.corpus_score(sys_mapping, [refs_mapping])

        # s = bleu.corpus_score(sys[:nb], [refs[0][:nb]])
        #print(s)
        #print(s.score)
        blue_scores.append(s.score)
    width = 100
    # fig.tight_layout()
    # plt.figure(figsize=(6,5))
    color = 'C0'
    ax1.bar(nb_phrases, blue_scores,width)
    # plt.minorticks_on()
    ax1.set_xlabel('Nombre des phrases utilisées')
    ax1.set_ylabel('Les scores Bleu',color=color)

    for i, j in zip(nb_phrases, blue_scores):
        plt.text(i-width, j+0.01 , f'{j:.1f}',color='C0')
    # print(bleu.get_signature().format(short=True))

    ax2 = ax1.twinx()
    color = 'C1'
    list1, list2 = zip(*sorted(zip(nb_phrases, [average_lengths(list(map(sys.__getitem__, idxs[:nb])) )for nb in nb_phrases])))
    ax2.plot(np.array(list1), list2,'--o' ,color=color,alpha=1)
    ax2.tick_params(axis='y', labelcolor=color,colors=color,which='both')
    ax2.set_ylabel('La longueur moyenne de phrases par groupe',color=color)

    print(list(zip(nb_phrases,blue_scores)))
    plt.title("Impact du nombre de phrases utilisées sur les scores Bleu")
    # plt.show()
    suffix = f"_len_sent_{min_len_sentences}_{max_len_sentences}" if min_len_sentences or max_len_sentences else ""
    plt.savefig(f'plots/nb_sentences_impact{suffix}.png')
    plt.savefig(f'plots/nb_sentences_impact{suffix}.eps')
    plt.savefig(f'plots/nb_sentences_impact{suffix}.pdf')

def average_lengths(l):
    return mean([len(x.split(' ')) for x in l])

def variability_len_sentences(refs, sys,fix_nb_sentences=None):
    """
    sacrebleu -t wmt14 -l en-fr -i translations.txt -m bleu --confidence -f json --short
    """
    blue_scores = []

    bleu = BLEU()
    idxs_ceiled = {}
    idxs = {}
    for i, sent in enumerate(sys):
        len_sent = len(sent.split(' '))
        ceiled = 5 * math.ceil(len_sent/5)
        # print(ceiled)
        if ceiled not in idxs_ceiled:
            idxs_ceiled[ceiled] = [i]
        else:
            idxs_ceiled[ceiled] += [i]
        # if len_sent not in idxs:
        #     idxs[len_sent] = [i]
        # else:
        #     idxs[len_sent] += [i]

    print(f"num sentences: {len(sum(idxs_ceiled.values(), []))}")
    idx = sum(idxs_ceiled.values(), [])
    print(f"avg len sentences: {average_lengths(list(map(sys.__getitem__, idx))):.2f}")
    if fix_nb_sentences is not None:
        print(f"fix_nb_sentences: {fix_nb_sentences}")
        d = {}
        for sent_len,idx in idxs_ceiled.items():
            if len(idx) >= fix_nb_sentences:
                d[sent_len] = idx[:fix_nb_sentences]
        idxs_ceiled = d
        # print(idxs)
        print(f"new num sentences: {len(sum(idxs_ceiled.values(), []))}")
        idx = sum(idxs_ceiled.values(), [])
        print(f"new avg len sentences: {average_lengths(list(map(sys.__getitem__, idx))):.2f}")
    for sent_len,idx in idxs_ceiled.items():
        # print(sent_len, idx)
        print(f"----------{sent_len}-------------")
        sys_mapping = list(map(sys.__getitem__, idx))
        refs_mapping = list(map(refs[0].__getitem__, idx))
        s = bleu.corpus_score(sys_mapping, [refs_mapping])
        print(sent_len, s)
        # print(s.score)
        blue_scores.append(s.score)

    fig, ax1 = plt.subplots(figsize=(8,5))
    width = 5  # the width of the bars
    # plt.figure(figsize=(5,5))
    color = 'C0'
    x_axis = np.array(list(idxs_ceiled.keys()))-width/2
    ax1.bar(x_axis, blue_scores,width, color=color)
    plt.ylim(22,42)
    ax1.set_xticks(sorted([0]+list(idxs_ceiled.keys())))
    ax1.set_xlabel('Longueur des phrases utilisées (nb. mots)')
    ax1.set_ylabel('Les scores Bleu', color=color)
    for i, j in zip(x_axis, blue_scores):
        plt.text(i-width/2, j+0.1 , f'{j:.1f}',color='C0')
    ax2 = ax1.twinx()
    color = 'C1'
    list1, list2 = zip(*sorted(zip(idxs_ceiled.keys(), [len(idx) for idx in idxs_ceiled.values()])))
    ax2.plot(np.array(list1) - width/2, list2,'--o' ,color=color,alpha=1)
    ax2.tick_params(axis='y', labelcolor=color,colors=color,which='both')
    ax2.set_ylabel('Le nombre de phrases par groupe',color=color)

    fig.tight_layout()
    # plt.show()
    plt.grid()
    plt.title("Impact des longueurs de phrases utilisées sur les scores Bleu")
    suffix = f"_fixed_nb_sent_{fix_nb_sentences}" if fix_nb_sentences else ""
    plt.savefig(f'plots/len_sentences_impact{suffix}.png')
    plt.savefig(f'plots/len_sentences_impact{suffix}.eps')
    plt.savefig(f'plots/len_sentences_impact{suffix}.pdf')

def plot_bars_variability():
    groups = ['c:lc','c:mixed']
    parametres = ['tok:zh','tok:13a',]
    blue_scores = []
    df = pd.DataFrame([ # mixed: sensitive to case, lowercase: insensitive
        ('c:mixed','tok:none',34.2),
        ('c:mixed','tok:zh',37.8),
        ('c:mixed','tok:13a',38.1),
        ('c:mixed','tok:char',69.4),
        ('c:mixed','tok:intl',40.9),
        ('c:mixed','tok:ja-mecab', 41.0),
        ('c:lc','tok:none',35.1),
        ('c:lc','tok:zh',38.7 ),
        ('c:lc','tok:13a',39.0),
        ('c:lc','tok:char', 69.8),
        ('c:lc','tok:intl',41.9),
        ('c:lc','tok:ja-mecab',41.9),
    ], columns=['Casse','Tokenisation','Score Bleu'])

    fig = px.bar(df, y='Score Bleu', x='Tokenisation',color='Casse',
                 barmode="group", text='Score Bleu',
                 width=700, height=400)
    fig.update_yaxes(range=[30, 75])
    fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_layout(
    title={
        'text': "Impact de tokenisation et la sensibilité à la casse sur le score Bleu",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    fig.write_image("plots/other_variability_impact.pdf")
    fig.show()

    # plt.savefig('plots/other_variability_impact.png')
    # plt.savefig('plots/other_variability_impact.eps')
    # plt.savefig('plots/other_variability_impact.pdf')


def make_image_distribution(scores_list):
    df = pd.DataFrame(scores_list, columns=['bleu score'])

    fig = px.violin(df, x='bleu score' ,box=True,violinmode='overlay',points='all',  orientation='h') #opacity=1,#barmode='group',facet_col='set',
    fig.update_xaxes(title_text="Les scores bleu",row=1,col=1)
    title={     'text':'Distribution des scores BLEU par phrase' ,
               'x':0.5,
                'xanchor': 'center'
            }
    fig.update_layout( title=title, height=400, width=800 )
    #fig.show()
    # fig.write_image("plots/distribution_des_scores_par_phrases.pdf")
    # fig.write_image("plots/distribution_des_scores_par_phrases.png")
    # fig.write_image('plots/distribution_des_scores_par_phrases.eps')


def q5_6_sentence_level_bleu(file):
    sentence_bleu_scores = []
    bad_translations = []

    lines = []
    with open(src_data, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    refs, sys = get_data_for_eval(file)
    references = refs[0]
    bleu = BLEU(effective_order=True)
    for i, s in enumerate(sys):
        bleu_score = bleu.sentence_score(s, [references[i]])

        if bleu_score.score == 0 : # question 6
            bad_translations.append( (bleu_score.score,lines[i].rstrip(), s, [references[i]] ) )

        sentence_bleu_scores.append(bleu_score.score)

    make_image_distribution(sentence_bleu_scores)
    print(len(sentence_bleu_scores))
    print(f'min = {min(sentence_bleu_scores)}')
    print(f'max = {max(sentence_bleu_scores)}')
    print(f'moy = {mean(sentence_bleu_scores)}')

    print(f'len of bad translation {len(bad_translations)}')
    print('bad traductions. bleu == 0', )
    # for i in range(len(bad_translations)):
    #     print(bad_translations[i])
    for sent in sorted(bad_translations):
        print(sent)



def test_q6():
    bleu = BLEU(effective_order=True)
    bleu_score = bleu.sentence_score("États-Unis habillés pour l'Halloween", ["Les États-Unis aux couleurs d'Halloween"])
    print(bleu_score)
    print(bleu.get_signature())
    print(bleu_score.score)

    # sent = "China plea paper 'to be overhauled'"
    # translator = pipeline("translation_en_to_fr")
    #
    # translation = translator(sent)
    # print(translation)
    #
    # bleu = BLEU()
    # bleu_score = bleu.sentence_score(sent, [translation[0]['translation_text']])
    # print(bleu.get_signature())
    # print(bleu_score.score)


def main():
    # translate(translations_file)
    # evaluate(translations_file)

    # refs, sys = get_data_for_eval(translations_file)
    # variability_nb_sentences(refs, sys)
    # variability_nb_sentences(refs, sys,20,25)
    # variability_len_sentences(refs, sys)
    # variability_len_sentences(refs, sys,200)
    # plot_bars_variability()

    q5_6_sentence_level_bleu(translations_file)
    # test_q6()

if __name__ == '__main__':
    main()
