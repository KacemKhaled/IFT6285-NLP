from transformers import pipeline
import spacy
import time
from tqdm import tqdm
from sacrebleu.metrics import BLEU
import matplotlib.pyplot as plt
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

    with open(file, 'w') as f:
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
    with open(file, "r") as f:
        translations = f.readlines()
        assert len(translations) == 3003
        sys = [t.rstrip() for t in translations]

    # read the reference file into a list of a list
    refs = []
    with open(tgt_data, "r", encoding='utf-8') as f:
        fr_sentences = f.readlines()
        assert len(fr_sentences) == 3003
        refs.append([s.rstrip() for s in fr_sentences])

    # print(sys)
    # print(refs)

    return refs, sys


def variability(refs, sys):
    nb_phrases = [500, 1000, 1500, 2000, 2500, 3000]
    blue_scores = []

    bleu = BLEU()

    for nb in nb_phrases:
        s = bleu.corpus_score(sys[:nb], [refs[0][:nb]])
        #print(s)
        #print(s.score)
        blue_scores.append(s.score)
        print(bleu.get_signature().format(short=True))

    plt.figure(figsize=(5,5))
    plt.plot( nb_phrases, blue_scores, marker='.')
    plt.xlabel('Nombre des phrases utilisées')
    plt.ylabel('Les scores Bleu')
    plt.title("Impact du nombre de phrases utilisées \n sur les scores Bleu")
    #plt.show()
    plt.savefig('plots/nb_sentences_impact.png')
    plt.savefig('plots/nb_sentences_impact.eps')
    plt.savefig('plots/nb_sentences_impact.pdf')


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
    fig.write_image("plots/distribution_des_scores_par_phrases.pdf")
    fig.write_image("plots/distribution_des_scores_par_phrases.png")
    fig.write_image('plots/distribution_des_scores_par_phrases.eps')


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
            bad_translations.append( (lines[i].rstrip(), s, [references[i]] ) )

        sentence_bleu_scores.append(bleu_score.score)

    make_image_distribution(sentence_bleu_scores)
    print(len(sentence_bleu_scores))
    print(f'min = {min(sentence_bleu_scores)}')
    print(f'max = {max(sentence_bleu_scores)}')
    print(f'moy = {mean(sentence_bleu_scores)}')

    print(f'len of bad translation {len(bad_translations)}')
    print('bad traductions. bleu == 0', )
    for i in range(len(bad_translations)):
        print(bad_translations[i])


def main():
    #translate(translations_file) #todo: try this on GPU
    #evaluate(translations_file)

    # refs, sys = get_data_for_eval(translations_file)
    # variability(refs, sys)

    q5_6_sentence_level_bleu(translations_file)


if __name__ == '__main__':
    main()
