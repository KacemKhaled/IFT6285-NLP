from transformers import pipeline
import spacy
import time
from tqdm import tqdm
from sacrebleu.metrics import BLEU


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
    translator = pipeline("translation_en_to_fr")
    translations = []
    total_translation_time = 0
    with open(src_data, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print('nb of sentences, ', len(lines))

        for i in tqdm(range(len(lines))):
            print('source: ', lines[i].rstrip())

            start_time = time.time()
            translation = translator(lines[i])
            finish_time = time.time() - start_time
            total_translation_time += finish_time

            print(f'translation: {translation} \n')
            print(f'time for the translation: {finish_time} \n')
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

    #print(sys)
    #print(refs)

    bleu = BLEU()
    print(bleu.corpus_score(sys, refs))
    print(bleu.get_signature())
    print(bleu.get_signature().format(short=True))
   # print(bleu.sentence_score("my name is mona", ['my name is mouna'])) #todo: confidence score


def main():
    #todo: try this on GPU
    # translate(translations_file)
    evaluate(translations_file)


if __name__ == '__main__':
    main()
