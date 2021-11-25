from transformers import pipeline
import spacy
import time
from tqdm import tqdm

# pip install transformers
# pip install sacrebleu

# sacrebleu -t wmt14 -l en-fr --echo src
# sacrebleu -t wmt14 -l en-fr --echo ref

src_data = 'wmt14/wmt14.en'
tgt_data = 'wmt14/wmt14.fr'
translations_file = 'translations.txt'


def main():
    #todo: try this on GPU
   # utiliser transformers ( un code similaire a https://huggingface.co/transformers/task_summary.html#translation
   # pour traduire EN---> FR au moins 1000 phrases du corpus WMT.
    # It leverages a T5 - base model that was only pre-trained on a multi-task mixture dataset (including WMT),

    translator = pipeline("translation_en_to_fr")
    translations = []
    total_translation_time = 0
    with open(src_data, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print('nb of sentences, ', len(lines))

        for i in tqdm(range(len(lines))):
            print('source: ',lines[i].rstrip())

            start_time = time.time()
            translation = translator(lines[i])
            finish_time = time.time() - start_time
            total_translation_time+=finish_time

            print(f'translation: {translation} \n')
            print(f'time for the translation: {finish_time} \n')
            translations.extend(translation)

        assert len(translations) == len(lines)

    print(f'total translation time is {total_translation_time}  seconds for {len(lines)} sentences')
    #print(translations)
    translations_text = [ t['translation_text'] for t in translations]
    #print(translations_text)

    with open(translations_file, 'w') as f:
        for tr in translations_text:
            f.write('%s\n' % tr)

   # Vous evaluerez avec  BLEU -- > use library scra bleu
    pass


if __name__ == '__main__':
    main()