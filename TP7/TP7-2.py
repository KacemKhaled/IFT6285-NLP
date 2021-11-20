import sys
from typing import List

from nltk.corpus import treebank
import os
from nltk.tag import CRFTagger
from new_crf_module import CRFTagger_v2
from tqdm import tqdm
from time import time
import spacy
from nltk.tag import TaggerI
from nltk.tag.brill import BrillTagger, Template, brill24
from nltk.tag.brill_trainer import BrillTaggerTrainer



def train_tagger(ct,save_name = 'models/model.crf.tagger'):
    train_data = treebank.tagged_sents()[:3000]
    test_data = treebank.tagged_sents()[3000:]
    t = time()
    ct.train(train_data, save_name)
    print(f"Training time: {time()-t:.3f} s")
    print(f"{save_name} : {ct.evaluate(test_data)}")

def test_tagger(ct,model):
    test_data = treebank.tagged_sents()[3000:]
    ct.set_model_file(model)
    print(f"{model} : {ct.evaluate(test_data)}")


def test_all_CRF(model=None):
    test_data = treebank.tagged_sents()[3000:]

    # test
    if model is None:
        models = [f for f in os.listdir() if f.endswith('tagger')]
        print(models)
    else:
        models =[model]
    for model in models:
        ct = CRFTagger_v2()
        ct.set_model_file(model)
        print(f"{model} : {ct.evaluate(test_data)}")

class SpacyTagger(TaggerI):
    # model = "en_core_web_sm" # try also the _lg one
    # nlp = spacy.load(model, disable=["parser", "ner"]) # to go faster
    # # we want to do this:
    # # doc = nlp(’hello world !’)
    # #
    # # but the tokenization would change from the one in treebank
    # # which would cause problems with the function evaluate
    # # so instead do this more convoluted thing:
    # tokens_of_my_sentence = ['hello', 'world', '!']
    # doc = spacy.tokens.doc.Doc(nlp.vocab, words=tokens_of_my_sentence)
    # for _, proc in nlp.pipeline:
    #     doc = proc(doc)
    # # now doc is ready:
    # for t in doc:
    #     print(f'{t.text:20s} {t.tag_}')
    def __init__(self):
        model = "en_core_web_sm"  # try also the _lg one
        self.nlp = spacy.load(model, disable=["parser", "ner"])  # to go faster

    def tag(self, tokens):
        doc = spacy.tokens.doc.Doc(self.nlp.vocab, words=tokens)
        for _, proc in self.nlp.pipeline:
            doc = proc(doc)
        # now doc is ready:
        # for t in doc:
        #     print(f'{t.text:20s} {t.tag_}')
        return [(t.text, t.tag_) for t in doc]

def q3():
    models = {
        0: 'model_base',
        1: 'model_ap',
        -1: 'model_av',
        2: 'model_ap+av'
    }
    for key, model_name in models.items():
        ct = CRFTagger_v2(extra_features=key, context='word')
        train_tagger(ct, save_name='models/' + model_name + '_w.crf.tagger')
        test_tagger(ct, model='models/' + model_name + '_w.crf.tagger')

    for key, model_name in models.items():
        ct = CRFTagger_v2(extra_features=key, context='features')
        train_tagger(ct, save_name='models/' + model_name + '_f.crf.tagger')
        test_tagger(ct, model='models/' + model_name + '_f.crf.tagger')

def q5(b=False,it=1):
    train_data = treebank.tagged_sents()[:3000]
    test_data = treebank.tagged_sents()[3000:]
    tagger = SpacyTagger()

    print(f"Spacy: {tagger.evaluate(test_data)}")
    if b:
        start_time = time()
        recursive_tagger: BrillTagger = None

        # Clean up templates
        Template._cleartemplates()
        templates = brill24()

        for i in range(int(it)):
            if i == 0:
                trainer = BrillTaggerTrainer(SpacyTagger(), templates)
            else:
                trainer = BrillTaggerTrainer(recursive_tagger, templates)
            recursive_tagger = trainer.train(train_data)
            print(f"Iteration {i + 1}, time elapsed: {time() - start_time}")
            print(f"Train score: {recursive_tagger.evaluate(train_data)}")
            print(f"Test score: {recursive_tagger.evaluate(test_data)}")


def main():
    train_data = treebank.tagged_sents()[:3000]
    test_data = treebank.tagged_sents()[3000:]
    print(f'len PTB {len(treebank.tagged_sents())}')
    print(len(treebank.sents()))
    q5(True,10)



if __name__ == '__main__':
    main()