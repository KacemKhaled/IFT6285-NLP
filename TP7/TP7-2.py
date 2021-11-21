import sys
from typing import List

from nltk.corpus import treebank
import os
from nltk.tag import CRFTagger
from new_crf_module import CRFTagger_v2
from tqdm import tqdm
from time import time
import spacy
from nltk.tag import TaggerI,RegexpTagger,BrillTaggerTrainer
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


def train_recursive_tagger(tagger,base,templates,iterations,start_time):
    train_data = treebank.tagged_sents()[:3000]
    test_data = treebank.tagged_sents()[3000:]
    if iterations > 1:
        # print(f"base = rec(tagger={tagger},base{base},templates,iterations-1,start_time)  ")
        base = train_recursive_tagger(tagger,base,templates,iterations-1,start_time)
    # trainer = tagger(base_tagger, templates)
    result_tagger = tagger(base, templates).train(train_data)
    print(f"Iteration {iterations}, time elapsed: {time() - start_time}")
    print(f"Test score: {result_tagger.evaluate(test_data)}")
    # print(f" result_tagger :  {result_tagger}  ")
    return result_tagger


def q4_1():
    train_data = treebank.tagged_sents()[:3000]
    test_data = treebank.tagged_sents()[3000:]
    # Template._cleartemplates()
    templates = brill24()
    crf_tagger = CRFTagger()
    crf_tagger.set_model_file('models/model_base_w.crf.tagger')
    regex_tagger = RegexpTagger([
        (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
        (r'(The|the|A|a|An|an)$', 'AT'),  # articles
        (r'.*able$', 'JJ'),  # adjectives
        (r'.*ness$', 'NN'),  # nouns formed from adjectives
        (r'.*ly$', 'RB'),  # adverbs
        (r'.*s$', 'NNS'),  # plural nouns
        (r'.*ing$', 'VBG'),  # gerunds
        (r'.*ed$', 'VBD'),  # past tense verbs
        (r'.*', 'NN')  # nouns (default)
    ])
    print(f"regex_tagger: {regex_tagger.evaluate(test_data)}")
    train_recursive_tagger(BrillTaggerTrainer, regex_tagger,templates, iterations=1,start_time=time())
    print(f"crf_tagger: {crf_tagger.evaluate(test_data)}")
    train_recursive_tagger(BrillTaggerTrainer, crf_tagger,templates, iterations=1,start_time=time())

def q4_2(iterations=1):
    train_data = treebank.tagged_sents()[:3000]
    test_data = treebank.tagged_sents()[3000:]
    # Template._cleartemplates()
    templates = brill24()
    crf_tagger = CRFTagger()
    crf_tagger.set_model_file('models/model_base_w.crf.tagger')
    print(f"crf_tagger: {crf_tagger.evaluate(test_data)}")
    train_recursive_tagger(BrillTaggerTrainer, crf_tagger,templates, iterations,start_time=time())

def q5_2(iterations=1):
    train_data = treebank.tagged_sents()[:3000]
    test_data = treebank.tagged_sents()[3000:]
    base_tagger = SpacyTagger()
    Template._cleartemplates()
    templates = brill24()

    print(f"Spacy: {base_tagger.evaluate(test_data)}")
    train_recursive_tagger(BrillTaggerTrainer, base_tagger, templates, iterations,start_time=time())

def main():
    train_data = treebank.tagged_sents()[:3000]
    test_data = treebank.tagged_sents()[3000:]
    print(f'len PTB {len(treebank.tagged_sents())}')
    print(len(treebank.sents()))
    q4_1() # converges at 2 acc test
    # q4_2(iterations=4) # converges at 2 acc test
    # q5(iterations=5) # converges at 3 acc test
    # q5_2(iterations=4) # converges at 3 acc test



if __name__ == '__main__':
    main()