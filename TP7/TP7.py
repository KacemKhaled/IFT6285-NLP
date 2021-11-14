import spacy
import nltk
from nltk.corpus import treebank
from nltk.tag import CRFTagger
from new_crf_module import CRFTagger_v2
from tqdm import tqdm
from nltk.tag import untag, RegexpTagger, BrillTaggerTrainer

# pip install python-crfsuite


def question4(train_data, test_data, test_gold_data):
    # — d’entraˆıner un  mod`ele transformationnel en prenant comme ´etiqueteur par d´efaut un ´etiqueteur RegexpTagger( documentation),
    # — d’entraˆıner un mod`ele transformationnel en prenant comme ´etiqueteur par d´efaut un mod `ele crf que vous avez d´evelopp´e,
    # — d’´evaluer ces mod `eles.Vous ´etudierez l’impact de m´etaparam `etres comme le nombre de r`egles ` a retenir ou les, patrons de r `egle

    # https://www.nltk.org/_modules/nltk/tag/brill.html
    # https://www.nltk.org/_modules/nltk/tag/brill_trainer.html

    regex_tagger = RegexpTagger([
        ...(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
        ...(r'(The|the|A|a|An|an)$', 'AT'),  # articles
        ...(r'.*able$', 'JJ'),  # adjectives
        ...(r'.*ness$', 'NN'),  # nouns formed from adjectives
        ...(r'.*ly$', 'RB'),  # adverbs
        ...(r'.*s$', 'NNS'),  # plural nouns
        ...(r'.*ing$', 'VBG'),  # gerunds
        ...(r'.*ed$', 'VBD'),  # past tense verbs
        ...(r'.*', 'NN')  # nouns (default)
        ])

    brill_tagger = BrillTaggerTrainer(regex_tagger,  trace=3) #todo: what is trace ?
    tagger1 = brill_tagger.train(train_data, max_rules=10)
    tagger1.evaluate(test_gold_data) #I think you don't need test, directly it tests


def question3(crt, model_file):
    # — d’entraˆıner un tagger sur les donn´ees  sur  Penn Tree Bank,
    # — de sauver ce mod `ele sur disque,
    # — d’´evaluer  ce  mod  `ele(`a l’aide  de  la fonction  evaluate).
    # https://www.nltk.org/api/nltk.tag.crf.html


    ### example:
    # crt = CRFTagger()
    # train_data = [[('University', 'Noun'), ('is', 'Verb'), ('a', 'Det'), ('good', 'Adj'), ('place', 'Noun')],
    #               [('dog', 'Noun'), ('eat', 'Verb'), ('meat', 'Noun')]]
    # crt.train(train_data, 'model.crf.tagger')
    # crt.tag_sents([['dog', 'is', 'good'], ['Cat', 'eat', 'meat']])
    # gold_sentences = [[('dog', 'Noun'), ('is', 'Verb'), ('good', 'Adj')],
    #                   [('Cat', 'Noun'), ('eat', 'Verb'), ('meat', 'Noun')]]
    # print(crt.evaluate(gold_sentences))


    #  3000 premi`eres phrases pour entraˆıner vos mod`eles, et les suivantes pour r´ealiser des tests
    print('training ... ')
    train_data = []
    for i in tqdm(range(3000)):
        train_data.append(treebank.tagged_sents()[i])
        #print(i, treebank.tagged_sents()[i])
    #print('train data', train_data)
    print('len train data', len(train_data))

    crt.train(train_data, model_file)  # the model will be saved in 'model.crf.tagger" file

    print('testing ... ')
    test_data = []
    for i in tqdm(range(3000, 3914)):
        #print(i, treebank.sents()[i] )
        test_data.append(treebank.sents()[i])
    #print('test data', test_data)
    print('len test data', len(test_data))

    crt.tag_sents(test_data) # test the data

    print('evaluating ... ')
    test_gold_data = []
    for i in tqdm(range(3000, 3914)):
     #   print(i, treebank.tagged_sents()[i])
        test_gold_data.append(treebank.tagged_sents()[i])
    #print('test gold data', test_gold_data)
    print('len golad data', len(test_gold_data))

    res = crt.evaluate(test_gold_data) # evaluate the data
    return res


def main():

    print(f'len PTB {len(treebank.tagged_sents())}')
    print( len( treebank.sents() ) )

    crt = CRFTagger()
    res1 = question3(crt, 'model.crf.tagger')
    print('res1', res1)

    print('adding context :')
    crt_v2 = CRFTagger_v2()
    res2 = question3(crt_v2, 'model_v2.crf.tagger')
    print('res2', res2)

    print(res1, res2)


if __name__ == '__main__':
    main()