import nltk
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import treebank
import os
from nltk.tag import CRFTagger
from new_crf_module import CRFTagger_v2
from tqdm import tqdm
from time import time
import spacy
from nltk.tag import TaggerI,RegexpTagger,BrillTaggerTrainer, untag
from nltk.tag.brill import BrillTagger, Template, brill24
from nltk.tag.brill_trainer import BrillTaggerTrainer


# pip install python-crfsuite

def question4_min_accuracies(train_data, test_data, test_gold_data, min_accuracies_list):

    res_regex_min_acc = []
    res_crf_min_acc = []
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
    templates = nltk.tag.brill.brill24()
    crt_tagger = CRFTagger()
    crt_tagger.set_model_file('model.crf.tagger')

    for min_acc in min_accuracies_list:

        brill_tagger1 = BrillTaggerTrainer(regex_tagger, templates, trace=1)
        tagger1 = brill_tagger1.train(train_data, min_acc=min_acc)
        res_regex = tagger1.evaluate( test_gold_data)
        res_regex_min_acc.append(res_regex)

        brill_tagger2 = BrillTaggerTrainer(crt_tagger, templates, trace=1)
        tagger2 = brill_tagger2.train(train_data, min_acc= min_acc)
        res_crf = tagger2.evaluate(test_gold_data)
        res_crf_min_acc.append(res_crf)

    return res_crf_min_acc, res_regex_min_acc


def question4_templates(train_data, test_data, test_gold_data, tem_categories_list):
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
    crt_tagger = CRFTagger()
    crt_tagger.set_model_file('model.crf.tagger')

    res_regex_tmp = []
    res_crf_tmp = []

    for temp in tem_categories_list:
        brill_tagger1 = BrillTaggerTrainer(regex_tagger, temp, trace=1)
        tagger1 = brill_tagger1.train(train_data)
        res_regex = tagger1.evaluate( test_gold_data)
        res_regex_tmp.append(res_regex)

        brill_tagger2 = BrillTaggerTrainer(crt_tagger, temp, trace=1)
        tagger2 = brill_tagger2.train(train_data)
        res_crf = tagger2.evaluate(test_gold_data)
        res_crf_tmp.append(res_crf)

    return res_crf_tmp, res_regex_tmp


def plot_templates(res_crf_tmp, res_regex_tmp,  tem_categories_names, name_fig):

    X_axis = np.arange(len(tem_categories_names))

    plt.figure(figsize=(6,3))
    plt.bar(X_axis -0.1 , res_crf_tmp,  0.2, label='CRF Tagger')
    plt.bar(X_axis +0.1 , res_regex_tmp, 0.2,  label='Regex tagger')

    plt.xticks(X_axis, tem_categories_names, rotation=5)

    for i, j in zip(X_axis, res_regex_tmp):
        plt.text(i, j+0.005 , f'{j:.4f}',color='C1')
    for i, j in zip(X_axis, res_crf_tmp):
        plt.text(i-0.2, j+0.001 , f'{j:.4f}',color='C0')

    plt.xlabel("Templates utilisees")
    plt.ylabel("Evaluation")
    plt.title("Impact du filtrage des templates sur la performance des modeles.")
    plt.legend(fancybox=True, framealpha=0.3,loc='lower left')
    plt.savefig(f"plots/" + name_fig + ".svg", format="svg")
    plt.savefig(f"plots/" + name_fig + ".png", format="png")
    plt.savefig(f"plots/" + name_fig + ".eps", format="eps",transparent=True)


def question4_nb_rules(train_data, test_data, test_gold_data, nb_max_rules_list):
    # — d’entraˆıner un  mod`ele transformationnel en prenant comme ´etiqueteur par d´efaut un ´etiqueteur RegexpTagger( documentation),
    # — d’entraˆıner un mod`ele transformationnel en prenant comme ´etiqueteur par d´efaut un mod `ele crf que vous avez d´evelopp´e,
    # — d’´evaluer ces mod `eles.Vous ´etudierez l’impact de m´etaparam `etres comme le nombre de r`egles ` a retenir ou les, patrons de r `egle

    # https://www.nltk.org/_modules/nltk/tag/brill.html
    # https://www.nltk.org/_modules/nltk/tag/brill_trainer.html

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
    templates = nltk.tag.brill.brill24()
    crt_tagger = CRFTagger()
    crt_tagger.set_model_file('model.crf.tagger')

    nb_max_rules = nb_max_rules_list
    res_crf_nb_rules = []
    res_regex_nb_rules = []

    for max_rules in nb_max_rules:

        brill_tagger1 = BrillTaggerTrainer(regex_tagger, templates,  trace=1)
        tagger1 = brill_tagger1.train(train_data, max_rules=max_rules)
        res_regex = tagger1.evaluate(test_gold_data) #I think you don't need test, directly it tests and evaluate gold data
        res_regex_nb_rules.append(res_regex)

        brill_tagger2 = BrillTaggerTrainer(crt_tagger, templates, trace=1)
        tagger2 = brill_tagger2.train(train_data, max_rules=max_rules)
        res_crf = tagger2.evaluate(test_gold_data)
        res_crf_nb_rules.append(res_crf)

    return res_crf_nb_rules, res_regex_nb_rules


def plot(x, y_crf, y_regex, name_fig, title, x_label, fig_size,offset=0.002):
    print('Creating the figure')

    plt.figure(figsize=fig_size)
    plt.plot(x , y_crf, label='CRF Tagger',marker='.')
    plt.plot(x,  y_regex, label='Regex Tagger',marker='.')
    plt.xticks(x,labels=x[:-1]+[int(x[-1])])
    plt.yticks(color='green')
    # plt.yticks(y_crf+y_regex)
    for i, j in zip(x, y_crf):
        plt.text(i, j + offset, f'{j:.4f}',color='green')
        if offset == 0.01:
            offset = -0.03
        elif offset == -0.03:
            offset = 0.01
        elif offset == 0.002:
            offset = -0.009
        elif offset == -0.009:
            offset = 0.002

    for i, j in zip(x, y_regex):
        plt.text(i, j + offset, f'{j:.4f}',color='green')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('Evaluation')
    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)

    plt.savefig(f"plots/"+name_fig+".svg", format="svg")
    plt.savefig(f"plots/"+name_fig+".png", format="png")
    plt.savefig(f"plots/"+name_fig+".eps", format="eps",transparent=True)


def q4_analyses_metaparametres(params = ['rules','accuracies','templates','analyse_rules']):
    train_data = treebank.tagged_sents()[:3000]
    test_data = treebank.sents()[3000:3914]
    test_gold_data = treebank.tagged_sents()[3000:3914]
    ############################  QUESTION 4

    if 'rules' in params:
                # impact of nb of rules

        nb_max_rules = [0, 10, 50, 100, 150, 200, 250, 300]
        res_crf_nb_rules, res_regex_nb_rules = question4_nb_rules(train_data, test_data, test_gold_data, nb_max_rules)
        print('res regex', res_regex_nb_rules)
        print('res crf', res_crf_nb_rules)
        # res regex [0.26324195985322685, 0.5313619684869415, 0.6762357004101014, 0.7275199654651414, 0.7589035182387223, 0.7788905676667386]
        # res crf [0.9474638463198791, 0.9480682063457803, 0.9488884092380747, 0.9497086121303691, 0.9502266350097129, 0.9501834664364343]
        #res_crf_nb_rules = [0.9474638463198791, 0.9480682063457803, 0.9488884092380747, 0.9497086121303691, 0.9502266350097129, 0.9501834664364343]
        #res_regex_nb_rules = [0.26324195985322685, 0.5313619684869415, 0.6762357004101014, 0.7275199654651414, 0.7589035182387223, 0.7788905676667386]
        plot(nb_max_rules, res_crf_nb_rules, res_regex_nb_rules, name_fig='max_rules', title='Impact du nombre de règles a retenir \n sur la performance des modèles.', x_label='Nombre des regles maximales a retenir', fig_size=(5, 4))

            # impact of accuracies
    if 'accuracies' in params:
        min_accuracies_list = [0.8 , 0.85, 0.9, 0.95, 0.99, 1]
        res_crf_min_acc, res_regex_min_acc = question4_min_accuracies(train_data, test_data, test_gold_data, min_accuracies_list)
        print('res regex min acc ', res_regex_min_acc)
        print('res crf min acc ', res_crf_min_acc)
        # res regex min acc  [0.8039715087416361, 0.8039715087416361, 0.8010360457586877, 0.7972372113101662, 0.793524714008202, 0.7660263328297]
        # res crf min acc  [0.9501834664364343, 0.9493200949708612, 0.9493200949708612, 0.9493200949708612, 0.9493200949708612, 0.9493200949708612]
        #res_crf_min_acc = [0.9501834664364343, 0.9493200949708612, 0.9493200949708612, 0.9493200949708612, 0.9493200949708612, 0.9493200949708612]
        #res_regex_min_acc = [0.8039715087416361, 0.8039715087416361, 0.8010360457586877, 0.7972372113101662, 0.793524714008202, 0.7660263328297]
        plot(min_accuracies_list, y_crf=res_crf_min_acc, y_regex=res_regex_min_acc, name_fig='min_accu',
             title='Impact du accuracy minimale sur la \n performance des modèles.',
              x_label='Accuracy minimale', fig_size = (6, 4))

            # impact of templates

    if 'templates' in params:

        templates = nltk.tag.brill.brill24()
        templates_pos_only = templates[:11]
        templates_word_only = templates[11:20]
        templates_word_pos = templates[20:]
        tem_categories = [templates, templates_pos_only, templates_word_only, templates_word_pos]
        res_crf_tmp, res_regex_tmp = question4_templates(train_data, test_data, test_gold_data, tem_categories)
        print('res crf temp ', res_crf_tmp)
        print('res regex temp ', res_regex_tmp)
        #res cr temp[0.9501834664364343, 0.9476365206129937, 0.9498812864234837, 0.9504424778761061]
        #res regex temp[0.7788905676667386, 0.44131232462767106, 0.7929635225555796, 0.5038635873084395]
        #res_regex_tmp = [0.7788905676667386, 0.44131232462767106, 0.7929635225555796, 0.5038635873084395]
        #res_crf_tmp = [0.9501834664364343, 0.9476365206129937, 0.9498812864234837, 0.9504424778761061]
        tem_categories_names = ['All templates', 'POS-only templates', 'Word-only templates', 'Word-Pos templates']
        plot_templates(res_crf_tmp, res_regex_tmp,  tem_categories_names, 'templates_cat' )


                # analyse des types des regles apprises:
    if 'analyse_rules' in params:
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
        templates = nltk.tag.brill.brill24()
        crt_tagger = CRFTagger()
        crt_tagger.set_model_file('model.crf.tagger')

        brill_tagger1 = BrillTaggerTrainer(regex_tagger, templates, trace=3)
        tagger1 = brill_tagger1.train(train_data)
        print('Rules')
        print( tagger1.rules() )
        print('Train stats')
        print( tagger1.train_stats())
        print('Train stats: rules scores')
        print(tagger1.train_stats()['rulescores'])
        print('Templatest Stats')
        print(tagger1.print_template_statistics())

        brill_tagger2 = BrillTaggerTrainer(crt_tagger, templates, trace=3)
        tagger2 = brill_tagger2.train(train_data)
        print('Rules')
        print(tagger2.rules())
        print('Train stats')
        print(tagger2.train_stats())
        print('Train stats: rules scores')
        print(tagger2.train_stats()['rulescores'])
        print('Templatest Stats')
        print(tagger2.print_template_statistics())



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
        model = "en_core_web_lg"  # try also the _lg one
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
    templates = brill24()
    crf_tagger = CRFTagger()
    crf_tagger.set_model_file('models/model_base_w.crf.tagger')
    print(f"crf_tagger: {crf_tagger.evaluate(test_data)}")
    train_recursive_tagger(BrillTaggerTrainer, crf_tagger,templates, iterations,start_time=time())

def q5_2(iterations=1):
    train_data = treebank.tagged_sents()[:3000]
    test_data = treebank.tagged_sents()[3000:]
    base_tagger = SpacyTagger()
    # Template._cleartemplates()
    templates = brill24()

    print(f"Spacy: {base_tagger.evaluate(test_data)}")
    train_recursive_tagger(BrillTaggerTrainer, base_tagger, templates, iterations,start_time=time())

def quick_plots():
    nb_max_rules = [0, 10, 50, 100, 150, 200, 250, 300]
    res_crf_nb_rules = [0.9474638463198791, 0.9480682063457803, 0.9488884092380747, 0.9497086121303691, 0.9502266350097129, 0.9501834664364343, 0.9503129721562702, 0.9503993093028276]
    res_regex_nb_rules = [0.26324195985322685, 0.5313619684869415, 0.6762357004101014, 0.7275199654651414, 0.7589035182387223, 0.7788905676667386, 0.7913231167709908, 0.8011655514785236]
    plot(nb_max_rules, res_crf_nb_rules, res_regex_nb_rules, name_fig='max_rules',
         title='Impact du nombre de règles a retenir \n sur la performance des modèles.',
         x_label='Nombre des regles maximales a retenir', fig_size=(5, 4),offset=0.01)

    min_accuracies_list = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
    res_crf_min_acc = [0.9501834664364343, 0.9493200949708612, 0.9493200949708612, 0.9493200949708612, 0.9493200949708612, 0.9493200949708612]
    res_regex_min_acc = [0.8039715087416361, 0.8039715087416361, 0.8010360457586877, 0.7972372113101662, 0.793524714008202, 0.7660263328297]
    plot(min_accuracies_list, y_crf=res_crf_min_acc, y_regex=res_regex_min_acc, name_fig='min_accu',
         title='Impact du accuracy minimale sur la \n performance des modèles.',
          x_label='Accuracy minimale', fig_size = (6, 4))

    res_regex_tmp = [0.7788905676667386, 0.44131232462767106, 0.7929635225555796, 0.5038635873084395]
    res_crf_tmp = [0.9501834664364343, 0.9476365206129937, 0.9498812864234837, 0.9504424778761061]
    tem_categories_names = ['All templates', 'POS-only templates', 'Word-only templates', 'Word-Pos templates']
    plot_templates(res_crf_tmp, res_regex_tmp, tem_categories_names, 'templates_cat')

def main():

    train_data = treebank.tagged_sents()[:3000]
    test_data = treebank.tagged_sents()[3000:]
    print(f'len PTB {len(treebank.tagged_sents())}')
    print(len(treebank.sents()))
    q3()
    q4_1() # converges at 2 acc test
    q4_analyses_metaparametres()
    q4_2(iterations=4) # converges at 2 acc test
    q5_2(iterations=4) # converges at 3 acc test

    quick_plots()



if __name__ == '__main__':
    main()