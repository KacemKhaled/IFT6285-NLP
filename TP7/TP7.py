import spacy
import nltk
from nltk.corpus import treebank
from nltk.tag import CRFTagger
from new_crf_module import CRFTagger_v2
from new_crf_module_v3 import CRFTagger_v3
from tqdm import tqdm
from nltk.tag import untag, RegexpTagger, BrillTaggerTrainer
import matplotlib.pyplot as plt
import numpy as np

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
    plt.bar(X_axis +0.1 , res_regex_tmp, 0.2,  label='Regex tagger')
    plt.bar(X_axis -0.1 , res_crf_tmp,  0.2, label='CRF Tagger')

    plt.xticks(X_axis, tem_categories_names, rotation=5)
    plt.xlabel("Templates utilisees")
    plt.ylabel("Evaluation")
    plt.title("Impact du filtrage des templates sur la performance des modeles.")
    plt.legend()
    plt.savefig(f"plots/" + name_fig + ".svg", format="svg")
    plt.savefig(f"plots/" + name_fig + ".png", format="png")
    plt.savefig(f"plots/" + name_fig + ".eps", format="eps")


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


def plot(x, y_crf, y_regex, name_fig, title, x_label, fig_size):
    print('Creating the figure')

    plt.figure(figsize=fig_size)
    plt.plot( x , y_crf, label='CRF Tagger')
    plt.plot(x,  y_regex, label='Regex Tagger')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('Evaluation')
    plt.legend()

    plt.savefig(f"plots/"+name_fig+".svg", format="svg")
    plt.savefig(f"plots/"+name_fig+".png", format="png")
    plt.savefig(f"plots/"+name_fig+".eps", format="eps")


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
    train_data = treebank.tagged_sents()[:3000]
    # for i in tqdm(range(3000)):
    #     #     train_data.append(treebank.tagged_sents()[i])
        #print(i, treebank.tagged_sents()[i])
    #print('train data', train_data)
    print('len train data', len(train_data))

    crt.train(train_data, model_file)  # the model will be saved in 'model.crf.tagger" file

    print('evaluating ... ')
    test_gold_data = treebank.tagged_sents()[3000:]
    # for i in tqdm(range(3000, 3914)):
    #  #   print(i, treebank.tagged_sents()[i])
    #     test_gold_data.append(treebank.tagged_sents()[i])
    #print('test gold data', test_gold_data)
    print('len gold data', len(test_gold_data))

    res = crt.evaluate(test_gold_data) # evaluate the data
    return res


def main():
    ############################  QUESTION 3

    # print(f'len PTB {len(treebank.tagged_sents())}')
    # print( len( treebank.sents() ) )
    #
    # crt = CRFTagger()
    # res1 = question3(crt, 'model.crf.tagger')
    # print('res1', res1)
    #
    # print('adding context before :')
    # crt_v2 = CRFTagger_v2(extra_features=-1)
    # res2 = question3(crt_v2, 'model_v2.crf.tagger')
    # print('res2', res2)
    #
    # print(res1, res2)
    #
    # print('adding context before + after :')
    # crt_v3 = CRFTagger_v3()
    # res3 = question3(crt_v3, 'model_v3.crf.tagger')
    # print('res3', res3)

    ############################  QUESTION 4

    # train_data = treebank.tagged_sents()[:3000]
    # test_data = treebank.sents()[3000:3914]
    # test_gold_data = treebank.tagged_sents()[3000:3914]
    #
    #         # impact of nb of rules
    #
    # nb_max_rules = [0, 10, 50, 100, 150, 200]
    # res_crf_nb_rules, res_regex_nb_rules = question4_nb_rules(train_data, test_data, test_gold_data, nb_max_rules)
    # print('res regex', res_regex_nb_rules)
    # print('res crf', res_crf_nb_rules)
    # # res regex [0.26324195985322685, 0.5313619684869415, 0.6762357004101014, 0.7275199654651414, 0.7589035182387223, 0.7788905676667386]
    # # res crf [0.9474638463198791, 0.9480682063457803, 0.9488884092380747, 0.9497086121303691, 0.9502266350097129, 0.9501834664364343]
    # #res_crf_nb_rules = [0.9474638463198791, 0.9480682063457803, 0.9488884092380747, 0.9497086121303691, 0.9502266350097129, 0.9501834664364343]
    # #res_regex_nb_rules = [0.26324195985322685, 0.5313619684869415, 0.6762357004101014, 0.7275199654651414, 0.7589035182387223, 0.7788905676667386]
    # plot(nb_max_rules, res_crf_nb_rules, res_regex_nb_rules, name_fig='max_rules', title='Impact du nombre de règles a retenir \n sur la performance des modèles.', x_label='Nombre des regles maximales a retenir', fig_size=(5, 4))

            ## impact of accuracies

    # min_accuracies_list = [0.8 , 0.85, 0.9, 0.95, 0.99, 1]
    # res_crf_min_acc, res_regex_min_acc = question4_min_accuracies(train_data, test_data, test_gold_data, min_accuracies_list)
    # print('res regex min acc ', res_regex_min_acc)
    # print('res crf min acc ', res_crf_min_acc)
    # # res regex min acc  [0.8039715087416361, 0.8039715087416361, 0.8010360457586877, 0.7972372113101662, 0.793524714008202, 0.7660263328297]
    # # res crf min acc  [0.9501834664364343, 0.9493200949708612, 0.9493200949708612, 0.9493200949708612, 0.9493200949708612, 0.9493200949708612]
    ##res_crf_min_acc = [0.9501834664364343, 0.9493200949708612, 0.9493200949708612, 0.9493200949708612, 0.9493200949708612, 0.9493200949708612]
    ##res_regex_min_acc = [0.8039715087416361, 0.8039715087416361, 0.8010360457586877, 0.7972372113101662, 0.793524714008202, 0.7660263328297]
    # plot(min_accuracies_list, y_crf=res_crf_min_acc, y_regex=res_regex_min_acc, name_fig='min_accu',
    #      title='Impact du accuracy minimale sur la \n performance des modèles.',
    #       x_label='Accuracy minimale', fig_size = (6, 4))

            ## impact of templates

    # todo: what does pos,  word templates mean ? ...
    # templates = nltk.tag.brill.brill24()
    # templates_pos_only = templates[:11]
    # templates_word_only = templates[11:20]
    # templates_word_pos = templates[20:]
    # tem_categories = [templates, templates_pos_only, templates_word_only, templates_word_pos]
    # res_crf_tmp, res_regex_tmp = question4_templates(train_data, test_data, test_gold_data, tem_categories)
    # print('res crf temp ', res_crf_tmp)
    # print('res regex temp ', res_regex_tmp)
    # #res cr temp[0.9501834664364343, 0.9476365206129937, 0.9498812864234837, 0.9504424778761061]
    # #res regex temp[0.7788905676667386, 0.44131232462767106, 0.7929635225555796, 0.5038635873084395]
    ##res_regex_tmp = [0.7788905676667386, 0.44131232462767106, 0.7929635225555796, 0.5038635873084395]
    ##res_crf_tmp = [0.9501834664364343, 0.9476365206129937, 0.9498812864234837, 0.9504424778761061]
    #tem_categories_names = ['All templates', 'POS-only templates', 'Word-only templates', 'Word-Pos templates']
    #plot_templates(res_crf_tmp, res_regex_tmp,  tem_categories_names, 'templates_cat' )


                ## analyse des types des regles apprises:
    # regex_tagger = RegexpTagger([
    #     (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    #     (r'(The|the|A|a|An|an)$', 'AT'),  # articles
    #     (r'.*able$', 'JJ'),  # adjectives
    #     (r'.*ness$', 'NN'),  # nouns formed from adjectives
    #     (r'.*ly$', 'RB'),  # adverbs
    #     (r'.*s$', 'NNS'),  # plural nouns
    #     (r'.*ing$', 'VBG'),  # gerunds
    #     (r'.*ed$', 'VBD'),  # past tense verbs
    #     (r'.*', 'NN')  # nouns (default)
    # ])
    # templates = nltk.tag.brill.brill24()
    # crt_tagger = CRFTagger()
    # crt_tagger.set_model_file('model.crf.tagger')
    #
    # brill_tagger1 = BrillTaggerTrainer(regex_tagger, templates, trace=3)
    # tagger1 = brill_tagger1.train(train_data)
    # print('Rules')
    # print( tagger1.rules() )
    # print('Train stats')
    # print( tagger1.train_stats())
    # print('Train stats: rules scores')
    # print(tagger1.train_stats()['rulescores'])
    # print('Templatest Stats')
    # print(tagger1.print_template_statistics())
    #
    # brill_tagger2 = BrillTaggerTrainer(crt_tagger, templates, trace=3)
    # tagger2 = brill_tagger2.train(train_data)
    # print('Rules')
    # print(tagger2.rules())
    # print('Train stats')
    # print(tagger2.train_stats())
    # print('Train stats: rules scores')
    # print(tagger2.train_stats()['rulescores'])
    # print('Templatest Stats')
    # print(tagger2.print_template_statistics())


            ### taggers chain

    #todo:

    ###### QUESTION 5
    #todo


if __name__ == '__main__':
    main()