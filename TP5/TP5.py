from nltk.corpus import treebank
from statistics import mean
import nltk
from nltk import Nonterminal, Production
from nltk.corpus import treebank
from nltk import induce_pcfg
from nltk.parse import ViterbiParser
import time
import numpy as np
import matplotlib.pyplot as plt


def install_treebank():
    import nltk
    nltk.download('treebank')


CoLa_train_file = 'CoLA_data/train.tsv'
Cola_dev_file = 'CoLA_data/dev.tsv'
Cola_test_file = 'CoLA_data/test.tsv'


def wrong_sentences(file):
    # Reads the file and returns a list of grammatically wrong sentences, those marked with a star
    sentences = []

    with open(file, 'r', encoding="utf8") as f:
        for line in f.readlines():
            if line.split('\t')[1] == '0':
                sentences.append(line.rstrip().split('\t')[3])
    f.close()
    print('Cola dev set has ', len(sentences), ' wrong sentences')
    return sentences


def filter_rules(productions, seuil):  # On garde les repetitions + l'ordre
    print('start filtering .. only keeping productions that appear > 1 ')
    productions_filtered = [x for x in productions if productions.count(x) > seuil] # rules that appear > seuil time
    return productions_filtered


def train_PCFG_grammar_using_PTB( filter_by_frequency = 0):
    # extract productions from treebank phrase trees and induce the PCFG grammar
    print("Induce PCFG grammar from treebank data:")

    productions = []
    for item in treebank.fileids()[:1]:
        for tree in treebank.parsed_sents(item):
            # perform optional tree transformations, e.g.:
            tree.collapse_unary(collapsePOS=False)  # Remove branches A-B-C into A-B+C
            tree.chomsky_normal_form(horzMarkov=2)  # Remove A->(B,C,D) into A->B,C+D->D
            productions += tree.productions()

    print('*** productions before filtering:')
    print(len(productions))  # 211968
    # print(productions)

    grammar_rules = productions

    # FILTER THE GRAMMAR
    if filter_by_frequency > 0 :
        # filter the grammar, before learning the probabilities
        productions_filtered = filter_rules(productions, filter_by_frequency )
        grammar_rules = productions_filtered

        print('*** productions after filtering:')
        print(len(productions_filtered))
        #print(productions_filtered)


    # Handle UNK words 1/2
    productions_with_UNK = grammar_rules.copy()
    # pour tout non terminal A intervenant dans une regle lexicale (e.g. A -> the) , ajouter des regles A -> UNK
    non_terminals_in_lexical_rules = set()  # set does not allow duplicates
    for p in grammar_rules:
        if p.is_lexical() and isinstance(p.lhs(), Nonterminal):
            non_terminals_in_lexical_rules.add(p.lhs())

    for t in non_terminals_in_lexical_rules:
        new_rule = Production(t, ['UNK'])  # create a new rule
        productions_with_UNK.append(new_rule)

    print('*** productions after handling UNK words:')
    print(len(productions_with_UNK))  # 212014 (filtrage) regles
    # print(productions_with_UNK)

    # induce the grammar
    S = Nonterminal('S')
    grammar = induce_pcfg(S, productions_with_UNK)  # different from the one in the doc

    return grammar


def parse_sentences(phrases, grammar, parser):
    print('Parsing sentences ')

    times = []
    num_parses = []
    lengthes = []  # nb of words

    for sent in phrases:
        print(sent)
        tokens = sent.split()  # tokenize the sentence
        lengthes.append(len(tokens))
        print(tokens)

        # Handle UNK words 2/2
        print('Checking coverage')
        for index, token in enumerate(tokens):
            try:
                grammar.check_coverage([token])  # takes a list !!
            except:
                tokens[index] = 'UNK'
        print(tokens)

        t = time.time()
        parses_of_a_phrase = parser.parse_all(tokens)
        times.append(time.time() - t)
        num_parses.append(len(parses_of_a_phrase))

        print(parses_of_a_phrase)

    return times, num_parses, lengthes


def plot_figure(x, y1, y2, label1, label2, xlabel, ylabel, name, title ):

    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()
    plt.savefig(f"plots/courbe-{name}.svg", format="svg")
    plt.savefig(f"plots/courbe-{name}.png", format="png")
    plt.savefig(f"plots/courbe-{name}.eps", format="eps")
    plt.show()


def save_grammar(name, grammar):
    # save the grammar:
    print('saving the grammar')
    with open(name+".pcfg", "w") as grammar_file:
        for i in range(len(grammar.productions())):
            grammar_file.write(str(grammar.productions()[i]))
            grammar_file.write('\n')
    grammar_file.close()


def analysis_question5_c(nb_of_cola_phrases, grammar, parser, sentences):
    print('Analyse***')

    times, num_parses, lengths = parse_sentences(sentences[:nb_of_cola_phrases], grammar, parser)
    print(times, num_parses, lengths)
    lengthes_sorted, times_sorted = zip(*sorted(zip(lengths, times)))

    grammar_filtered = train_PCFG_grammar_using_PTB(filter_by_frequency=1)
    save_grammar('grammar_filtered', grammar_filtered)
    times_filtered, num_parses_filtered, lengths_filtered = parse_sentences(sentences[:nb_of_cola_phrases], grammar_filtered,
                                                                            parser)
    assert lengths == lengths_filtered

    lengthes_filtered_sorted, times_filtered_sorted = zip(*sorted(zip(lengths_filtered, times_filtered)))
    assert lengthes_sorted == lengthes_filtered_sorted

    plot_figure(lengthes_sorted, times_sorted, times_filtered_sorted, "pas de filtrage", "avec filtrage",
                "Longeurs de phrases considérées",
                "Le temps d'analyse (en sec)", 'time-length',
                "Le temps d'analyse en\nfonction des longeurs des phrases considérées")

    print(num_parses)
    print(num_parses_filtered)

    print('nb of pharses non_reconnus', num_parses.count(0))
    print('nb of pharses non_reconnus apres filtrage des regles', num_parses_filtered.count(0))

    print('Done Analyse')


def main():
    #install_treebank()  # first time only

    # Question 3 : train a PCFG grammar using the phrase trees from the PTB corpus
    grammar = train_PCFG_grammar_using_PTB() # no filtering
    print(grammar)
    save_grammar('grammar', grammar)

    # # load the grammar    #todo: loading the grammar did not work
    # print('loading the grammar .. ')
    # with open("grammar.pcfg", "r") as grammar_file:
    #     S = grammar_file.read()
    # grammar_file.close()
    # print(S)
    # #pcfg = nltk.PCFG.fromstring(S)
    #
    # feat0 = nltk.data.load('grammar.pcfg', verbose=True)
    # print(feat0)

    # Question 4 : use the grammar to parse grammatically wrong sentences from Cola using ViterbiParser
    wrong_sents_cola = wrong_sentences(Cola_dev_file)

    parser = ViterbiParser(grammar)
    parser.trace(0)  # put 3 for a verbose output

    #parse_sentences(wrong_sents_cola, grammar, parser)

    # Question 5.C Analyse
    analysis_question5_c(50, grammar, parser, wrong_sents_cola)

    # # Question 5.B
    #
    #     # longeur moyenne Cola
    # sents_cola = wrong_sentences(Cola_dev_file)
    # average_length_cola = mean([len(sent.split()) for sent in sents_cola]) #todo: should we count punctuation '.' , ','as words ?  the leaves() does
    # print("%.2f" % average_length_cola)
    # print(round(average_length_cola))  # round it to have exact nb of words
    #
    #
    #      # longeur moyenne Treebank
    # sentences_lengths_PTB = []
    # print(len(treebank.fileids())) # 199 file
    #
    # for item in treebank.fileids():
    #     for tree in treebank.parsed_sents(item):
    #         # print(tree.leaves())
    #         sentences_lengths_PTB.append(len(tree.leaves()))
    #
    # print(len(sentences_lengths_PTB)) #  3914 phrases in PTB
    #
    # average_length_PTB = mean(sentences_lengths_PTB)
    # print("%.2f" % average_length_PTB )
    # print(round(average_length_PTB))  # round it to have exact nb of words
    #
    #
    # # Vous pouvez obtenir les arbres de ces phrases comme suit :
    # #for item in treebank.fileids():
    #  #   for tree in treebank.parsed_sents(item):
    #   #      print(tree)
    #
    # # see a tree
    # print ( treebank.parsed_sents(treebank.fileids()[0]) )
    # t = treebank.parsed_sents(treebank.fileids()[0])[0]
    # t.draw()

if __name__ == '__main__':
    main()
