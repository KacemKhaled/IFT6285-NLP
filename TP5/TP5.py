from nltk.corpus import treebank
from statistics import mean
from nltk import Nonterminal, Production
from nltk.corpus import treebank
from nltk import induce_pcfg
from nltk.parse import ViterbiParser

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
    return sentences


def filter_rules(productions):
    d = {i: productions.count(i) for i in set(productions)}  # regles , frequencies
    productions_filtered = [item[0] for item in d.items() if int(item[1]) > 1] # rules that appear > 1 time
    return productions_filtered


def train_PCFG_grammar_using_PTB( filter_by_frequency = False):
    # extract productions from treebank phrase trees and induce the PCFG grammar
    print("Induce PCFG grammar from treebank data:")

    productions = []
    for item in treebank.fileids()[:2]:
        for tree in treebank.parsed_sents(item):
            # perform optional tree transformations, e.g.:
            tree.collapse_unary(collapsePOS=False)  # Remove branches A-B-C into A-B+C
            tree.chomsky_normal_form(horzMarkov=2)  # Remove A->(B,C,D) into A->B,C+D->D
            productions += tree.productions()

    print('*** productions:')
    print(len(productions))  # 115
    print(productions)

    grammar_rules = productions

    # FILTER THE GRAMMAR
    if filter_by_frequency:
        # filter the grammar, before learning the probabilities : from 115 to 20 regle >1 why do we do it ?
        productions_filtered = filter_rules(productions)
        print(len(productions_filtered))  # 20 regles
        grammar_rules = productions_filtered

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

    # induce the grammar
    S = Nonterminal('S')
    grammar = induce_pcfg(S, productions_with_UNK)  # different from the one in the doc
    print(grammar)

    return grammar


def parse_sentences(phrases, grammar, parser):
    print('Parsing sentences ')

    for sent in phrases:
        print(sent)
        tokens = sent.split()  # tokenize the sentence
        print(tokens)

        # Handle UNK words 2/2, todo prb: too many UNK ?
        print('Checking coverage')
        for index, token in enumerate(tokens):
            try:
                grammar.check_coverage([token])  # takes a list !!
            except:
                tokens[index] = 'UNK'
        print(tokens)

        parses_of_a_phrase = parser.parse_all(tokens)  # todo: do something with the parses later ?
        print(parses_of_a_phrase)


def main():
    #install_treebank()  # first time only

    # Question 3 : train a PCFG grammar using the phrase trees from the PTB corpus
    grammar = train_PCFG_grammar_using_PTB(filter_by_frequency=False )

    # Question 4 : use the grammar to parse grammatically wrong sentences from Cola using ViterbiParser
    wrong_sents_cola = wrong_sentences(Cola_dev_file)

    parser = ViterbiParser(grammar)
    parser.trace(0)  # put 3 for a verbose output

    parse_sentences(wrong_sents_cola, grammar, parser)


    # Question 5.B

        # longeur moyenne Cola
    sents_cola = wrong_sentences(Cola_dev_file)
    average_length_cola = mean([len(sent.split()) for sent in sents_cola]) #todo: should we count punctuation '.' , ','as words ?  the leaves() does
    print("%.2f" % average_length_cola)
    print(round(average_length_cola))  # round it to have exact nb of words


         # longeur moyenne Treebank
    sentences_lengths_PTB = []
    print(len(treebank.fileids())) # 199 file

    for item in treebank.fileids():
        for tree in treebank.parsed_sents(item):
            # print(tree.leaves())
            sentences_lengths_PTB.append(len(tree.leaves()))

    print(len(sentences_lengths_PTB)) #  3914 phrases in PTB

    average_length_PTB = mean(sentences_lengths_PTB)
    print("%.2f" % average_length_PTB )
    print(round(average_length_PTB))  # round it to have exact nb of words

    # Vous pouvez obtenir les arbres de ces phrases comme suit :
    #for item in treebank.fileids():
     #   for tree in treebank.parsed_sents(item):
      #      print(tree)

    # see a tree
    print ( treebank.parsed_sents(treebank.fileids()[0]) )
    t = treebank.parsed_sents(treebank.fileids()[0])[0]
    t.draw()

if __name__ == '__main__':
    main()
