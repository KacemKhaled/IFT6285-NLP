from nltk.corpus import treebank
from statistics import mean
from nltk import Nonterminal
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


def main():
    #install_treebank()  # first time only

    #Question 3 : train a PCFG grammar using the phrase trees from the PTB corpus
    # extract productions from three trees and induce the PCFG
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

    #filter the grammar, before learning the probabilities : from 115 to 20 regle >1 why do we do it ?
    d = {i: productions.count(i) for i in set(productions)} # regles , frequencies
    productions_filtered = [item[0] for item in d.items() if int(item[1]) > 1]
    print(len(productions_filtered)) # 20 regles

    S = Nonterminal('S')
    grammar = induce_pcfg(S, productions)  # different from the one in the doc
    print(grammar)

    # Question 4 : use the grammar to parse grammatically wrong sentences from Cola using ViterbiParser
    print('Parsing sentences ')
    parser = ViterbiParser(grammar)
    parser.trace(0) # put 3 for a verbose output

    sents_cola = wrong_sentences(Cola_dev_file)
    for sent in sents_cola:
        print(sent)
        tokens = sent.split()  # tokenize the sentence

        print('Checking coverage')
        try:
            grammar.check_coverage(tokens) # takes a list !!
            print("All words covered")
        except:
            print("Some words not covered")

        parses = parser.parse_all(tokens)     # todo: do something with the parses
        print(parses)




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
