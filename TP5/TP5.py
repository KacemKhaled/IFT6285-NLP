from nltk.corpus import treebank
from statistics import mean

def install_treebank():
    import nltk
    nltk.download('treebank')


CoLA_train_file = 'CoLA_data/train.tsv'
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

    # Question 5.B

        # longeur moyenne Cola
    sents_cola = wrong_sentences(CoLA_train_file)
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
