from nltk.corpus import treebank


def install_treebank():
    import nltk
    nltk.download('treebank')



def main():
    # install_treebank()  # first time only
    print(len(treebank.fileids()))
    for item in treebank.fileids():
       for tree in treebank.parsed_sents(item):
            print(tree)




if __name__ == '__main__':
        main()