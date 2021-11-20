# Natural Language Toolkit: Interface to the CRFSuite Tagger
#
# Copyright (C) 2001-2021 NLTK Project
# Author: Long Duong <longdt219@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A module for POS tagging using CRFSuite
"""

import re
import unicodedata
from tqdm import tqdm

from nltk.tag.api import TaggerI

try:
    import pycrfsuite
except ImportError:
    pass


class CRFTagger_v2(TaggerI):
    """
    A module for POS tagging using CRFSuite https://pypi.python.org/pypi/python-crfsuite

    >>> from nltk.tag import CRFTagger
    >>> ct = CRFTagger()

    >>> train_data = [[('University','Noun'), ('is','Verb'), ('a','Det'), ('good','Adj'), ('place','Noun')],
    ... [('dog','Noun'),('eat','Verb'),('meat','Noun')]]

    >>> ct.train(train_data,'model.crf.tagger')
    >>> ct.tag_sents([['dog','is','good'], ['Cat','eat','meat']])
    [[('dog', 'Noun'), ('is', 'Verb'), ('good', 'Adj')], [('Cat', 'Noun'), ('eat', 'Verb'), ('meat', 'Noun')]]

    >>> gold_sentences = [[('dog','Noun'),('is','Verb'),('good','Adj')] , [('Cat','Noun'),('eat','Verb'), ('meat','Noun')]]
    >>> ct.evaluate(gold_sentences)
    1.0

    Setting learned model file
    >>> ct = CRFTagger()
    >>> ct.set_model_file('model.crf.tagger')
    >>> ct.evaluate(gold_sentences)
    1.0
    """

    def __init__(self, feature_func=None, verbose=False, training_opt={},extra_features=0,context=None):
        """
        Initialize the CRFSuite tagger

        :param feature_func: The function that extracts features for each token of a sentence. This function should take
            2 parameters: tokens and index which extract features at index position from tokens list. See the build in
            _get_features function for more detail.
        :param verbose: output the debugging messages during training.
        :type verbose: boolean
        :param training_opt: python-crfsuite training options
        :type training_opt: dictionary

        Set of possible training options (using LBFGS training algorithm).
            :'feature.minfreq': The minimum frequency of features.
            :'feature.possible_states': Force to generate possible state features.
            :'feature.possible_transitions': Force to generate possible transition features.
            :'c1': Coefficient for L1 regularization.
            :'c2': Coefficient for L2 regularization.
            :'max_iterations': The maximum number of iterations for L-BFGS optimization.
            :'num_memories': The number of limited memories for approximating the inverse hessian matrix.
            :'epsilon': Epsilon for testing the convergence of the objective.
            :'period': The duration of iterations to test the stopping criterion.
            :'delta': The threshold for the stopping criterion; an L-BFGS iteration stops when the
                improvement of the log likelihood over the last ${period} iterations is no greater than this threshold.
            :'linesearch': The line search algorithm used in L-BFGS updates:

                - 'MoreThuente': More and Thuente's method,
                - 'Backtracking': Backtracking method with regular Wolfe condition,
                - 'StrongBacktracking': Backtracking method with strong Wolfe condition
            :'max_linesearch':  The maximum number of trials for the line search algorithm.
        :param extra_features: extras features to take into consideration
                0: no extra features, -1: previous word, 1: next word, 2 or higher: previous and next word
        :type int

        """

        self._model_file = ""
        self._tagger = pycrfsuite.Tagger()

        if feature_func is None:
            self._feature_func = self._get_features
        else:
            self._feature_func = feature_func

        self._verbose = verbose
        self._training_options = training_opt
        self._pattern = re.compile(r"\d")
        self.extra_features = extra_features
        self.context = context


    def set_model_file(self, model_file):
        self._model_file = model_file
        self._tagger.open(self._model_file)


    def _get_features(self, tokens, idx, context=None):
        """
        Extract basic features about this word including
            - Current word
            - is it capitalized?
            - Does it have punctuation?
            - Does it have a number?
            - Suffixes up to length 3

        Note that : we might include feature over previous word, next word etc.

        :return: a list which contains the features
        :rtype: list(str)
        """
        token = tokens[idx]

        feature_list = []

        if not token:
            return feature_list


        # Capitalization
        if token[0].isupper():
            feature_list.append("CAPITALIZATION")

        # Number
        if re.search(self._pattern, token) is not None:
            feature_list.append("HAS_NUM")

        # Punctuation
        punc_cat = {"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"}
        if all(unicodedata.category(x) in punc_cat for x in token):
            feature_list.append("PUNCTUATION")

        # Suffix up to length 3
        if len(token) > 1:
            feature_list.append("SUF_" + token[-1:])
        if len(token) > 2:
            feature_list.append("SUF_" + token[-2:])
        if len(token) > 3:
            feature_list.append("SUF_" + token[-3:])

        feature_list.append("WORD_" + token)

        if context != "": context= self.context
        # Context
        if self.extra_features == -1 or self.extra_features > 1:
            if idx > 0 and tokens[idx - 1]:  # pour que idx-1 soit toujours >= 0
                if context=='features':
                    feature_list.extend(
                        ["PREV_WORD_"+feature for feature in self._get_features(tokens,idx - 1, context="")])
                elif context=='word':
                    feature_list.append("PREV_WORD_" + tokens[idx - 1])  # previous word
        if self.extra_features > 0:
            if idx + 1 < len(tokens) and tokens[idx + 1]:
                if context == 'features':
                    feature_list.extend(
                        ["NEXT_WORD_" + feature for feature in self._get_features(tokens, idx + 1, context="")])
                elif context=='word':
                    feature_list.append("NEXT_WORD_" + tokens[idx + 1])  # next word

        return feature_list

    def tag_sents(self, sents):
        """
        Tag a list of sentences. NB before using this function, user should specify the mode_file either by

        - Train a new model using ``train`` function
        - Use the pre-trained model which is set via ``set_model_file`` function

        :params sentences: list of sentences needed to tag.
        :type sentences: list(list(str))
        :return: list of tagged sentences.
        :rtype: list(list(tuple(str,str)))
        """
        if self._model_file == "":
            raise Exception(
                " No model file is found !! Please use train or set_model_file function"
            )

        # We need the list of sentences instead of the list generator for matching the input and output
        result = []
        for tokens in sents:
            features = [self._feature_func(tokens, i) for i in range(len(tokens))]
            labels = self._tagger.tag(features)

            if len(labels) != len(tokens):
                raise Exception(" Predicted Length Not Matched, Expect Errors !")

            tagged_sent = list(zip(tokens, labels))
            result.append(tagged_sent)

        return result


    def train(self, train_data, model_file):
        """
        Train the CRF tagger using CRFSuite
        :params train_data : is the list of annotated sentences.
        :type train_data : list (list(tuple(str,str)))
        :params model_file : the model will be saved to this file.

        """
        trainer = pycrfsuite.Trainer(verbose=self._verbose)
        trainer.set_params(self._training_options)

        for sent in tqdm(train_data):
            tokens, labels = zip(*sent)
            features = [self._feature_func(tokens, i) for i in range(len(tokens))]
            trainer.append(features, labels)

        # Now train the model, the output should be model_file
        trainer.train(model_file)
        # Save the model file
        self.set_model_file(model_file)


    def tag(self, tokens):
        """
        Tag a sentence using Python CRFSuite Tagger. NB before using this function, user should specify the mode_file either by

        - Train a new model using ``train`` function
        - Use the pre-trained model which is set via ``set_model_file`` function

        :params tokens: list of tokens needed to tag.
        :type tokens: list(str)
        :return: list of tagged tokens.
        :rtype: list(tuple(str,str))
        """

        return self.tag_sents([tokens])[0]