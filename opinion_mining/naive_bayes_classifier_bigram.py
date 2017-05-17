# -*- coding: utf-8 -*-

import itertools
from random import shuffle
from nltk.classify import NaiveBayesClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


# train bigram:
def bigrams_words_features(words, nbigrams=200, measure=BigramAssocMeasures.chi_sq):
    # words = [w for w in words if w != ' ']
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(measure, nbigrams)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


def train_and_test(reviews_pos, reviews_neg):
    posfeatures = [(bigrams_words_features(r.words, 500), 'pos') for r in reviews_pos]
    negfeatures = [(bigrams_words_features(r.words, 500), 'neg') for r in reviews_neg]

    portionpos = int(len(posfeatures) * 0.8)
    portionneg = int(len(negfeatures) * 0.8)
    print(portionpos, '-', portionneg)
    trainfeatures = negfeatures[:portionpos] + posfeatures[:portionneg]
    print(len(trainfeatures))
    classifier = NaiveBayesClassifier.train(trainfeatures)
    ##test bigram
    testfeatures = negfeatures[portionneg:] + posfeatures[portionpos:]
    shuffle(testfeatures)
    err = 0
    print('test on: ', len(testfeatures))
    for r in testfeatures:
        sent = classifier.classify(r[0])
        # print(r[1],'-pred: ',sent)
        if sent != r[1]:
            err += 1.
    print('error rate: ', err / float(len(testfeatures)))
