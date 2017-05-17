# -*- coding: utf-8 -*-

import itertools
from random import shuffle
from nltk.classify import NaiveBayesClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def bigrams_words_features(words, nbigrams=200, measure=BigramAssocMeasures.chi_sq):
    """
    바이그램 단어 특징 추출
    :param words: 단어 list 
    :param nbigrams: 추출할 바이그램 갯수
    :param measure: 점수 측정 함수
    :return: 추출된 바이그램 dict
    """
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(measure, nbigrams)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


def train_and_test(reviews_pos, reviews_neg):
    """
    훈련 및 테스트
    :param reviews_pos: 긍정 리뷰 list
    :param reviews_neg: 부정 리뷰 list
    :return:
    """
    posfeatures = [(bigrams_words_features(r.words, 500), 'pos') for r in reviews_pos]
    negfeatures = [(bigrams_words_features(r.words, 500), 'neg') for r in reviews_neg]

    # 훈련 집합 80%와 테스트 집합 20% 분리
    portionpos = int(len(posfeatures) * 0.8)
    portionneg = int(len(negfeatures) * 0.8)
    print(portionpos, '-', portionneg)

    trainfeatures = negfeatures[:portionpos] + posfeatures[:portionneg]
    print(len(trainfeatures))

    # 훈련
    classifier = NaiveBayesClassifier.train(trainfeatures)

    # 테스트
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
