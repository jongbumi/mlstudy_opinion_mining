# -*- coding: utf-8 -*-

from random import shuffle
from nltk.classify import NaiveBayesClassifier


def word_features(words):
    """
    list를 dict로 변환
    :param words: 단어 list
    :return: 
    """
    return dict([(word, True) for word in words])


def train_and_test(reviews_pos, reviews_neg):
    """
    훈련 및 테스트
    :param reviews_pos: 긍정 리뷰 list
    :param reviews_neg: 부정 리뷰 list
    :return:
    """
    posfeatures = [(word_features(r.words), 'pos') for r in reviews_pos]
    negfeatures = [(word_features(r.words), 'neg') for r in reviews_neg]

    # 훈련 집합 80%와 테스트 집합 20% 분리
    portionpos = int(len(posfeatures) * 0.8)
    portionneg = int(len(negfeatures) * 0.8)
    print(portionpos, '-', portionneg)

    trainfeatures = negfeatures[:portionneg] + posfeatures[:portionpos]
    print(len(trainfeatures))

    testfeatures = negfeatures[portionneg:] + posfeatures[portionpos:]
    shuffle(testfeatures)

    # 훈련
    classifier = NaiveBayesClassifier.train(trainfeatures)

    # 테스트
    err = 0
    print('test on: ', len(testfeatures))
    for r in testfeatures:
        sent = classifier.classify(r[0])
        if sent != r[1]:
            err += 1.
    print('error rate: ', err / float(len(testfeatures)))
