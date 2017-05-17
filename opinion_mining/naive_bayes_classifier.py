# -*- coding: utf-8 -*-

from random import shuffle
from nltk.classify import NaiveBayesClassifier


# 긍정 및 부정 리뷰에서의 단어 특징 추출 (전체 단어 대상)
def word_features(words):
    return dict([(word, True) for word in words])


def train_and_test(reviews_pos, reviews_neg):
    posfeatures = [(word_features(r.words), 'pos') for r in reviews_pos]
    negfeatures = [(word_features(r.words), 'neg') for r in reviews_neg]
    portionpos = int(len(posfeatures) * 0.8)
    portionneg = int(len(negfeatures) * 0.8)
    print(portionpos, '-', portionneg)
    trainfeatures = negfeatures[:portionneg] + posfeatures[:portionpos]
    print(len(trainfeatures))
    testfeatures = negfeatures[portionneg:] + posfeatures[portionpos:]
    shuffle(testfeatures)

    # 추출된 단어 특징으로 훈련 및 테스트
    classifier = NaiveBayesClassifier.train(trainfeatures)

    # testing
    err = 0
    print('test on: ', len(testfeatures))
    for r in testfeatures:
        sent = classifier.classify(r[0])
        if sent != r[1]:
            err += 1.
    print('error rate: ', err / float(len(testfeatures)))
