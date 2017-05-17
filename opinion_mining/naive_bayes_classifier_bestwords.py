# -*- coding: utf-8 -*-

from random import shuffle
from nltk.metrics import BigramAssocMeasures
from nltk.classify import NaiveBayesClassifier
from nltk.probability import FreqDist, ConditionalFreqDist


def best_words_features(words, bestwords):
    """
    words list에서 bestwords에 포함된 단어만 dict로 변환
    :param words: 단어 list
    :param bestwords: 베스트 단어 list
    :return: 최빈도 단어 dict
    """
    return dict([(word, True) for word in words if word in bestwords])


def train_and_test(reviews_pos, reviews_neg):
    """
    훈련 및 테스트
    :param reviews_pos: 긍정 리뷰 list
    :param reviews_neg: 부정 리뷰 list
    :return:
    """

    # 긍정 리뷰, 부정 리뷰 각각에서의 전체 단어에 대한 빈도수 계산
    tot_poswords = [val for l in [r.words for r in reviews_pos] for val in l]
    tot_negwords = [val for l in [r.words for r in reviews_neg] for val in l]

    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    for word in tot_poswords:
        word_fd[word.lower()] += 1
        label_word_fd['pos'][word.lower()] += 1
    for word in tot_negwords:
        word_fd[word.lower()] += 1
        label_word_fd['neg'][word.lower()] += 1

    pos_words = len(tot_poswords)
    neg_words = len(tot_negwords)
    tot_words = pos_words + neg_words

    # 각 단어별 점수
    word_scores = {}
    for word, freq in iter(word_fd.items()):
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word], (freq, pos_words), tot_words)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word], (freq, neg_words), tot_words)
        word_scores[word] = pos_score + neg_score
    print('total: ', len(word_scores))

    # 점수가 높은 10000개의 단어만 추출
    best = sorted(iter(word_scores.items()), key=lambda args: args[1], reverse=True)[:10000]
    bestwords = set([w for w, s in best])

    negfeatures = [(best_words_features(r.words, bestwords), 'neg') for r in reviews_neg]
    posfeatures = [(best_words_features(r.words, bestwords), 'pos') for r in reviews_pos]

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
        # print(r[1], '-pred: ', sent)
        if sent != r[1]:
            err += 1.
    print('error rate: ', err / float(len(testfeatures)))
