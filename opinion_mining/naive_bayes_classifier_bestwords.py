# -*- coding: utf-8 -*-

from nltk.metrics import BigramAssocMeasures
from nltk.classify import NaiveBayesClassifier
from random import shuffle


# training naive bayes with chi square feature selection of best words
def best_words_features(words, bestwords):
    return dict([(word, True) for word in words if word in bestwords])


def train_and_test(reviews_pos, reviews_neg):
    tot_poswords = [val for l in [r.words for r in reviews_pos] for val in l]
    tot_negwords = [val for l in [r.words for r in reviews_neg] for val in l]
    from nltk.probability import FreqDist, ConditionalFreqDist

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
    # select the best words in terms of information contained in the two classes pos and neg
    word_scores = {}

    for word, freq in iter(word_fd.items()):
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
                                               (freq, pos_words), tot_words)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
                                               (freq, neg_words), tot_words)
        word_scores[word] = pos_score + neg_score
    print('total: ', len(word_scores))
    best = sorted(iter(word_scores.items()), key=lambda args: args[1], reverse=True)[:10000]
    bestwords = set([w for w, s in best])

    negfeatures = [(best_words_features(r.words, bestwords), 'neg') for r in reviews_neg]
    posfeatures = [(best_words_features(r.words, bestwords), 'pos') for r in reviews_pos]
    portionpos = int(len(posfeatures) * 0.8)
    portionneg = int(len(negfeatures) * 0.8)
    print(portionpos, '-', portionneg)
    trainfeatures = negfeatures[:portionpos] + posfeatures[:portionneg]
    print(len(trainfeatures))
    classifier = NaiveBayesClassifier.train(trainfeatures)
    ##test with feature chi square selection
    testfeatures = negfeatures[portionneg:] + posfeatures[portionpos:]
    shuffle(testfeatures)
    err = 0
    print('test on: ', len(testfeatures))
    for r in testfeatures:
        sent = classifier.classify(r[0])
        # print r[1],'-pred: ',sent
        if sent != r[1]:
            err += 1.
    print('error rate: ', err / float(len(testfeatures)))
