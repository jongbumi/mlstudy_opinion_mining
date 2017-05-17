# -*- coding: utf-8 -*-

from random import shuffle
import numpy as np
from gensim.models import Doc2Vec
import multiprocessing


def train_and_test(reviews_pos, reviews_neg):
    tot_reviews = reviews_pos + reviews_neg
    shuffle(tot_reviews)
    cores = multiprocessing.cpu_count()
    vec_size = 500
    model_d2v = Doc2Vec(dm=1, dm_concat=0, size=vec_size, window=5, negative=0, hs=0, min_count=1, workers=cores)

    # build vocab
    model_d2v.build_vocab(tot_reviews)
    # train
    numepochs = 20
    for epoch in range(numepochs):
        try:
            print('epoch %d' % (epoch))
            model_d2v.train(tot_reviews, total_examples=model_d2v.corpus_count, epochs=model_d2v.iter)
            model_d2v.alpha *= 0.99
            model_d2v.min_alpha = model_d2v.alpha
        except (KeyboardInterrupt, SystemExit):
            break

    trainingsize = 2 * int(len(reviews_pos) * 0.8)

    train_d2v = np.zeros((trainingsize, vec_size))
    train_labels = np.zeros(trainingsize)
    test_size = len(tot_reviews) - trainingsize
    test_d2v = np.zeros((test_size, vec_size))
    test_labels = np.zeros(test_size)

    cnt_train = 0
    cnt_test = 0
    for r in reviews_pos:
        name_pos = r.tags[0]
        if int(name_pos.split('_')[1]) >= int(trainingsize / 2.):
            test_d2v[cnt_test] = model_d2v.docvecs[name_pos]
            test_labels[cnt_test] = 1
            cnt_test += 1
        else:
            train_d2v[cnt_train] = model_d2v.docvecs[name_pos]
            train_labels[cnt_train] = 1
            cnt_train += 1

    for r in reviews_neg:
        name_neg = r.tags[0]
        if int(name_neg.split('_')[1]) >= int(trainingsize / 2.):
            test_d2v[cnt_test] = model_d2v.docvecs[name_neg]
            test_labels[cnt_test] = 0
            cnt_test += 1
        else:
            train_d2v[cnt_train] = model_d2v.docvecs[name_neg]
            train_labels[cnt_train] = 0
            cnt_train += 1

    # train log regre
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression()
    classifier.fit(train_d2v, train_labels)
    print('accuracy:', classifier.score(test_d2v, test_labels))

    from sklearn.svm import SVC

    clf = SVC()
    clf.fit(train_d2v, train_labels)
    print('accuracy:', clf.score(test_d2v, test_labels))