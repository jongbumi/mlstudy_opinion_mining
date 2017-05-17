# -*- coding: utf-8 -*-

from opinion_mining import preprocess

from opinion_mining import naive_bayes_classifier
from opinion_mining import naive_bayes_classifier_bigram
from opinion_mining import naive_bayes_classifier_bestwords
from opinion_mining import doc2vec


print("전처리 시작")
movie_titles = preprocess.get_movie_titles()
reviews_pos = preprocess.get_positive_reviews(movie_titles)
reviews_neg = preprocess.get_negative_reviews(movie_titles)
print("전처리 종료\n")

print("'나이브 베이즈 분류기' 훈련 및 테스트 시작")
naive_bayes_classifier.train_and_test(reviews_pos, reviews_neg)
print("'나이브 베이즈 분류기' 훈련 및 테스트 종료\n")

print("'나이브 베이즈 분류기 + 바이그램' 훈련 및 테스트 시작")
naive_bayes_classifier_bigram.train_and_test(reviews_pos, reviews_neg)
print("'나이브 베이즈 분류기 + 바이그램' 훈련 및 테스트 종료\n")

print("'나이브 베이즈 분류기 + 최빈도 단어' 훈련 및 테스트 시작")
naive_bayes_classifier_bestwords.train_and_test(reviews_pos, reviews_neg)
print("'나이브 베이즈 분류기 + 최빈도 단어' 훈련 및 테스트 종료\n")

print("'Doc2Vec + 로지스틱 회기 분석 모델/SVM 분류기' 훈련 및 테스트 시작")
doc2vec.train_and_test(reviews_pos, reviews_neg)
print("'Doc2Vec + 로지스틱 회기 분석 모델/SVM 분류기' 훈련 및 테스트 종료\n")
