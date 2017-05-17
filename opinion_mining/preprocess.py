# -*- coding: utf-8 -*-

import os
import nltk
from bs4 import BeautifulSoup
from collections import namedtuple
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# 전처리를 위한 라이브러리
tknzr = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
nltk.download('stopwords')
stoplist = stopwords.words('english')
stemmer = PorterStemmer()

# 영화 및 리뷰 파일 경로
movie_html_dir = './data/movie/'
Review = namedtuple('Review', 'words title tags')
reviews_dir = './data/review_polarity/txt_sentoken/'
do2vecstem = True

# 캐시 파일 경로
movie_titles_cache_file_path = './cache/movie_titles.txt'
reviews_pos_cache_file_path = './cache/reviews_pos.txt'
reviews_neg_cache_file_path = './cache/reviews_neg.txt'


def get_movie_titles():
    """
    영화 제목 dict
    :return: dict
    """
    if os.path.exists(movie_titles_cache_file_path):
        f = open(movie_titles_cache_file_path, 'r')
        data = f.read()
        f.close()
        movie_titles = eval(data)

    else:
        movie_titles = {}
        for filename in [f for f in os.listdir(movie_html_dir) if f[0] != '.']:
            id = filename.split('.')[0]
            f = open(movie_html_dir + '/' + filename)
            parsed_html = BeautifulSoup(f.read())
            try:
                title = parsed_html.body.h1.text

            except:
                title = 'none'
            movie_titles[id] = title

        f = open(movie_titles_cache_file_path, 'w')
        f.write(str(movie_titles))
        f.close()

    return movie_titles


def preprocess_reviews(filename, text, stop=[], stem=False):
    """
    리뷰 내용 전처리
    :param filename: 파일 경로
    :param text: 리뷰 내용
    :param stop: 불용어 리스트
    :param stem: 어간 추출 여부
    :return: list
    """
    # 토큰화
    words = tknzr.tokenize(text)
    if stem:
        try:
            # 불용어(stopwords) 제거 및 어간 추출(stemming)
            words_clean = [stemmer.stem(w) for w in [i.lower() for i in words if i not in stop]]
        except IndexError as e:
            print("IndexError: " + filename)
            words_clean = []
    else:
        words_clean = [i.lower() for i in words if i not in stop]
    return words_clean


def get_positive_reviews(movie_titles):
    """
    긍정 리뷰 list
    :param movie_titles: 영화 제목 dict
    :return: 리뷰 list
    """
    if os.path.exists(reviews_pos_cache_file_path):
        f = open(reviews_pos_cache_file_path, 'r')
        data = f.read()
        f.close()
        reviews_pos = eval(data)

    else:
        reviews_pos = []
        cnt = 0
        for filename in [f for f in os.listdir(reviews_dir + 'pos/') if str(f)[0] != '.']:
            f = open(reviews_dir + 'pos/' + filename, 'r')
            id = filename.split('.')[0].split('_')[1]
            reviews_pos.append(
                Review(preprocess_reviews(filename, f.read(), stoplist, do2vecstem), movie_titles[id], ['pos_' + str(cnt)]))
            cnt += 1

        f = open(reviews_pos_cache_file_path, 'w')
        f.write(str(reviews_pos))
        f.close()

    return reviews_pos


def get_negative_reviews(movie_titles):
    """
    부정 리뷰 list
    :param movie_titles: 영화 제목 dict
    :return: 리뷰 list
    """
    if os.path.exists(reviews_pos_cache_file_path):
        f = open(reviews_neg_cache_file_path, 'r')
        data = f.read()
        f.close()
        reviews_neg = eval(data)

    else:
        reviews_neg = []
        cnt = 0
        for filename in [f for f in os.listdir(reviews_dir + 'neg/') if str(f)[0] != '.']:
            f = open(reviews_dir + 'neg/' + filename, 'r')
            id = filename.split('.')[0].split('_')[1]
            reviews_neg.append(
                Review(preprocess_reviews(filename, f.read(), stoplist, do2vecstem), movie_titles[id], ['neg_' + str(cnt)]))
            cnt += 1

        f = open(reviews_neg_cache_file_path, 'w')
        f.write(str(reviews_neg))
        f.close()

    return reviews_neg
