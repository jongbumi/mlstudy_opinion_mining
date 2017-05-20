# 머신러닝 스터디 - 오피니언 마이닝 예제

이 소스는 '웹을 위한 머신러닝' 도서의 181p ~ 188p에 있는 오피니언 마이닝 방법에 대한 예제 소스입니다.

원 소스는 다음 링크에 있습니다.

https://github.com/ai2010/machine_learning_for_the_web/tree/master/chapter_4

Python 2에 맞게 작성되어 있던 소스를 Python 3에서 동작 가능하도록 수정 및 약간의 리팩토링을 진행하였으며 전체 흐름은 원 소스와 동일합니다.

## 필수사항
- Python3
- Virtualenv

## 설치할 패키지
- BeautifulSoup4
- nltk
- gensim
- sklearn

## 디렉토리 구조
- data
  - movie : 영화 정보
  - review_polarity/txt_sentoken : 영화 리뷰 내용
    - pos : 긍정 리뷰
    - neg : 부정 리뷰
- cache : 영화 제목 및 리뷰에 대한 전처리가 이미 되어 있는 캐시 데이터
- opinion_mining : 오피니언 마이닝 전처리 및 분류기 사용 예제 패키지

## 실행 방법
```
git clone https://github.com/jongbumi/mlstudy_opinion_mining.git

cd mlstudy_opinion_mining
virtualenv -p python3 venv
. venv/bin/activate

pip install BeautifulSoup4 nltk gensim sklearn

python ./main.py
```
