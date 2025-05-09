# [본문 추출]
import requests # HTML 페이지 긁어오기
from bs4 import BeautifulSoup # <p> 태그 안 텍스트만 추출, 본문만 추출

url = "https://www.bbc.com/news/articles/c8rgrejkvmjo"
headers = {'User-Agent': 'Mozilla/5.0'}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# 기사 본문 추출
article_body = soup.find_all('p') # paragraph 문단 긁어오기
article_text = ' '.join([p.get_text() for p in article_body])

import nltk
nltk.download('punkt')
nltk.download('all') 

# [전처리]
import re # 줄바꿈, 공백 전처리
import nltk # 자연어처리
nltk.download('punkt')
from nltk.tokenize import sent_tokenize # 문장 분리

# 특수 문자 제거
clean_text = re.sub(r'\s+', ' ', article_text) # 공백 하나로 깔끔하게 정리

# 문장 단위로 분리
sentences = sent_tokenize(clean_text) # 리스트

# [중요도 계산]
# TF - 특정단어 빈도수 확인
# IDF - 단어 희귀 정도
# >> 중요단어 판단용

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# TF-IDF 벡터화 > 중요도 계산 위해
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)  # fit_transform - 중복제거, TF-IDF 점수로 변환
# 희소 행렬로 변환됨

# 문장별 TF-IDF 점수의 평균 계산
sentence_scores = X.mean(axis=1).A1  # .A1은 numpy matrix를 1D array로 변환

# 중요도 기준 상위 3개 문장 선택
top_n = 3
top_indices = np.argsort(sentence_scores)[-top_n:][::-1]  # 내림차순 정렬

# 요약 결과 출력
print("🔝 중요도 기준 상위 3개 문장:")
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank}. 점수: {sentence_scores[idx]:.4f} | 문장: {sentences[idx]}")
#----------------------------------------------------------------------------
sentence_scores = X.mean(axis=1).A1  # .A1 = numpy matrix → 1D ndarray

top_n = 10
top_indices = np.argsort(sentence_scores)[-top_n:][::-1]

print("🔝 중요도 기준 상위 10개 문장:")
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank}. 점수: {sentence_scores[idx]:.4f} | 문장: {sentences[idx]}")

