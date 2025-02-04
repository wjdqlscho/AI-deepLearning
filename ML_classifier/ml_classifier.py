# 라이브러리 & 데이터 로드
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('exercise1.csv')

nrows, ncols = 2, 3
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
fig.set_size_inches(10, 6)

for i in range(nrows):
  for j in range(ncols):
    attr = i * ncols + j
    sns.histplot(x=df.columns[attr], data=df, hue = 'target', ax = axs[i][j])

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. 데이터 분할
X = df.iloc[:, :6] #독립변수
y = df['target'] #종속변수

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y) #20퍼센트만 테스트로 사용하는 것으로 지정
# random_state는 동일한 크기로 사용하게 함.
# stratify는 균일하게 분할할 수 있게 해준다.

# 2. 모델 인스턴스 생성
model = DecisionTreeClassifier(random_state=42)

# 3. 모델 학습
model.fit(X_train, y_train)

# 4. 모델 평가
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)