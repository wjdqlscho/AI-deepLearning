from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('exercise1.csv')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. 데이터 분할
X = df.iloc[:, :6]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# 2. 모델 인스턴스 생성
model = DecisionTreeClassifier(random_state=42)

# 3. 모델 학습
model.fit(X_train, y_train)

# 4. 모델 평가
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
accuracy_score(y_test, y_pred)

sc = StandardScaler()
X_scale = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size = 0.2, random_state = 42, stratify = y)

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
accuracy_score(y_test, y_pred) #모델을 평가할 때는 무조건 스케일(크기)를 같게 해서 비교를 해야 한다.
