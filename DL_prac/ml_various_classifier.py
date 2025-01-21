import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

df = pd.read_csv('exercise1.csv')
X = df.iloc[:, :6] #독립변수
y = df['target'] #종속변수
# 2. 모델 인스턴스 생성
rf_cls = RandomForestClassifier(random_state=42)
gb_cls = GradientBoostingClassifier(random_state=42)
xgb_cls = XGBClassifier(random_state=42)
lgb_cls = LGBMClassifier(random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y) #20퍼센트만 테스트로 사용하는 것으로 지정

# 3. 모델 학습
rf_cls.fit(X_train, y_train)
gb_cls.fit(X_train, y_train)
xgb_cls.fit(X_train, y_train)
lgb_cls.fit(X_train, y_train)

# 4. 모델 평가
y_pred_rf = rf_cls.predict(X_test)
y_pred_gb = gb_cls.predict(X_test)
y_pred_xgb = xgb_cls.predict(X_test)
y_pred_lgb = lgb_cls.predict(X_test)

print('RandomForest accuracy:{}'.format(accuracy_score(y_test, y_pred_rf)))
print('GB accuracy:{}'.format(accuracy_score(y_test, y_pred_gb)))
print('XGB accuracy:{}'.format(accuracy_score(y_test, y_pred_xgb)))
print('LGB accuracy:{}'.format(accuracy_score(y_test, y_pred_lgb)))