# 희귀 데이터를 classifier로 만들었을 경우의 에러 확인

#
# 실습 diabets
# 1. loss 와 R2로 평가
# MinMax와 Standard 결과를 명시
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# 데이터 전처리
datasets = load_diabetes()
x = datasets.data
y = datasets.target

    
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=9) 



scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

print(x_train.shape) 
print(x_test.shape)
'''
(309, 10)
(133, 10)

'''
from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score

# 모델
# model = LinearSVC()
#. ValueError: Unknown label type: 'continuous'
# model = SVC()
#. ValueError: Unknown label type: 'continuous'
# model = KNeighborsClassifier()
#. ValueError: Unknown label type: 'continuous'
# model = KNeighborsRegressor()
#. model_score :  0.4853128420823485`
# model = DecisionTreeRegressor()
#. model_score :  0.8133700013379184
# model = RandomForestRegressor()
#. model_score :  0.5000730349312725
model = LinearRegression()
#. model_score :  0.5900352656383736



# 컴파일, 훈련
# 컴파일 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

results = model.score(x_test, y_test)
print('model_score : ', results)

r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)
