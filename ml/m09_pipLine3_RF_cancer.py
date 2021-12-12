# 2진분류
from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 데이터 전처리
datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (569, 30) (569,)

# print(y[:20])
# print(np.unique(y)) # [0 1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=66) 

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train) #테스트 데이터는 트레인 데이터에 관여하면안된다.
# x_test = scaler.transform(x_test)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.pipeline import make_pipeline, Pipeline


# print(x_train.shape) 
# print(x_test.shape)

from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
import tensorflow as tf
from sklearn.metrics import accuracy_score
model = make_pipeline(MinMaxScaler() , SVC())
# 모델구성
# model = LinearSVC()
#. acc_score :  0.9766081871345029
# model = SVC()
#. acc_score :  0.9590643274853801
# model = KNeighborsClassifier()
#. acc_score :  0.9590643274853801
# model = LogisticRegression()
#. acc_score :  0.9824561403508771
# model = DecisionTreeClassifier()
#. acc_score :  0.9532163742690059
model = RandomForestClassifier()
#. acc_score :  0.9649122807017544



#3 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

results = model.score(x_test, y_test)
print('model_score : ', results)

acc = accuracy_score(y_test, y_predict)
print("acc_score : ", acc)


# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1) # patience는 epoches 기준으로 설정, mode #loss 에서 val_los로 바꿈 // 너무 빨리 끝나면 patience를 조절하자
# hist = model.fit(x_train,y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])