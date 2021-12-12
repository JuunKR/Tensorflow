# grid 격자 #@.다른사람꺼

# 테이스 데이이터를 활용해 훈련을 함 / 데스트 데이터를 버리지 않음. 
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

from sklearn.model_selection import train_test_split, KFold, cross_val_score #. train test split과 함께 사용 
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.8, shuffle=True, random_state=66) 


from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.pipeline import make_pipeline, Pipeline



    

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV #. 새로 하는 것
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66) # n_splites=5 = test 데이터가 20프로?

# # 모델구성
from sklearn.svm import LinearSVC, SVC #. 레거시한 머신러닝 기법은 대부분 sklrean에 있음 support vector machine
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor           #? 리그레서 클래스파이어 차이 회귀 vs 분류
from sklearn.linear_model import LogisticRegression #. 이름에서 낚시 분류모델임
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #. 기본은 tree구조 tree가 여러개 모여 앙상블을 이룸 위보다 성능이 좋음
from sklearn.metrics import accuracy_score
#@ 여기가 달라요
pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())



# model = LinearSVC() #. 모델의 정의만 해주면 됨 중요한 모델의 파라미터는 알필요가 있지만 그 외는 상관 x 디폴트도 성능이 뛰어나기 때문에
                    #. 기본적으로 머신러닝은 1차원을 받아들이기 때문에 y가 2차원이상의 데이터는 못돌림 but reshape하면됨
'''
Acc :  [0.96666667 0.96666667 1.         0.9        1.        ] 평균값: 0.9667
'''
parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"]},
    {"C":[1,10,100], "kernel": ['rbf'], 'gamma':[0.001, 0.0001]},
    {"C":[1,10,100,1000], "kernel": ["sigmoid"], "gamma": [0.001, 0.0001]}
]



# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1) # Fitting 5 folds for each of 18 candidates, totalling 90 fits vervose 몇번 돌았는지 확인
#@ 여기가 달라요
model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=1) # Fitting 5 folds for each of 10 candidates, totalling 50 fits /크로스 5번 파라미터 10번을 해서 50번 

# 컴파일 훈련
model.fit(x_train,y_train)

# 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)  # 최적의 매개변수 :  SVC(C=1, kernel='linear')
print('best_score_ : ', model.best_score_) # best_score_ : best_score_ :  0.9916666666666668 #. cross validation 에서 나옴

print("model.score : ", model.score(x_test, y_test)) # model.score :  0.9666666666666667 #. 아래와 한통속

y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict)) # accuracy_score :  0.9666666666666667

# #. fit에서 스코어 까지 끝냄
# scores = cross_val_score(model, x, y, cv=kfold)
# print('Acc : ', scores, '평균값:', round(np.mean(scores),4))

# [0.96666667 0.96666667 1.         0.9        1.        ] # 분류이기때문에 acc 5회의 교차검증 결과



