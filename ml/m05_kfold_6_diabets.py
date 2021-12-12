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


#@ 여기가 새로운거
from sklearn.model_selection import train_test_split, KFold, cross_val_score #. train test split과 함께 사용 
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.8, shuffle=True, random_state=66) 

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66) # n_splites=5 = test 데이터가 20프로?

# # 모델구성
from sklearn.svm import LinearSVC, SVC #. 레거시한 머신러닝 기법은 대부분 sklrean에 있음 support vector machine
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor           #? 리그레서 클래스파이어 차이 회귀 vs 분류
from sklearn.linear_model import LogisticRegression , LinearRegression#. 이름에서 낚시 분류모델임
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #. 기본은 tree구조 tree가 여러개 모여 앙상블을 이룸 위보다 성능이 좋음


# model = LinearSVC() #. 모델의 정의만 해주면 됨 중요한 모델의 파라미터는 알필요가 있지만 그 외는 상관 x 디폴트도 성능이 뛰어나기 때문에
                    #. 기본적으로 머신러닝은 1차원을 받아들이기 때문에 y가 2차원이상의 데이터는 못돌림 but reshape하면됨
'''
Acc :  [0.96666667 0.96666667 1.         0.9        1.        ] 평균값: 0.9667
'''
# 모델

model = KNeighborsRegressor()
#. R2 :  [0.37000683 0.35477108 0.32086338 0.51614896 0.41040527] 평균값: 0.3944
# model = DecisionTreeRegressor()
#. R2 :  [ 0.0430096   0.07021788 -0.06543745  0.23887087 -0.13667004] 평균값: 0.03
# model = RandomForestRegressor()
#. R2 :  [0.4832973  0.53747199 0.40221453 0.56563129 0.43670467] 평균값: 0.4851
# model = LinearRegression()
#. R2 :  [0.53550031 0.49362737 0.47105167 0.55090349 0.36810479] 평균값: 0.4838

# 컴파일 훈련
# 평가, 예측
#. fit에서 스코어 까지 끝냄
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('R2 : ', scores, '평균값:', round(np.mean(scores),4))

# [0.96666667 0.96666667 1.         0.9        1.        ] # 분류이기때문에 acc 5회의 교차검증 결과

