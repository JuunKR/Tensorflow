# 희귀 데이터를 classifier로 만들었을 경우의 에러 확인

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

# 데이터 전처리
datasets = load_boston()
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
from sklearn.linear_model import LogisticRegression , LinearRegression #. 이름에서 낚시 분류모델임
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #. 기본은 tree구조 tree가 여러개 모여 앙상블을 이룸 위보다 성능이 좋음


# model = LinearSVC() #. 모델의 정의만 해주면 됨 중요한 모델의 파라미터는 알필요가 있지만 그 외는 상관 x 디폴트도 성능이 뛰어나기 때문에
                    #. 기본적으로 머신러닝은 1차원을 받아들이기 때문에 y가 2차원이상의 데이터는 못돌림 but reshape하면됨
'''
Acc :  [0.96666667 0.96666667 1.         0.9        1.        ] 평균값: 0.9667
'''
model = KNeighborsRegressor()
#. R2 :  [0.38689566 0.52994483 0.3434155  0.55325748 0.51995804] 평균값: 0.4667
# model = DecisionTreeRegressor()
#. R2 :  [0.70893284 0.62354775 0.58210513 0.70429978 0.77729226] 평균값: 0.6792
# model = RandomForestRegressor()
#. R2 :  [0.87201279 0.73169979 0.78878149 0.86208124 0.88328909] 평균값: 0.8276
# model = LinearRegression()
#. R2 :  [0.5815212  0.69885237 0.6537276  0.77449543 0.70223459] 평균값: 0.6822

# 컴파일 훈련
# 평가, 예측
import numpy as np
#. fit에서 스코어 까지 끝냄
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('R2 : ', scores, '평균값:', round(np.mean(scores),4))

# [0.96666667 0.96666667 1.         0.9        1.        ] # 분류이기때문에 acc 5회의 교차검증 결과