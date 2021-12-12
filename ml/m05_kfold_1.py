# 테스트 데이이터를 활용해 훈련을 함 / 데스트 데이터를 버리지 않음. 
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

datasets = load_iris()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66) # n_splites=5 = test 데이터가 20프로?

# # 모델구성
from sklearn.svm import LinearSVC, SVC #. 레거시한 머신러닝 기법은 대부분 sklrean에 있음 support vector machine
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor           #? 리그레서 클래스파이어 차이 회귀 vs 분류
from sklearn.linear_model import LogisticRegression #. 이름에서 낚시 분류모델임
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #. 기본은 tree구조 tree가 여러개 모여 앙상블을 이룸 위보다 성능이 좋음


# model = LinearSVC() #. 모델의 정의만 해주면 됨 중요한 모델의 파라미터는 알필요가 있지만 그 외는 상관 x 디폴트도 성능이 뛰어나기 때문에
                    #. 기본적으로 머신러닝은 1차원을 받아들이기 때문에 y가 2차원이상의 데이터는 못돌림 but reshape하면됨
'''
Acc :  [0.96666667 0.96666667 1.         0.9        1.        ] 평균값: 0.9667
'''
# model = SVC()
# Acc :  [0.96666667 0.96666667 1.         0.93333333 0.96666667] 평균값: 0.9667
# model = KNeighborsClassifier()
# Acc :  [0.96666667 0.96666667 1.         0.9        0.96666667] 평균값: 0.96
# model = LogisticRegression()
# Acc :  [1.         0.96666667 1.         0.9        0.96666667] 평균값: 0.9667
# model = DecisionTreeClassifier()
# Acc :  [0.93333333 0.96666667 1.         0.9        0.93333333] 평균값: 0.9467
model = RandomForestClassifier()
# Acc :  [0.96666667 0.96666667 1.         0.9        0.96666667] 평균값: 0.96

# 컴파일 훈련
# 평가, 예측
#. fit에서 스코어 까지 끝냄
scores = cross_val_score(model, x, y, cv=kfold)
print('Acc : ', scores, '평균값:', round(np.mean(scores),4))

# [0.96666667 0.96666667 1.         0.9        1.        ] # 분류이기때문에 acc 5회의 교차검증 결과



