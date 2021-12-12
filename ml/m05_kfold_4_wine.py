import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
#./ : 현재폴더
#../ : 상위폴더


datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                    index_col=None, header=0)

# # 데이터 전처리
datasets = datasets.to_numpy()
x = datasets[:,:-1] 
y = datasets[ : , -1:]


#@ 여기가 새로운거
from sklearn.model_selection import train_test_split, KFold, cross_val_score #. train test split과 함께 사용 
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.8, shuffle=True, random_state=66) 

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
# Acc :  [0.45918367 0.39795918 0.48979592 0.47959184 0.43367347] 평균값: 0.452
# model = KNeighborsClassifier()
# Acc :  [0.45408163 0.41326531 0.38265306 0.44897959 0.51530612] 평균값: 0.4429
# model = LogisticRegression()
# Acc :  [0.4744898  0.45918367 0.5        0.43877551 0.44897959] 평균값: 0.4643
# model = DecisionTreeClassifier()
# Acc :  [0.44897959 0.46428571 0.5255102  0.46938776 0.52040816] 평균값: 0.4857
model = RandomForestClassifier()
# Acc :  [0.59693878 0.56122449 0.57142857 0.57142857 0.60714286] 평균값: 0.5816

# 컴파일 훈련
# 평가, 예측
#. fit에서 스코어 까지 끝냄
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('Acc : ', scores, '평균값:', round(np.mean(scores),4))

# [0.96666667 0.96666667 1.         0.9        1.        ] # 분류이기때문에 acc 5회의 교차검증 결과
