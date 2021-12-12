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


from sklearn.utils import all_estimators
#. from sklearn.utils.testing import all_estimators
import warnings
warnings.filterwarnings('ignore')

# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms) #. 각종 모델이 들어가 있음 
print('모델의 개수 : ', len(allAlgorithms)) # 모델의 개수 :  54

cnt = 0
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name, '의 r2_score : ', r2z)
        cnt += 1
        print(cnt)

    except:
        print(name, '은 없는놈!!!')
'''
모델의 개수 :  54
ARDRegression 의 r2_score :  0.6002025719805103
1
AdaBoostRegressor 의 r2_score :  0.4748970615462259
2
BaggingRegressor 의 r2_score :  0.4729675080078368
3
BayesianRidge 의 r2_score :  0.6031317857525211
4
CCA 의 r2_score :  0.5825461541443031
5
DecisionTreeRegressor 의 r2_score :  0.0356210738275905
6
DummyRegressor 의 r2_score :  -0.010307682180093813
7
ElasticNet 의 r2_score :  0.14911424482260882
8
ElasticNetCV 의 r2_score :  0.5917708881624744
9
ExtraTreeRegressor 의 r2_score :  0.02302913919641747
10
ExtraTreesRegressor 의 r2_score :  0.5029662235469275
11
GammaRegressor 의 r2_score :  0.0920059553784578
12
GaussianProcessRegressor 의 r2_score :  -12.368912123846886
13
GradientBoostingRegressor 의 r2_score :  0.5033324747309343
14
HistGradientBoostingRegressor 의 r2_score :  0.4799971728301524
15
HuberRegressor 의 r2_score :  0.5864804455211765
16
IsotonicRegression 은 없는놈!!!
KNeighborsRegressor 의 r2_score :  0.4853128420823485
17
KernelRidge 의 r2_score :  0.5974945257701708
18
Lars 의 r2_score :  0.5900352656383722
19
LarsCV 의 r2_score :  0.6007838659835665
20
Lasso 의 r2_score :  0.5796130592301109
21
LassoCV 의 r2_score :  0.6010792177557813
22
LassoLars 의 r2_score :  0.45468470967039465
23
LassoLarsCV 의 r2_score :  0.6007838659835665
24
LassoLarsIC 의 r2_score :  0.602764944184214
25
LinearRegression 의 r2_score :  0.5900352656383736
26
LinearSVR 의 r2_score :  0.27909545802111835
27
MLPRegressor 의 r2_score :  -0.63116465506543
28
MultiOutputRegressor 은 없는놈!!!
MultiTaskElasticNet 은 없는놈!!!
MultiTaskElasticNetCV 은 없는놈!!!
MultiTaskLasso 은 없는놈!!!
MultiTaskLassoCV 은 없는놈!!!
NuSVR 의 r2_score :  0.1444650616814751
29
OrthogonalMatchingPursuit 의 r2_score :  0.3443972776662052
30
OrthogonalMatchingPursuitCV 의 r2_score :  0.5950203281004389
31
PLSCanonical 의 r2_score :  -1.256553805294621
32
PLSRegression 의 r2_score :  0.610470552100115
33
PassiveAggressiveRegressor 의 r2_score :  0.5859490132681288
34
PoissonRegressor 의 r2_score :  0.591222726542848
35
RANSACRegressor 의 r2_score :  0.19359763175072708
36
RadiusNeighborsRegressor 의 r2_score :  0.19209707227744155
37
RandomForestRegressor 의 r2_score :  0.5066598575693495
38
RegressorChain 은 없는놈!!!
Ridge 의 r2_score :  0.6019449568140334
39
RidgeCV 의 r2_score :  0.6007676294745006
40
SGDRegressor 의 r2_score :  0.5996793849088287
41
SVR 의 r2_score :  0.16486638677450483
42
StackingRegressor 은 없는놈!!!
TheilSenRegressor 의 r2_score :  0.5974172635293409
43
TransformedTargetRegressor 의 r2_score :  0.5900352656383736
44
TweedieRegressor 의 r2_score :  0.08886348560205204
45
VotingRegressor 은 없는놈!!!

'''

# # 모델
# # model = LinearSVC()
# #. ValueError: Unknown label type: 'continuous'
# # model = SVC()
# #. ValueError: Unknown label type: 'continuous'
# # model = KNeighborsClassifier()
# #. ValueError: Unknown label type: 'continuous'
# # model = KNeighborsRegressor()
# #. model_score :  0.4853128420823485`
# # model = DecisionTreeRegressor()
# #. model_score :  0.8133700013379184
# # model = RandomForestRegressor()
# #. model_score :  0.5000730349312725
# model = LinearRegression()
# #. model_score :  0.5900352656383736



# # 컴파일, 훈련
# # 컴파일 훈련
# model.fit(x_train, y_train)

# #4. 평가, 예측
# y_predict = model.predict(x_test)

# results = model.score(x_test, y_test)
# print('model_score : ', results)

# # acc = accuracy_score(y_test, y_predict)
# # print("acc_score : ", acc)
