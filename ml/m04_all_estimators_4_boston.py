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

print(x.shape) # (506, 13)
print(y.shape) # (506,)

print(datasets.feature_names)
print(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=66) 

# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
import tensorflow as tf
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators
#. from sklearn.utils.testing import all_estimators
import warnings
warnings.filterwarnings('ignore')

# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms) #. 각종 모델이 들어가 있음 
print('모델의 개수 : ', len(allAlgorithms)) # 모델의 개수 : 54

cnt = 0
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name, '의 r2_score : ', r2)
        cnt += 1
        print(cnt)

    except:
        print(name, '은 없는놈!!!')

'''
모델의 개수 :  54
ARDRegression 의 r2_score :  0.805568340492246
1
AdaBoostRegressor 의 r2_score :  0.8354334873346867
2
BaggingRegressor 의 r2_score :  0.8786185819275211
3
BayesianRidge 의 r2_score :  0.814449232995407
4
CCA 의 r2_score :  0.8137619833391814
5
DecisionTreeRegressor 의 r2_score :  0.7051282728985091
6
DummyRegressor 의 r2_score :  -0.005227869326375867
7
ElasticNet 의 r2_score :  0.7057404845560893
8
ElasticNetCV 의 r2_score :  0.8139701792949572
9
ExtraTreeRegressor 의 r2_score :  0.4959212363258143
10
ExtraTreesRegressor 의 r2_score :  0.8966733508114277
11
GammaRegressor 의 r2_score :  0.6985799136926647
12
GaussianProcessRegressor 의 r2_score :  0.4219746956654109
13
GradientBoostingRegressor 의 r2_score :  0.9046826926034658
14
HistGradientBoostingRegressor 의 r2_score :  0.8912297161279814
15
HuberRegressor 의 r2_score :  0.8061950010958285
16
IsotonicRegression 은 없는놈!!!
KNeighborsRegressor 의 r2_score :  0.8407834418231728
17
KernelRidge 의 r2_score :  0.8141152393199854
18
Lars 의 r2_score :  0.8133700013379181
19
LarsCV 의 r2_score :  0.8133700013379181
20
Lasso 의 r2_score :  0.7298147438566898
21
LassoCV 의 r2_score :  0.8133113325336573
22
LassoLars 의 r2_score :  -0.005227869326375867
23
LassoLarsCV 의 r2_score :  0.8133700013379181
24
LassoLarsIC 의 r2_score :  0.8040408681706304
25
LinearRegression 의 r2_score :  0.8133700013379184
26
LinearSVR 의 r2_score :  0.7916250703596385
27
MLPRegressor 의 r2_score :  0.6374183819600621
28
MultiOutputRegressor 은 없는놈!!!
MultiTaskElasticNet 은 없는놈!!!
MultiTaskElasticNetCV 은 없는놈!!!
MultiTaskLasso 은 없는놈!!!
MultiTaskLassoCV 은 없는놈!!!
NuSVR 의 r2_score :  0.7260812601451674
29
OrthogonalMatchingPursuit 의 r2_score :  0.6782976489881782
30
OrthogonalMatchingPursuitCV 의 r2_score :  0.7813855082466543
31
PLSCanonical 의 r2_score :  -2.1712219721817014
32
PLSRegression 의 r2_score :  0.784328686774908
33
PassiveAggressiveRegressor 의 r2_score :  0.24964488466116197
34
PoissonRegressor 의 r2_score :  0.8478018666828309
35
RANSACRegressor 의 r2_score :  0.648537972862776
36
RadiusNeighborsRegressor 은 없는놈!!!
RandomForestRegressor 의 r2_score :  0.8859855221051003
37
RegressorChain 은 없는놈!!!
Ridge 의 r2_score :  0.8136989148169409
38
RidgeCV 의 r2_score :  0.8136989148169831
39
SGDRegressor 의 r2_score :  0.815880753377383
40
SVR 의 r2_score :  0.7381239340692907
41
StackingRegressor 은 없는놈!!!
TheilSenRegressor 의 r2_score :  0.764584892304805
42
TransformedTargetRegressor 의 r2_score :  0.8133700013379184
43
TweedieRegressor 의 r2_score :  0.6867302204779187
44
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
# #. model_score :  0.8407834418231728
# # model = DecisionTreeRegressor()
# #. model_score :  0.7194699576276645
# # model = RandomForestRegressor()
# #. model_score :  0.8847107406342373
# model = LinearRegression()
# #. model_score :  0.8133700013379184



# # 컴파일 훈련
# model.fit(x_train, y_train)

# #4. 평가, 예측
# y_predict = model.predict(x_test)

# results = model.score(x_test, y_test)
# print('model_score : ', results)

# # acc = accuracy_score(y_test, y_predict)
# # print("acc_score : ", acc)


