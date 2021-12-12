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

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) #테스트 데이터는 트레인 데이터에 관여하면안된다.
x_test = scaler.transform(x_test)


# print(x_train.shape) 
# print(x_test.shape)

from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
import tensorflow as tf
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators
#. from sklearn.utils.testing import all_estimators
import warnings
warnings.filterwarnings('ignore')

allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms) #. 각종 모델이 들어가 있음 
print('모델의 개수 : ', len(allAlgorithms)) # 모델의 개수 :  41

cnt = 0
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
        cnt += 1
        print(cnt)

    except:
        print(name, '은 없는놈!!!')

'''
모델의 개수 :  41
AdaBoostClassifier 의 정답률 :  0.9532163742690059
1
BaggingClassifier 의 정답률 :  0.9473684210526315
2
BernoulliNB 의 정답률 :  0.9415204678362573
3
CalibratedClassifierCV 의 정답률 :  0.9649122807017544
4
CategoricalNB 은 없는놈!!!
ClassifierChain 은 없는놈!!!
ComplementNB 은 없는놈!!!
DecisionTreeClassifier 의 정답률 :  0.9415204678362573
5
DummyClassifier 의 정답률 :  0.6432748538011696
6
ExtraTreeClassifier 의 정답률 :  0.9239766081871345
7
ExtraTreesClassifier 의 정답률 :  0.9649122807017544
8
GaussianNB 의 정답률 :  0.9473684210526315
9
GaussianProcessClassifier 의 정답률 :  0.9649122807017544
10
GradientBoostingClassifier 의 정답률 :  0.9649122807017544
11
HistGradientBoostingClassifier 의 정답률 :  0.9707602339181286
12
KNeighborsClassifier 의 정답률 :  0.9590643274853801
13
LabelPropagation 의 정답률 :  0.9473684210526315
14
LabelSpreading 의 정답률 :  0.9473684210526315
15
LinearDiscriminantAnalysis 의 정답률 :  0.9649122807017544
16
LinearSVC 의 정답률 :  0.9766081871345029
17
LogisticRegression 의 정답률 :  0.9824561403508771
18
LogisticRegressionCV 의 정답률 :  0.9883040935672515
19
MLPClassifier 의 정답률 :  0.9766081871345029
20
MultiOutputClassifier 은 없는놈!!!
MultinomialNB 은 없는놈!!!
NearestCentroid 의 정답률 :  0.9415204678362573
21
NuSVC 의 정답률 :  0.9473684210526315
22
OneVsOneClassifier 은 없는놈!!!
OneVsRestClassifier 은 없는놈!!!
OutputCodeClassifier 은 없는놈!!!
PassiveAggressiveClassifier 의 정답률 :  0.9766081871345029
23
Perceptron 의 정답률 :  0.9824561403508771
24
QuadraticDiscriminantAnalysis 의 정답률 :  0.9473684210526315
25
RadiusNeighborsClassifier 은 없는놈!!!
RandomForestClassifier 의 정답률 :  0.9707602339181286
26
RidgeClassifier 의 정답률 :  0.9590643274853801
27
RidgeClassifierCV 의 정답률 :  0.9590643274853801
28
SGDClassifier 의 정답률 :  0.8771929824561403
29
SVC 의 정답률 :  0.9766081871345029
30
StackingClassifier 은 없는놈!!!
VotingClassifier 은 없는놈!!!
'''
# # 모델구성
# # model = LinearSVC()
# #. acc_score :  0.9766081871345029
# model = SVC()
# #. acc_score :  0.9590643274853801
# # model = KNeighborsClassifier()
# #. acc_score :  0.9590643274853801
# # model = LogisticRegression()
# #. acc_score :  0.9824561403508771
# # model = DecisionTreeClassifier()
# #. acc_score :  0.9532163742690059
# model = RandomForestClassifier()
# #. acc_score :  0.9649122807017544



# #3 훈련
# model.fit(x_train, y_train)

# #4. 평가, 예측
# y_predict = model.predict(x_test)

# results = model.score(x_test, y_test)
# print('model_score : ', results)

# acc = accuracy_score(y_test, y_predict)
# print("acc_score : ", acc)


# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1) # patience는 epoches 기준으로 설정, mode #loss 에서 val_los로 바꿈 // 너무 빨리 끝나면 patience를 조절하자
# hist = model.fit(x_train,y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])