from csv import excel_tab
from sklearn.metrics import accuracy_score
from os import name
from sklearn.utils import all_estimators
#. from sklearn.utils.testing import all_estimators


import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150,)
print(y)


#. 대부분의 머신러닝은 원핫 인코딩은 통상적으로 안해도됨 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=66) 


# # 모델구성
from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor          
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 


from csv import excel_tab
from sklearn.metrics import accuracy_score
from os import name


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
# 34개 돌아감 


'''
모델의 개수 :  41
AdaBoostClassifier 의 정답률 :  0.9111111111111111
1
BaggingClassifier 의 정답률 :  0.8888888888888888
2
BernoulliNB 의 정답률 :  0.28888888888888886
3
CalibratedClassifierCV 의 정답률 :  0.8444444444444444
4
CategoricalNB 의 정답률 :  0.9111111111111111
5
ComplementNB 의 정답률 :  0.6222222222222222
6
DecisionTreeClassifier 의 정답률 :  0.9111111111111111
7
DummyClassifier 의 정답률 :  0.28888888888888886
8
ExtraTreeClassifier 의 정답률 :  0.9333333333333333
9
ExtraTreesClassifier 의 정답률 :  0.9555555555555556
10
GaussianNB 의 정답률 :  0.9555555555555556
11
GaussianProcessClassifier 의 정답률 :  0.9333333333333333
12
GradientBoostingClassifier 의 정답률 :  0.8888888888888888
13
HistGradientBoostingClassifier 의 정답률 :  0.9111111111111111
14
KNeighborsClassifier 의 정답률 :  0.9555555555555556
15
LabelPropagation 의 정답률 :  0.9333333333333333
16
LabelSpreading 의 정답률 :  0.9333333333333333
17
LinearDiscriminantAnalysis 의 정답률 :  1.0
18
LinearSVC 의 정답률 :  0.9555555555555556
19
LogisticRegression 의 정답률 :  0.9777777777777777
20
LogisticRegressionCV 의 정답률 :  0.9777777777777777
21
MLPClassifier 의 정답률 :  0.9555555555555556
22
MultinomialNB 의 정답률 :  0.7555555555555555
23
24
NuSVC 의 정답률 :  0.9777777777777777
25
PassiveAggressiveClassifier 의 정답률 :  0.6888888888888889
26
Perceptron 의 정답률 :  0.6222222222222222
27
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
28
RadiusNeighborsClassifier 의 정답률 :  0.9333333333333333
29
RandomForestClassifier 의 정답률 :  0.9111111111111111
30
RidgeClassifier 의 정답률 :  0.8222222222222222
31
RidgeClassifierCV 의 정답률 :  0.8222222222222222
32
SGDClassifier 의 정답률 :  0.6666666666666666
33
SVC 의 정답률 :  0.9777777777777777
34
'''
#. 스케일러에 따라 결과가 다르다/ 나는 여기서 스케일러 사용안했오 
    

# model = LinearSVC() #. 모델의 정의만 해주면 됨 중요한 모델의 파라미터는 알필요가 있지만 그 외는 상관 x 디폴트도 성능이 뛰어나기 때문에
                    #. 기본적으로 머신러닝은 1차원을 받아들이기 때문에 y가 2차원이상의 데이터는 못돌림 but reshape하면됨
#. acc_score :  0.9555555555555556
# model = SVC()
#. acc_score :  0.9777777777777777
# model = KNeighborsClassifier()
#. acc_score :  0.9555555555555556
# model = LogisticRegression()
#. acc_score :  0.9777777777777777
# model = DecisionTreeClassifier()
#. acc_score :  0.9111111111111111
#model = RandomForestClassifier()
#. acc_score :  0.9111111111111111




# model = Sequential()
# model.add(Dense(128,activation='relu', input_shape=(4,)))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(3, activation='softmax')) # 다중 분류의 라벨의 수가 3개 





# 컴파일 훈련


# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 2진분류를 위한
 
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1) # patience는 epoches 기준으로 설정, mode #loss 에서 val_los로 바꿈 // 너무 빨리 끝나면 patience를 조절하자
# hist = model.fit(x_train,y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])

#평가, 예측
# results = model.score(x_test, y_test)  #. evaluate = score
# print('model_score : ', results)
# #. acc 와 같음 0.9555555555555556

# from sklearn.metrics import r2_score, accuracy_score #. R2 는 회귀에 대한 스코어, acc는 분류에 대한 스코어
# y_prdecit = model.predict(x_test)
# acc = accuracy_score(y_test, y_prdecit)
# print('acc_score : ', acc)
# #. acc_score :  0.9555555555555556 모델 스코어와 acc스코어가 같다. 

# y_prdecit2 = model.predict(x_test[:5])
# print(y_prdecit)
# #. [1 1 1 0 1]

