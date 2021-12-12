# 2진분류
from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from csv import excel_tab
from sklearn.metrics import accuracy_score
from os import name
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold, cross_val_score

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


# # 모델구성
from sklearn.svm import LinearSVC, SVC #. 레거시한 머신러닝 기법은 대부분 sklrean에 있음 support vector machine
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor           #? 리그레서 클래스파이어 차이 회귀 vs 분류
from sklearn.linear_model import LogisticRegression #. 이름에서 낚시 분류모델임
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #. 기본은 tree구조 tree가 여러개 모여 앙상블을 이룸 위보다 성능이 좋음


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

kfold = KFold(n_splits=5, shuffle=True, random_state=66)

cnt = 0
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        scores = cross_val_score(model, x, y, cv=kfold)
        print(name, scores,'평균 : ', round(np.mean(scores), 4))
        # print(cnt)

    except:
        print(name, '은 없는놈!!!')
    

'''
모델의 개수 :  41
AdaBoostClassifier [0.94736842 0.99122807 0.94736842 0.96491228 0.97345133] 평균 :  0.9649
BaggingClassifier [0.93859649 0.94736842 0.93859649 0.92982456 0.96460177] 평균 :  0.9438
BernoulliNB [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 평균 :  0.6274
CalibratedClassifierCV [0.89473684 0.93859649 0.89473684 0.92982456 0.97345133] 평균 :  0.9263
CategoricalNB [nan nan nan nan nan] 평균 :  nan
ClassifierChain 은 없는놈!!!
ComplementNB [0.86842105 0.92982456 0.87719298 0.9122807  0.89380531] 평균 :  0.8963
DecisionTreeClassifier [0.92982456 0.95614035 0.92105263 0.89473684 0.95575221] 평균 :  0.9315
DummyClassifier [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 평균 :  0.6274
ExtraTreeClassifier [0.86842105 0.93859649 0.90350877 0.92982456 0.92920354] 평균 :  0.9139
ExtraTreesClassifier [0.95614035 0.98245614 0.93859649 0.94736842 0.99115044] 평균 :  0.9631
GaussianNB [0.93859649 0.96491228 0.9122807  0.93859649 0.95575221] 평균 :  0.942
GaussianProcessClassifier [0.87719298 0.89473684 0.89473684 0.94736842 0.94690265] 평균 :  0.9122
GradientBoostingClassifier [0.94736842 0.96491228 0.95614035 0.94736842 0.98230088] 평균 :  0.9596
HistGradientBoostingClassifier [0.97368421 0.98245614 0.96491228 0.96491228 0.98230088] 평균 :  0.9737
KNeighborsClassifier [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221] 평균 :  0.928
LabelPropagation [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 평균 :  0.3902
LabelSpreading [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 평균 :  0.3902
LinearDiscriminantAnalysis [0.94736842 0.98245614 0.94736842 0.95614035 0.97345133] 평균 :  0.9614
LinearSVC [0.80701754 0.92105263 0.85087719 0.9122807  0.97345133] 평균 :  0.8929
LogisticRegression [0.94736842 0.95614035 0.88596491 0.96491228 0.96460177] 평균 :  0.9438
LogisticRegressionCV [0.95614035 0.96491228 0.9122807  0.96491228 0.96460177] 평균 :  0.9526
MLPClassifier [0.90350877 0.92982456 0.92105263 0.9122807  0.96460177] 평균 :  0.9263
MultiOutputClassifier 은 없는놈!!!
MultinomialNB [0.85964912 0.92105263 0.87719298 0.9122807  0.89380531] 평균 :  0.8928
NearestCentroid [0.86842105 0.89473684 0.85964912 0.9122807  0.91150442] 평균 :  0.8893
NuSVC [0.85964912 0.9122807  0.83333333 0.87719298 0.88495575] 평균 :  0.8735
OneVsOneClassifier 은 없는놈!!!
OneVsRestClassifier 은 없는놈!!!
OutputCodeClassifier 은 없는놈!!!
PassiveAggressiveClassifier [0.90350877 0.92105263 0.87719298 0.77192982 0.92920354] 평균 :  0.8806
Perceptron [0.40350877 0.80701754 0.85964912 0.86842105 0.94690265] 평균 :  0.7771
QuadraticDiscriminantAnalysis [0.93859649 0.95614035 0.93859649 0.98245614 0.94690265] 평균 :  0.9525
RadiusNeighborsClassifier [nan nan nan nan nan] 평균 :  nan
RandomForestClassifier [0.96491228 0.95614035 0.94736842 0.95614035 0.98230088] 평균 :  0.9614
RidgeClassifier [0.95614035 0.98245614 0.92105263 0.95614035 0.95575221] 평균 :  0.9543
RidgeClassifierCV [0.94736842 0.97368421 0.93859649 0.95614035 0.96460177] 평균 :  0.9561
SGDClassifier [0.84210526 0.90350877 0.86842105 0.90350877 0.5840708 ] 평균 :  0.8203
SVC [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177] 평균 :  0.921
StackingClassifier 은 없는놈!!!
VotingClassifier 은 없는놈!!!
'''