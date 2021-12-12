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
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150,)
print(y)


#. 대부분의 머신러닝은 원핫 인코딩은 통상적으로 안해도됨 
# from tensorflow.keras.utils import to_categorical #얘는 0부터 시작함 만약 3부터 시작하면 0123을 채워 shape 크기를 늘림 : wind2 확인

# y = to_categorical(y) #원핫인풋 // 수치에 대한 라벨링을 하는 작업 남자는 여자의 두배의 가치가 잇는게 아니다 그냥 단지 라벨을 통해 1 2로 표현할 뿐


# print(y[:5])
# print(y.shape) # (150, 3)


# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x,y,
    # train_size=0.7, shuffle=True, random_state=66) 



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
# 34개 돌아감 
'''
모델의 개수 :  41
AdaBoostClassifier [0.63333333 0.93333333 1.         0.9        0.96666667] 평균 :  0.8867
BaggingClassifier [0.93333333 0.96666667 1.         0.9        0.96666667] 평균 :  0.9533
BernoulliNB [0.3        0.33333333 0.3        0.23333333 0.3       ] 평균 :  0.2933
CalibratedClassifierCV [0.9        0.83333333 1.         0.86666667 0.96666667] 평균 :  0.9133
CategoricalNB [0.9        0.93333333 0.93333333 0.9        1.        ] 평균 :  0.9333
ClassifierChain 은 없는놈!!!
ComplementNB [0.66666667 0.66666667 0.7        0.6        0.7       ] 평균 :  0.6667
DecisionTreeClassifier [0.96666667 0.96666667 1.         0.9        0.93333333] 평균 :  0.9533
DummyClassifier [0.3        0.33333333 0.3        0.23333333 0.3       ] 평균 :  0.2933
ExtraTreeClassifier [0.76666667 1.         0.96666667 0.9        1.        ] 평균 :  0.9267
ExtraTreesClassifier [0.93333333 0.96666667 1.         0.86666667 0.96666667] 평균 :  0.9467
GaussianNB [0.96666667 0.9        1.         0.9        0.96666667] 평균 :  0.9467
GaussianProcessClassifier [0.96666667 0.96666667 1.         0.9        0.96666667] 평균 :  0.96
GradientBoostingClassifier [0.93333333 0.96666667 1.         0.93333333 0.96666667] 평균 :  0.96
HistGradientBoostingClassifier [0.86666667 0.96666667 1.         0.9        0.96666667] 평균 :  0.94
KNeighborsClassifier [0.96666667 0.96666667 1.         0.9        0.96666667] 평균 :  0.96
LabelPropagation [0.93333333 1.         1.         0.9        0.96666667] 평균 :  0.96
LabelSpreading [0.93333333 1.         1.         0.9        0.96666667] 평균 :  0.96
LinearDiscriminantAnalysis [1.  1.  1.  0.9 1. ] 평균 :  0.98
LinearSVC [0.96666667 0.96666667 1.         0.9        1.        ] 평균 :  0.9667
LogisticRegression [1.         0.96666667 1.         0.9        0.96666667] 평균 :  0.9667
LogisticRegressionCV [1.         0.96666667 1.         0.9        1.        ] 평균 :  0.9733
MLPClassifier [0.96666667 0.96666667 1.         0.93333333 1.        ] 평균 :  0.9733
MultiOutputClassifier 은 없는놈!!!
MultinomialNB [0.96666667 0.93333333 1.         0.93333333 1.        ] 평균 :  0.9667
NearestCentroid [0.93333333 0.9        0.96666667 0.9        0.96666667] 평균 :  0.9333
NuSVC [0.96666667 0.96666667 1.         0.93333333 1.        ] 평균 :  0.9733
OneVsOneClassifier 은 없는놈!!!
OneVsRestClassifier 은 없는놈!!!
OutputCodeClassifier 은 없는놈!!!
PassiveAggressiveClassifier [0.96666667 0.86666667 1.         0.9        0.9       ] 평균 :  0.9267
Perceptron [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ] 평균 :  0.78
QuadraticDiscriminantAnalysis [1.         0.96666667 1.         0.93333333 1.        ] 평균 :  0.98
RadiusNeighborsClassifier [0.96666667 0.9        0.96666667 0.93333333 1.        ] 평균 :  0.9533
RandomForestClassifier [0.9        0.96666667 1.         0.86666667 0.96666667] 평균 :  0.94
RidgeClassifier [0.86666667 0.8        0.93333333 0.7        0.9       ] 평균 :  0.84
RidgeClassifierCV [0.86666667 0.8        0.93333333 0.7        0.9       ] 평균 :  0.84
SGDClassifier [0.7        0.83333333 1.         0.9        0.93333333] 평균 :  0.8733
SVC [0.96666667 0.96666667 1.         0.93333333 0.96666667] 평균 :  0.9667
StackingClassifier 은 없는놈!!!
VotingClassifier 은 없는놈!!!
'''


