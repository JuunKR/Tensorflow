import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
#./ : 현재폴더
#../ : 상위폴더


datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                    index_col=None, header=0)

# print(datasets) #(4898, 12)

# print(datasets.info())
# print(datasets.describe())
# 다중분류
# 모델링하고
# 0.8 이상 완성!!

#1 판다스 -> 넘파이
#2 x와 y를 분리
#3 y의 라벨을 확인 np.unique(y)
#4 sklearn의 onehot??? 사용할것
#5 y의 shape 확인 (4898, ) -> (4898, 7)

# # 데이터 전처리
datasets = datasets.to_numpy()
# print(datasets)
x = datasets[:,:-1] 
# print(x.shape) #(4898, 11)
y = datasets[ : , -1:]
# print(y.shape) #(4898, 1)
# # print(np.unique(y)) #[3. 4. 5. 6. 7. 8. 9.]

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# np.set_printoptions(threshold=np.inf)

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)
# print(y)
# print(y.shape) #(4898, 7)

# print(type(x))
# print(type(y))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=9) 

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
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
AdaBoostClassifier 의 정답률 :  0.43197278911564624
1
BaggingClassifier 의 정답률 :  0.6346938775510204
2
BernoulliNB 의 정답률 :  0.4421768707482993
3
CalibratedClassifierCV 의 정답률 :  0.5265306122448979
4
CategoricalNB 의 정답률 :  0.44285714285714284
5
ClassifierChain 은 없는놈!!!
ComplementNB 의 정답률 :  0.38571428571428573
6
DecisionTreeClassifier 의 정답률 :  0.5925170068027211
7
DummyClassifier 의 정답률 :  0.4421768707482993
8
ExtraTreeClassifier 의 정답률 :  0.5918367346938775
9
ExtraTreesClassifier 의 정답률 :  0.6666666666666666
10
GaussianNB 의 정답률 :  0.45170068027210886
11
GaussianProcessClassifier 의 정답률 :  0.5265306122448979
12
GradientBoostingClassifier 의 정답률 :  0.5700680272108843
13
HistGradientBoostingClassifier 의 정답률 :  0.6408163265306123
14
KNeighborsClassifier 의 정답률 :  0.5435374149659864
15
LabelPropagation 의 정답률 :  0.5176870748299319
16
LabelSpreading 의 정답률 :  0.5142857142857142
17
LinearDiscriminantAnalysis 의 정답률 :  0.5170068027210885
18
LinearSVC 의 정답률 :  0.5319727891156463
19
LogisticRegression 의 정답률 :  0.5231292517006803
20
LogisticRegressionCV 의 정답률 :  0.5238095238095238
21
MLPClassifier 의 정답률 :  0.5163265306122449
22
MultiOutputClassifier 은 없는놈!!!
MultinomialNB 의 정답률 :  0.4421768707482993
23
NearestCentroid 의 정답률 :  0.32040816326530613
24
NuSVC 은 없는놈!!!
OneVsOneClassifier 은 없는놈!!!
OneVsRestClassifier 은 없는놈!!!
OutputCodeClassifier 은 없는놈!!!
PassiveAggressiveClassifier 의 정답률 :  0.43197278911564624
25
Perceptron 의 정답률 :  0.43605442176870746
26
QuadraticDiscriminantAnalysis 의 정답률 :  0.482312925170068
27
RadiusNeighborsClassifier 의 정답률 :  0.4435374149659864
28
RandomForestClassifier 의 정답률 :  0.6421768707482993
29
RidgeClassifier 의 정답률 :  0.5272108843537415
30
RidgeClassifierCV 의 정답률 :  0.527891156462585
31
SGDClassifier 의 정답률 :  0.3653061224489796
32
SVC 의 정답률 :  0.5340136054421769
33
StackingClassifier 은 없는놈!!!
VotingClassifier 은 없는놈!!!
'''


# # 모델구성
# # model = LinearSVC()
# #. acc_score :  0.5319727891156463
# # model = SVC()
# #. acc_score :  0.5340136054421769
# # model = KNeighborsClassifier()
# #. acc_score :  0.5435374149659864
# # model = LogisticRegression()
# #. acc_score :  0.5231292517006803
# # model = DecisionTreeClassifier()
# #. acc_score :  0.5897959183673469
# model = RandomForestClassifier()
# #. acc_score :  0.6462585034013606

# #3 훈련
# model.fit(x_train, y_train)

# #4. 평가, 예측
# y_predict = model.predict(x_test)

# results = model.score(x_test, y_test)
# print('model_score : ', results)

# acc = accuracy_score(y_test, y_predict)
# print("acc_score : ", acc)