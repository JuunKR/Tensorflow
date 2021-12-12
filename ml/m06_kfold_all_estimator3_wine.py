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
from sklearn.model_selection import KFold, cross_val_score
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
AdaBoostClassifier [0.41428571 0.45       0.42244898 0.36261491 0.43615935] 평균 :  0.4171
BaggingClassifier [0.67653061 0.63571429 0.67346939 0.659857   0.64964249] 평균 :  0.659
BernoulliNB [0.45816327 0.43367347 0.44285714 0.46271706 0.44637385] 평균 :  0.4488
CalibratedClassifierCV [0.51326531 0.47142857 0.49693878 0.55158325 0.48416752] 평균 :  0.5035
CategoricalNB [       nan        nan 0.50306122 0.51072523        nan] 평균 :  nan
ClassifierChain 은 없는놈!!!
ComplementNB [0.38163265 0.37653061 0.36632653 0.34320735 0.36159346] 평균 :  0.3659
DecisionTreeClassifier [0.6377551  0.59693878 0.59387755 0.58529111 0.60776302] 평균 :  0.6043
DummyClassifier [0.45816327 0.43367347 0.44285714 0.46271706 0.44637385] 평균 :  0.4488
ExtraTreeClassifier [0.63061224 0.6        0.58163265 0.61082737 0.58835546] 평균 :  0.6023
ExtraTreesClassifier [0.72346939 0.66530612 0.69387755 0.71501532 0.68947906] 평균 :  0.6974
GaussianNB [0.46530612 0.44591837 0.45510204 0.41266599 0.46373851] 평균 :  0.4485
GaussianProcessClassifier [0.59693878 0.57244898 0.58163265 0.57405516 0.57099081] 평균 :  0.5792
GradientBoostingClassifier [0.61326531 0.5755102  0.59489796 0.61287028 0.59039837] 평균 :  0.5974
HistGradientBoostingClassifier [0.69183673 0.6622449  0.67653061 0.67109295 0.6639428 ] 평균 :  0.6731
KNeighborsClassifier [0.48979592 0.48469388 0.4755102  0.46373851 0.45863126] 평균 :  0.4745
LabelPropagation [0.59387755 0.57244898 0.57040816 0.5628192  0.56588355] 평균 :  0.5731
LabelSpreading [0.59387755 0.57244898 0.57040816 0.56384065 0.56588355] 평균 :  0.5733
LinearDiscriminantAnalysis [0.5255102  0.51326531 0.5244898  0.56384065 0.52706844] 평균 :  0.5308
LinearSVC [0.45918367 0.41632653 0.41530612 0.19101124 0.11848825] 평균 :  0.3201
LogisticRegression [0.47142857 0.45204082 0.44897959 0.48723187 0.46373851] 평균 :  0.4647
LogisticRegressionCV [0.49693878 0.49897959 0.49387755 0.53728294 0.49846782] 평균 :  0.5051
MLPClassifier [0.51734694 0.49285714 0.52857143 0.52706844 0.48621042] 평균 :  0.5104
MultiOutputClassifier 은 없는놈!!!
MultinomialNB [0.41326531 0.39693878 0.3877551  0.38304392 0.40653728] 평균 :  0.3975
NearestCentroid [0.12959184 0.10204082 0.10102041 0.11235955 0.09090909] 평균 :  0.1072
NuSVC [nan nan nan nan nan] 평균 :  nan
OneVsOneClassifier 은 없는놈!!!
OneVsRestClassifier 은 없는놈!!!
OutputCodeClassifier 은 없는놈!!!
PassiveAggressiveClassifier [0.29591837 0.28469388 0.38571429 0.46271706 0.4494382 ] 평균 :  0.3757
Perceptron [0.45816327 0.43367347 0.32244898 0.33094995 0.09499489] 평균 :  0.328
QuadraticDiscriminantAnalysis [0.48367347 0.45102041 0.50306122 0.46782431 0.48008172] 평균 :  0.4771
RadiusNeighborsClassifier [nan nan nan nan nan] 평균 :  nan
RandomForestClassifier [0.70816327 0.67755102 0.68673469 0.70480082 0.68028601] 평균 :  0.6915
RidgeClassifier [0.53163265 0.5122449  0.52142857 0.54954035 0.51276813] 평균 :  0.5255
RidgeClassifierCV [0.53163265 0.5122449  0.52142857 0.54954035 0.51276813] 평균 :  0.5255
SGDClassifier [0.45816327 0.22755102 0.48061224 0.48927477 0.41266599] 평균 :  0.4137
SVC [0.4622449  0.4377551  0.44693878 0.46373851 0.4473953 ] 평균 :  0.4516
StackingClassifier 은 없는놈!!!
VotingClassifier 은 없는놈!!!
'''