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

allAlgorithms = all_estimators(type_filter='regressor')
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
모델의 개수 :  54
ARDRegression [0.49874835 0.48765748 0.56284846 0.37728801 0.53474369] 평균 :  0.4923
AdaBoostRegressor [0.3660001  0.44929932 0.51893937 0.41915814 0.44040354] 평균 :  0.4388
BaggingRegressor [0.37703989 0.41554864 0.38785144 0.37852045 0.38107833] 평균 :  0.388
BayesianRidge [0.50082189 0.48431051 0.55459312 0.37600508 0.5307344 ] 평균 :  0.4893
CCA [0.48696409 0.42605855 0.55244322 0.21708682 0.50764701] 평균 :  0.438
DecisionTreeRegressor [-0.23152399 -0.10941878 -0.18019563 -0.09844515  0.03037616] 평균 :  -0.1178
DummyRegressor [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
 -9.58335111e-03] 평균 :  -0.0033
ElasticNet [ 0.00810127  0.00637294  0.00924848  0.0040621  -0.00081988] 평균 :  0.0054
ElasticNetCV [0.43071558 0.461506   0.49133954 0.35674829 0.4567084 ] 평균 :  0.4394
ExtraTreeRegressor [-0.06464772  0.16348617 -0.12170887 -0.28562721 -0.15364824] 평균 :  -0.0924
ExtraTreesRegressor [0.39636256 0.47835888 0.53858681 0.41578781 0.46400224] 평균 :  0.4586
GammaRegressor [ 0.00523561  0.00367973  0.0060814   0.00174734 -0.00306898] 평균 :  0.0027
GaussianProcessRegressor [ -5.63608222 -15.27407233  -9.94979908 -12.46885984 -12.04801195] 평균 :  -11.0754
GradientBoostingRegressor [0.39218778 0.48201209 0.48615834 0.39632953 0.44358615] 평균 :  0.4401
HistGradientBoostingRegressor [0.28899498 0.43812684 0.51713242 0.37267554 0.35643755] 평균 :  0.3947
HuberRegressor [0.50334736 0.47508232 0.54634137 0.36878539 0.51729827] 평균 :  0.4822
IsotonicRegression [nan nan nan nan nan] 평균 :  nan
KNeighborsRegressor [0.39683913 0.32569788 0.43311217 0.32635899 0.35466969] 평균 :  0.3673
KernelRidge [-3.38476443 -3.49366182 -4.0996205  -3.39039111 -3.60041537] 평균 :  -3.5938
Lars [ 0.49198665 -0.66475442 -1.04410299 -0.04236657  0.51190679] 평균 :  -0.1495
LarsCV [0.4931481  0.48774421 0.55427158 0.38001456 0.52413596] 평균 :  0.4879
Lasso [0.34315574 0.35348212 0.38594431 0.31614536 0.3604865 ] 평균 :  0.3518
LassoCV [0.49799859 0.48389346 0.55926851 0.37740074 0.51636393] 평균 :  0.487
LassoLars [0.36543887 0.37812653 0.40638095 0.33639271 0.38444891] 평균 :  0.3742
LassoLarsCV [0.49719648 0.48426377 0.55975856 0.37984022 0.51190679] 평균 :  0.4866
LassoLarsIC [0.49940515 0.49108789 0.56130589 0.37942384 0.5247894 ] 평균 :  0.4912
LinearRegression [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 평균 :  0.4876
LinearSVR [-0.33470258 -0.31629592 -0.4279369  -0.30195029 -0.47345315] 평균 :  -0.3709
MLPRegressor [-2.88008394 -3.06903941 -3.43867003 -2.93859054 -3.16963779] 평균 :  -3.0992
MultiOutputRegressor 은 없는놈!!!
MultiTaskElasticNet [nan nan nan nan nan] 평균 :  nan
MultiTaskElasticNetCV [nan nan nan nan nan] 평균 :  nan
MultiTaskLasso [nan nan nan nan nan] 평균 :  nan
MultiTaskLassoCV [nan nan nan nan nan] 평균 :  nan
NuSVR [0.14471275 0.17351835 0.18539957 0.13894135 0.1663745 ] 평균 :  0.1618
OrthogonalMatchingPursuit [0.32934491 0.285747   0.38943221 0.19671679 0.35916077] 평균 :  0.3121
OrthogonalMatchingPursuitCV [0.47845357 0.48661326 0.55695148 0.37039612 0.53615516] 평균 :  0.4857
PLSCanonical [-0.97507923 -1.68534502 -0.8821301  -1.33987816 -1.16041996] 평균 :  -1.2086
PLSRegression [0.47661395 0.4762657  0.5388494  0.38191443 0.54717873] 평균 :  0.4842
PassiveAggressiveRegressor [0.4551155  0.48548991 0.48908643 0.35253494 0.49102906] 평균 :  0.4547
PoissonRegressor [0.32061441 0.35803358 0.3666005  0.28203414 0.34340626] 평균 :  0.3341
RANSACRegressor [ 0.13375783  0.1214814   0.35627656 -1.32366386 -0.01538842] 평균 :  -0.1455
RadiusNeighborsRegressor [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
 -9.58335111e-03] 평균 :  -0.0033
RandomForestRegressor [0.3746037  0.48366423 0.4624227  0.39251353 0.41040501] 평균 :  0.4247
RegressorChain 은 없는놈!!!
Ridge [0.40936669 0.44788406 0.47057299 0.34467674 0.43339091] 평균 :  0.4212
RidgeCV [0.49525464 0.48761091 0.55171354 0.3801769  0.52749194] 평균 :  0.4884
SGDRegressor [0.39341979 0.44184075 0.46474913 0.32961578 0.41505875] 평균 :  0.4089
SVR [0.14331635 0.18438697 0.17864042 0.1424597  0.1468719 ] 평균 :  0.1591
StackingRegressor 은 없는놈!!!
TheilSenRegressor [0.50078751 0.44331883 0.55194074 0.33290922 0.52521794] 평균 :  0.4708
TransformedTargetRegressor [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 평균 :  0.4876
TweedieRegressor [ 0.00585525  0.00425899  0.00702558  0.00183408 -0.00315042] 평균 :  0.0032
VotingRegressor 은 없는놈!!!
'''