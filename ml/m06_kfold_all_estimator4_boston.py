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
import numpy as np

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
ARDRegression [0.80125693 0.76317071 0.56809285 0.6400258  0.71991866] 평균 :  0.6985
AdaBoostRegressor [0.91235479 0.79184941 0.79455134 0.82415013 0.87307048] 평균 :  0.8392
BaggingRegressor [0.89491013 0.8027143  0.79960002 0.87867376 0.86948187] 평균 :  0.8491
BayesianRidge [0.79379186 0.81123808 0.57943979 0.62721388 0.70719051] 평균 :  0.7038
CCA [0.79134772 0.73828469 0.39419624 0.5795108  0.73224276] 평균 :  0.6471
DecisionTreeRegressor [0.70819872 0.64940434 0.78295821 0.71116373 0.82310288] 평균 :  0.735
DummyRegressor [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911] 평균 :  -0.0135
ElasticNet [0.73383355 0.76745241 0.59979782 0.60616114 0.64658354] 평균 :  0.6708
ElasticNetCV [0.71677604 0.75276545 0.59116613 0.59289916 0.62888608] 평균 :  0.6565
ExtraTreeRegressor [0.85751384 0.71240629 0.70258444 0.68744608 0.8643778 ] 평균 :  0.7649
ExtraTreesRegressor [0.93860713 0.85111742 0.79415626 0.87565434 0.92505382] 평균 :  0.8769
GammaRegressor [-0.00058757 -0.03146716 -0.00463664 -0.02807276 -0.00298635] 평균 :  -0.0136
GaussianProcessRegressor [-6.07310526 -5.51957093 -6.33482574 -6.36383476 -5.35160828] 평균 :  -5.9286
GradientBoostingRegressor [0.94586911 0.83126785 0.82692461 0.88423428 0.93149959] 평균 :  0.884
HistGradientBoostingRegressor [0.93235978 0.82415907 0.78740524 0.88879806 0.85766226] 평균 :  0.8581
HuberRegressor [0.71663623 0.66405398 0.52849847 0.36282932 0.64608284] 평균 :  0.5836
IsotonicRegression [nan nan nan nan nan] 평균 :  nan
KNeighborsRegressor [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856] 평균 :  0.5286
KernelRidge [0.83333255 0.76712443 0.5304997  0.5836223  0.71226555] 평균 :  0.6854
Lars [0.77467361 0.79839316 0.5903683  0.64083802 0.68439384] 평균 :  0.6977
LarsCV [0.80141197 0.77573678 0.57807429 0.60068407 0.70833854] 평균 :  0.6928
Lasso [0.7240751  0.76027388 0.60141929 0.60458689 0.63793473] 평균 :  0.6657
LassoCV [0.71314939 0.79141061 0.60734295 0.61617714 0.66137127] 평균 :  0.6779
LassoLars [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911] 평균 :  -0.0135
LassoLarsCV [0.80301044 0.77573678 0.57807429 0.60068407 0.72486787] 평균 :  0.6965
LassoLarsIC [0.81314239 0.79765276 0.59012698 0.63974189 0.72415009] 평균 :  0.713
LinearRegression [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 평균 :  0.7128
LinearSVR [0.57764764 0.56994416 0.53737045 0.55896384 0.40163608] 평균 :  0.5291
MLPRegressor [0.41892079 0.63656666 0.55864927 0.42409773 0.4489319 ] 평균 :  0.4974
MultiOutputRegressor 은 없는놈!!!
MultiTaskElasticNet [nan nan nan nan nan] 평균 :  nan
MultiTaskElasticNetCV [nan nan nan nan nan] 평균 :  nan
MultiTaskLasso [nan nan nan nan nan] 평균 :  nan
MultiTaskLassoCV [nan nan nan nan nan] 평균 :  nan
NuSVR [0.2594254  0.33427351 0.263857   0.11914968 0.170599  ] 평균 :  0.2295
OrthogonalMatchingPursuit [0.58276176 0.565867   0.48689774 0.51545117 0.52049576] 평균 :  0.5343
OrthogonalMatchingPursuitCV [0.75264599 0.75091171 0.52333619 0.59442374 0.66783377] 평균 :  0.6578
PLSCanonical [-2.23170797 -2.33245351 -2.89155602 -2.14746527 -1.44488868] 평균 :  -2.2096
PLSRegression [0.80273131 0.76619347 0.52249555 0.59721829 0.73503313] 평균 :  0.6847
PassiveAggressiveRegressor [ 0.2615862  -0.49091784 -0.15453605  0.08267163 -1.24468718] 평균 :  -0.3092
PoissonRegressor [0.85659559 0.8189993  0.66805858 0.67998756 0.75423831] 평균 :  0.7556
RANSACRegressor [-0.06087574  0.74683967  0.56390725 -0.07526525 -0.68442309] 평균 :  0.098
RadiusNeighborsRegressor [nan nan nan nan nan] 평균 :  nan
RandomForestRegressor [0.92401583 0.85779991 0.82116713 0.88498101 0.90359643] 평균 :  0.8783
RegressorChain 은 없는놈!!!
Ridge [0.80984876 0.80618063 0.58111378 0.63459427 0.72264776] 평균 :  0.7109
RidgeCV [0.81125292 0.80010536 0.58888305 0.64008984 0.72362911] 평균 :  0.7128
SGDRegressor [-9.09142639e+25 -4.14227171e+26 -5.17834052e+25 -9.88744603e+26
 -1.49984481e+25] 평균 :  -3.1213357828151356e+26
SVR [0.23475113 0.31583258 0.24121157 0.04946335 0.14020554] 평균 :  0.1963
StackingRegressor 은 없는놈!!!
TheilSenRegressor [0.79046684 0.72425135 0.58902364 0.5600968  0.7234936 ] 평균 :  0.6775
TransformedTargetRegressor [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 평균 :  0.7128
TweedieRegressor [0.74518926 0.75477245 0.56401081 0.57693341 0.6293039 ] 평균 :  0.654
VotingRegressor 은 없는놈!!!
'''