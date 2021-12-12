from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#@ 일반##########################################################################

# 1. 데이터 수집, 로드
data = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# # 표준화
# mms = MinMaxScaler()
# X_train = mms.fit_transform(X_train)
# X_test = mms.transform(X_test)

# # 학습
# svc = SVC(gamma="auto")
# svc.fit(X_train, y_train)

# # 예측 및 평가
# pred = svc.predict(X_test)
# print('테스트점수 :{:.2f}'.format(svc.score(X_test, y_test)))

#@ pipeline##########################################################################
# X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
# pipline = Pipeline([('scaler',MinMaxScaler()), ('svm', SVC(gamma='auto')) ])
# pipline.fit(X_train, y_train)
# print('테스트점수 :{:.2f}'.format(pipline.score(X_test, y_test)))
# print(pipline.get_params())
# print(pipline.get_params().keys())

#@ GridSearch##########################################################################
from sklearn.model_selection import GridSearchCV

# X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# # # 표준화
# mms = MinMaxScaler()
# X_train = mms.fit_transform(X_train)
# X_test = mms.transform(X_test)


# # 매개변수 세팅
# params = {
#             'C'     : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
#             'gamma' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000] 
#          }
# grid = GridSearchCV(SVC(), param_grid = params, cv = 5)
# grid.fit(X_train, y_train)
# print('최상의 교차검증 정확도 {:.2f}'.format(grid.best_score_))
# print('테스트 점수 {:.2f}'.format(grid.score(X_test, y_test)))
# print('최적의 매개변수 : {}'.format(grid.best_params_))

'''
그러나 여기에는 큰 문제점이 있습니다. 
그리드 서치는 훈련폴드와 검증폴드를 나누는데 
★스케일링은 훈련폴드에만★ 적용되어야합니다. 
그러나 그리드 서치 수행 전 이미 스케일링을 시켰기때문에 검증폴드에도 전처리과정이 적용되었습니다. 
그럼 이러한 오류를 일으키지 않으려면 어떻게 해야할까요?
'''

#@ GridSearch + pip line##########################################################################
# X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
# pipline = Pipeline([('scaler',MinMaxScaler()), ('svm', SVC()) ])
# pipline.fit(X_train, y_train)
# print('테스트점수 :{:.2f}'.format(pipline.score(X_test, y_test)))

# params = {
#             'svm__C'     : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
#             'svm__gamma' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000] 
#          }
# grid = GridSearchCV(pipline, param_grid = params, cv = 5)
# grid.fit(X_train, y_train)
# print('최상의 교차검증 정확도 {:.2f}'.format(grid.best_score_))
# print('테스트 점수 {:.2f}'.format(grid.score(X_test, y_test)))
# print('최적의 매개변수 : {}'.format(grid.best_params_))

#@ make_pipeline##########################################################################
# 표준 방법 
# pipline = Pipeline([('scaler',MinMaxScaler()), ('svm', SVC()) ])
# # 이름을 개발자가 임의로 지정이 가능하다.
# pipline_short = make_pipeline(MinMaxScaler(), SVC())

# # 파이프 라인 확인
# # 해당함수의 이름을 자동적으로 지정된다.
# for pip in pipline_short.steps:
#     print(pip)

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

# X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# pipeline = make_pipeline(StandardScaler(), PCA(n_components = 2), RobustScaler())
# pipeline.fit(X_train, y_train)

# # pca를 통해 주성분 2개를 획득
# print(pipeline.named_steps['pca'].components_)


#@ 파이프 라인(스케일링, 로지스틱 회귀) + 하이퍼 파라미터@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
from sklearn.linear_model import LogisticRegression
# pipline = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=100000))

# # step은 (이름, 객체)의 형태를 원소로 가지는 list이다.
# lr_name = pipline.steps[1][0]

# # 파라미터 대상 모델을 지정 : 키 = "알고리즘별칭__파라미터명"
# params = {
#             '{}__C'.format(lr_name): [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
#          }
# # 데이터 가공, 전처리(훈련/테스트 데이터 분류)
# X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

# grid = GridSearchCV(pipline, param_grid = params, cv = 5)
# grid.fit(X_train, y_train)
# print('최상의 모델 {}'.format(grid.best_estimator_))
# print('로지스틱 회귀 단계 {}'.format(grid.best_estimator_.named_steps[lr_name]))
# print('로지스틱 회귀 계수 {}'.format(grid.best_estimator_.named_steps[lr_name].coef_))
# print('최상의 교차검증 정확도 {:.2f}'.format(grid.best_score_))
# print('테스트 점수 {:.2f}'.format(grid.score(X_test, y_test)))
# print('최적의 매개변수 : {}'.format(grid.best_params_))


#@ 전처리기 2개에 모델 1개를 가지는 파이프라인@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge, Lasso
data2 = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data2.data, data2.target)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.preprocessing import PolynomialFeatures
pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())

# 파라미터 -> alpha가 커질수록 계수가 작아지면서 복잡도가 감소 -> 단순화
# 다른 작업에 대한 파라미터이지만 한꺼번에 수행이 가능하다 점이 바로 파이프라인의 장점
param = {
    # 전처리기에 대한 파라미터
    'polynomialfeatures__degree': [1,2,3],
    # 알고리즘에 파라미터
    'ridge__alpha' : [0.001, 0.01, 0.1, 1, 10, 100]
}
#그리드 서치
# iid : 각 테스트 세트의 샘플 수, 가중치가 적용된 폴드(cv에서 세트 규정)에 평균점수 반환
grid = GridSearchCV(pipe, param_grid = param, cv = 5, n_jobs = -1, iid = True)

#훈련
grid.fit(X_train, y_train)

#@ 출처 https://hhhh88.tistory.com/6