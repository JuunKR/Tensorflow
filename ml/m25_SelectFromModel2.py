# 실습
# 1.상단 모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성
#   최적의 R2 피처임폰턴스 구할것 

# 2. 위 스레드값으로 SelectFromModel 돌려서 최적의 피쳐 개수 구할 것

# 3. 위 피쳐 갯수로 피쳐 갯수를 조정한뒤 그걸로 다시 랜덤 서치 그리드서치 하기
#    최적의 r2구할것

# 4. 1번값과 3번 값 비교

from sklearn.datasets import load_diabetes
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
import numpy as np


x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=66
)

#2. 모델
parameters =  [{"n_estimators":[1000,4000,8000], "learning_rate": [0.1, 0.3, 0.001, 0.01], "max_depth": [4,5,6]},
    {"n_estimators":[99, 100, 110], "learning_rate": [0.1, 0.001, 0.01], "max_depth": [4,5,6], "colsample_bytree":[0.6,0.9,1]},
     {"n_estimators":[90,100], "learning_rate": [0.1, 0.001, 0.5], "max_depth": [4,5,6], "colsample_bytree":[0.6,0.9,1], "colsample_bylevel":[0.6,0.7,0.9]},]
    

n_jobs = -1


# model = GridSearchCV(XGBRegressor(n_jobs = 8), parameters, verbose=1)
model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
             colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=4,
             min_child_weight=1,  monotone_constraints='()',
             n_estimators=90, n_jobs=8, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None,)


#3. 훈련
model.fit(x_train,y_train)

#4, 평가, 예측
# print("최적의 매개변수 : ", model.best_estimator_) 
# print('best_score_ : ', model.best_score_) 

# exit()
'''
최적의 매개변수 :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
             colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=4,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=90, n_jobs=8, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)

best_score_ :  0.4517813235384625
'''

score = model.score(x_test,y_test)
print('model.score : ', score) 

thresholds = np.sort(model.feature_importances_)
print(thresholds)

