# 실습

# 모델 : RandomforestClassifier

import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #LogisticRegression은 로지스틱회귀분석 :  분류모델 
from sklearn.tree import DecisionTreeClassifier #의사결정 나무 = 분류모델과 회귀모델 이 있다
from sklearn.ensemble import RandomForestClassifier #랜덤포레스트는 앙상블 모델이고 앙상블에는 배깅과 부스트가 있다 
import warnings
warnings.filterwarnings('ignore')
#워닝을 무시해준다 


datasets = load_iris()

x = datasets.data
y = datasets.target


from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

parameters=[
    {'n_estimators' : [100, 200]},
    {'max_depth': [6,8,10,12]},
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split': [2, 3, 5, 10]},
    {'n_jobs' : [-1,2,4]}
]
# -1 은 전부다  위의 파라미터 조정하기 여러개 섞기 # n_estimators = epochs  디폴트 알아보기 
# 시간도 체크해주기 

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, 
                random_state=66)

model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1)
import time

start_time = time.time()
model.fit(x, y)
end_time = time.time() - start_time 

print('totla time : ', end_time)
print('Best estimator : ', model.best_estimator_)
print('Best score  :', model.best_score_)