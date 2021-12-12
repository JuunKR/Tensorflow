'''
parameters = [
    {"n_esrimators": [100, 200]},
    {"max_depth": [6, 8, 10, 12]},
    {"min_sample_leaf": [3, 5, 7, 10]},
    {"min_sample_split": [2, 3, 5, 10]},
    {"n_jobs": [-1, 2, 4]},
]
'''

import numpy as np
import warnings 
from sklearn.datasets import load_breast_cancer
warnings.filterwarnings(action='ignore')

# 1. data
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target


n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, 
                random_state=66)

# 2. model 구성

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

parameters = [{
    "n_estimators": [100, 200], # =. epochs, default = 100
    "max_depth": [6, 8, 10, 12],
    "min_samples_leaf": [3, 5, 7, 10],
    "min_samples_split": [2, 3, 5, 10],
    "n_jobs": [-1] # =. qauntity of cpu; -1 = all
}]

model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1)
# Fitting 5 folds for each of 128 candidates, totalling 640 fits

# 3. 컴파일 훈련
import time
st = time.time()
model.fit(x, y)
et = time.time() - st

# 4. 평가 예측

print('totla time : ', et)
print('Best estimator : ', model.best_estimator_)
print('Best score  :', model.best_score_)

# totla time :  69.42868947982788
# Best estimator :  
# RandomForestClassifier(max_depth=8, 
#                        min_samples_leaf=3, 
#                        min_samples_split=5,
#                        n_jobs=-1)
# Best score  : 0.9666356155876417