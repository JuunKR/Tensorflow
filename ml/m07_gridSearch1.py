# grid 격자 
import numpy as np
from sklearn.datasets import load_iris
import warnings 
warnings.filterwarnings(action='ignore')

# 1. data
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split

datasets = load_iris()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (150,4) (150,)
# print(y) # y = 0,1,2

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.25, shuffle=True, random_state=66)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, 
                random_state=66)

# 2. model 구성

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"]},
    # [] in 4 * kfold 5 = 20
    {"C":[1,10,100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
    # [] in 3 * kernel 1 * gamma 2 * kfold 5= 30
    {"C":[1,10,100,1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]} 
    # [] in 4 * kernel 1 * gamma 2 * kfold 5= 40
    # 20 + 30 + 40 = 90times run model
    ]

model = GridSearchCV(SVC(), parameters, cv=kfold)
# SVC() -> can replace with Tensorflow model

#GridSearchCV - 모델, 러닝레이트, 크로스발리데이션 까지 한꺼번에 랩으로 싸겠다. 그리고 이 자체를 하나의 모델로 본다 
#이 모델은 기존 모델과 우리가 넣고싶은 하이퍼 파라미터의모임을 딕셔너리 형태로 구성해주고 크로스 발리데이션 까지명시해주면 모델의 형태가 된다
#크로스 발리데이션만큼 돌아간다 cv를 5번 주면 모델이 5번 돌아가고 노드가 3개면 15번 돌아가고 러닝레이트가 3이면 45번 돌아간다 

# 3. 컴파일 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
from sklearn.metrics import accuracy_score

print('Best estimator : ', model.best_estimator_)
# Best estimator :  SVC(C=1, kernel='linear') -> best parameter
print('Best score  :', model.best_score_)
# Best score :  0.9800000000000001 -> accuracy


# train, test split
# Best estimator :  SVC(C=10, kernel='linear')
# Best score :  0.9727272727272727

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score   :', acc)
print('model score :', model.score(x_test, y_test))


