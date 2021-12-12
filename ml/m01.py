import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC #. 레거시한 머신러닝 기법은 대부분 sklrean에 있음 support vector machine
from sklearn.metrics import r2_score, accuracy_score #. R2 는 회귀에 대한 스코어, acc는 분류에 대한 스코어

#@ 데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150,)
print(y)

#. 대부분의 머신러닝은 원핫 인코딩은 통상적으로 안해도됨 

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=66) 

#@ 모델구성
model = LinearSVC() #. 모델의 정의만 해주면 됨 중요한 모델의 파라미터는 알필요가 있지만 그 외는 상관 x 디폴트도 성능이 뛰어나기 때문에
                    #. 기본적으로 머신러닝은 1차원을 받아들이기 때문에 y가 2차원이상의 데이터는 못돌림 but reshape하면됨

#@ 컴파일 훈련
model.fit(x_train, y_train)

#@ 평가, 예측
results = model.score(x_test, y_test)  #. evaluate = score
print('model_score : ', results)
#. acc 와 같음 0.9555555555555556

y_prdecit = model.predict(x_test)
acc = accuracy_score(y_test, y_prdecit)
print('acc_score : ', acc)
#. acc_score :  0.9555555555555556 모델 스코어와 acc스코어가 같다. 

y_prdecit2 = model.predict(x_test[:5])
print(y_prdecit)
#. [1 1 1 0 1]

