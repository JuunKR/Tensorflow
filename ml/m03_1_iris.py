import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#@ 데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data   
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150,)
print(type(y))
print(np.unique(y)) # [0 1 2]

#. 대부분의 머신러닝은 원핫 인코딩은 통상적으로 안해도됨 

#@ 모델구성
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                train_size=0.7, shuffle=True, random_state=66)

from sklearn.svm import LinearSVC, SVC #. 레거시한 머신러닝 기법은 대부분 sklrean에 있음 support vector machine
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor           #? 리그레서 클래스파이어 차이 회귀 vs 분류
from sklearn.linear_model import LogisticRegression #. 이름에서 낚시 분류모델임
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #. 기본은 tree구조 tree가 여러개 모여 앙상블을 이룸 위보다 성능이 좋음


# model = LinearSVC() #. 모델의 정의만 해주면 됨 중요한 모델의 파라미터는 알필요가 있지만 그 외는 상관 x 디폴트도 성능이 뛰어나기 때문에
                    #. 기본적으로 머신러닝은 1차원을 받아들이기 때문에 y가 2차원이상의 데이터는 못돌림 but reshape하면됨
#. acc_score :  0.9555555555555556
# model = SVC()
#. acc_score :  0.9777777777777777
# model = KNeighborsClassifier()
#. acc_score :  0.9555555555555556
# model = LogisticRegression()
#. acc_score :  0.9777777777777777
# model = DecisionTreeClassifier()
#. acc_score :  0.9111111111111111
model = RandomForestClassifier()
#. acc_score :  0.9111111111111111

#@ 평가, 예측
results = model.score(x_test, y_test) #. evaluate = score
print('model_score : ', results)

from sklearn.metrics import r2_socre, accuracy_score  #. R2 는 회귀에 대한 스코어, acc는 분류에 대한 스코어
y_prdecit = model.predict(x_test)
acc = accuracy_score(y_test, y_prdecit)

#. acc_score :  0.9555555555555556 모델 스코어와 acc스코어가 같다. 

y_prdecit2 = model.predict(x_test[:5])
print(y_prdecit)
#. [1 1 1 0 1]

#r2 는 회귀 모델이서쓰고 acuracy는 분류에서 사용함 남자 여자 이 둘중하나 무조건 이어야함

# print('예측값 : ', y_predict)

# r2 = r2_score(y_test, y_predict) # 예측한 값과 원래값 을 비교해 오차를 확인한다.    
# print('r2 스코어 : ', r2) 


# import matplotlib.pyplot as plt

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])

# #한글로 쓰면 깨짐 과제
# plt.title('loss, val_loss')
# plt.xlabel('epochs')
# plt.ylabel('loss, val_loss')
# #범례 생성
# plt.legend('train loss', 'val_loss')       
# plt.show()