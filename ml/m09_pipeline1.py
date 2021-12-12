import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150,)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=66) 

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.pipeline import make_pipeline, Pipeline


# # 모델구성
from sklearn.svm import LinearSVC, SVC #. 레거시한 머신러닝 기법은 대부분 sklrean에 있음 support vector machine
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor           #? 리그레서 클래스파이어 차이 회귀 vs 분류
from sklearn.linear_model import LogisticRegression #. 이름에서 낚시 분류모델임
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #. 기본은 tree구조 tree가 여러개 모여 앙상블을 이룸 위보다 성능이 좋음

model = make_pipeline(MinMaxScaler(), SVC())
#@ 사이킷런의 Pipeline 클래스는 연속된 변환을 순차적으로 처리할 수 있는 기능을 제공하는 유용한 래퍼(Wrapper) 도구입니다.
#@ to assemble several steps that can be cross-validated together while setting different parameters
#@ 데이터변환(전처리)와 모델을 연결하여 코드를 줄이고 재사용성을 높이기위함

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
# model = RandomForestClassifier()
#. acc_score :  0.9111111111111111




# model = Sequential()
# model.add(Dense(128,activation='relu', input_shape=(4,)))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(3, activation='softmax')) # 다중 분류의 라벨의 수가 3개 

# 컴파일 훈련
model.fit(x_train, y_train)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 2진분류를 위한
 
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1) # patience는 epoches 기준으로 설정, mode #loss 에서 val_los로 바꿈 // 너무 빨리 끝나면 patience를 조절하자
# hist = model.fit(x_train,y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])

#평가, 예측
results = model.score(x_test, y_test)  #. evaluate = score
print('model_score : ', results)
#. acc 와 같음 0.9555555555555556

from sklearn.metrics import r2_score, accuracy_score #. R2 는 회귀에 대한 스코어, acc는 분류에 대한 스코어
y_prdecit = model.predict(x_test)
acc = accuracy_score(y_test, y_prdecit)
print('acc_score : ', acc)
#. acc_score :  0.9555555555555556 모델 스코어와 acc스코어가 같다. 

y_prdecit2 = model.predict(x_test[:5])
print(y_prdecit)
#. [1 1 1 0 1]

