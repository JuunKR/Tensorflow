from tensorflow.keras.models import Sequential # 모델에는 두가지가 있음 순차형/함수형 
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


# Deep 인공신경망의 layer의 깊이 / layer는 node로 이루어져 있다.


#1. 데이터 전처리(특기)
x_train = np.array([1, 2, 3, 4, 5, 6, 7])
y_train = np.array([1, 2, 3, 4, 5, 6, 7])
x_test = np.array([8,9,10])         # test와 train 나누는 이유 과적합을 막기위해 
y_test = np.array([8,9,10])         # 평가를 위해 만든 데이터이기 때문에 인공지능 성능에 반영이 되지 않음


#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(1)) # 예시 그림을 모델링 한 것

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict([11]) 
print('11의 예측값 : ', y_predict)


# plt.scatter(x,y, c='blue', alpha=0.5)
# plt.plot(x,y_predict, color='red')
# plt.show()d

