from tensorflow.keras.models import Sequential # 모델에는 두가지가 있음 순차형/함수형 
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


# Deep 인공신경망의 layer의 깊이 / layer는 node로 이루어져 있다.


#1. 데이터 전처리(특기)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12])


x_pred = [11]


#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(1)) # 예시 그림을 모델링 한 것

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)

# result = model.predict(x_pred) 
# print('10의 예측값 : ', result)

y_predict = model.predict(x)

plt.scatter(x,y, c='blue', alpha=0.5)
plt.plot(x,y_predict, color='red')
plt.show()

'''
loss :  3.7167935371398926
10의 예측값 :  [[10.262048]]

'''

