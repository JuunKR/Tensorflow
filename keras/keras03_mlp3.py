from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np



#1. 데이터
x = np.array([range(10), range(21, 31),range(201,211)])
x = np.transpose(x)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
                [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                 1.5, 1.4, 1.3],
                 [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
y = np.transpose(y)
print(y)

x_pred = np.array([[0, 21, 201]])


#2. 모델링
model = Sequential()
model.add(Dense(4, input_dim=3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)


#4. 평가, 예측s
loss = model.evaluate(x,y)
print('loss : ', loss)

# result = model.predict(x_pred) 
# print('[0, 21, 201] 의 예측값 : ', result)



y_predict = model.predict(x)
plt.scatter(x,y)


plt.plot(x,y_predict, color='red')
plt.show()

'''
loss :  6.66992207243311e-07
[10, 1.3, 1] 의 예측값 :  [[20.000637]]

loss :  2.852923444152111e-06
[10, 1.3, 1] 의 예측값 :  [[19.997194]]d
'''
