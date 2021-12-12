from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np



#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
                [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                 1.5, 1.4, 1.3],
                 [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])

# print(x.shape) #(3,10)
x = np.transpose(x) # 행의 개수가 y과 동일해야하기 때문에
print(x)
print(x.shape) #(10,3)

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# print(y.shape) # (10,) != (10,1)

x_pred = np.array([[10, 1.3, 1]])
# print(x_pred.shape) # (1, 3)

#2. 모델링
model = Sequential()
model.add(Dense(4, input_dim=3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)


print(x[:])

y_predict = model.predict(x)

plt.scatter(x[:,0],y)
plt.scatter(x[:,1],y)
plt.scatter(x[:,2],y)
plt.plot(x,y_predict, color='red')
plt.show()

# result = model.predict(x_pred) 
# print('[10, 1.3, 1] 의 예측값 : ', result)


'''
loss :  6.66992207243311e-07
[10, 1.3, 1] 의 예측값 :  [[20.000637]]

loss :  2.852923444152111e-06
[10, 1.3, 1] 의 예측값 :  [[19.997194]]d
'''
