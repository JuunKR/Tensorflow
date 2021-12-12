from tensorflow.keras.models import Sequential # 모델에는 두가지가 있음 순차형/함수형 
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


# Deep 인공신경망의 layer의 깊이 / layer는 node로 이루어져 있다.
x = np.array([1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13])
y = np.array([1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13])

# train_test_split으로 만들어라!!
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, shuffle=True, random_state=66)
x_test, x_val, y_test, y_val = train_test_split(x_test,y_test,test_size=0.2, shuffle=True, random_state=66)


# #1. 데이터 전처리(특기)
# x_train = np.array([1, 2, 3, 4, 5, 6, 7]) #훈련자료
# y_train = np.array([1, 2, 3, 4, 5, 6, 7])
# x_test = np.array([8,9,10])         # test와 train 나누는 이유 과적합을 막기위해 
# y_test = np.array([8,9,10])         # 평가를 위해 만든 데이터이기 때문에 인공지능 성능에 반영이 되지 않음
# #평가자료 - 문제집을 풀어보지 않고 바로 평가를 함
# x_val = np.array([11,12,13])
# y_val = np.array([11,12,13])d

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(1)) # 예시 그림을 모델링 한 것

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_val,y_val)) # loss가 통상적으로 좋다/ 과적합에 더 잘걸림 /valloss가 더 중요  


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict([11]) 
print('11의 예측값 : ', y_predict)


# plt.scatter(x,y, c='blue', alpha=0.5)
# plt.plot(x,y_predict, color='red')
# plt.show()

