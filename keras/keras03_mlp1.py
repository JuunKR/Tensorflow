from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic



#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
                [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                 1.5, 1.4, 1.3]])

'''
ic| x.shape: (2, 10)
ic| y.shape: (10,)
ic| x: array([[ 1. ,  2. ,  3. ,  4. ,  5. ,  6. ,  7. ,  8. ,  9. , 10. ], 
              [ 1. ,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.5,  1.4,  1.3]])
ic| y: array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
'''
'''
ic| x.shape: (10, 2)
ic| y.shape: (10,)
ic| x: array([[ 1. ,  1. ],
              [ 2. ,  1.1],
              [ 3. ,  1.2],
              [ 4. ,  1.3],
              [ 5. ,  1.4],
              [ 6. ,  1.5],
              [ 7. ,  1.6],
              [ 8. ,  1.5],
              [ 9. ,  1.4],
              [10. ,  1.3]])
ic| y: array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
'''


# print(x.shape)
# x = np.transpose(x) # 행의 개수가 y과 동일해야하기 때문에
# # print(x)
# # print(x.shape)
# print(x)

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y1 = np.transpose(y) 

ic(x)
ic(y1)
ic(x.shape)
ic(y1.shape)




# # print(y.shape) # (10,) != (10,1)

# x_pred = np.array([[10, 1.3]])
# # print(x_pred.shape) # (1, 2)

# #2. 모델링
# model = Sequential()
# model.add(Dense(4, input_dim=2))
# model.add(Dense(1))

# #3. 컴파일 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=1, batch_size=1)


# #4. 평가, 예측
# loss = model.evaluate(x,y)
# # print('loss : ', loss)

# # result = model.predict(x_pred) 
# # print('[10, 1.3] 의 예측값 : ', result)





# print('가즈아ㅏㅏㅏㅏ')
# y_predict = model.predict(x)
# # [[-7.570955]]
# print(y_predict)


# plt.scatter(x[:,0],y)
# plt.scatter(x[:,1],y)
# plt.plot(x,y_predict, color='red')
# plt.show()


# '''

# loss :  9.094947153254554e-14
# [10, 1.3]의 예측값 :  [[20.000002]]

# '''

