from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import time


#1. 데이터
x = np.array([range(10)])
x = np.transpose(x)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
                [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                 1.5, 1.4, 1.3],
                 [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
y = np.transpose(y)
x_pred = np.array([[9]])


#2. 모델링
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(3))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

start = time.time()
model.fit(x, y, epochs=1000, batch_size=1, verbose=3)  # verbose 화면에 진행과정을 보여줌 = 처리 과정에 딜레이를 줘야함
end = time.time() - start
print('걸린시간', end)
'''
verbose 
0
걸린 시간 : 9.596216440200806
1
걸린 시간 : 15.431369066238403
2
걸린 시간 : 11.429808616638184
3
걸린 시간 : 10.899047374725342
정리할 것 

verbose =1 일때 
batch가 1일 때 10일때 시간 측정하기 
'''


# #4. 평가, 예측s
# loss = model.evaluate(x,y) # 원데이터를 주었기 때문에 정답지를 제공한 것과 같음 // 응용력이 없음// 과적합 상태
#                            #평소와 다른 데이터 를 줘보자
# print('loss : ', loss)

# result = model.predict(x_pred) 
# print('[9] 의 예측값 : ', result)


# # y_predict = model.predict(x)
# # plt.scatter(x[:,0],y[:,0])
# # plt.scatter(x[:,0],y[:,1])
# # plt.scatter(x[:,0],y[:,2])

# # plt.plot(x,y_predict, color='red')
# # plt.show()
# '''
# loss :  0.0053230877965688705
# [0, 21, 201] 의 예측값 :  [[9.999012   1.5345944  0.99733484]]d
# '''
