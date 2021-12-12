from tensorflow.keras.models import Sequential # 모델에는 두가지가 있음 순차형/함수형 
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


# Deep 인공신경망의 layer의 깊이 / layer는 node로 이루어져 있다.


#1. 데이터 전처리(특기)
x = np.array(range(100))   
y = np.array(range(1, 101))

s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]


from sklearn.model_selection import train_test_split
# 텐서플로 나오기 전에는 사이킥런이 최고였음 : 레거시한 머신러닝 - 하지만 오래전부터 개발해왔기 때문에 자체 부수적 기능들이 많고 유용함
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=66) # 랜덤 난수표 고정
# train_size = 몇퍼센트 데이터 사용할건지 나머지는 자연스레 test

print(x_test)
print(y_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(1)) 

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) # 훈련시키고 나온 W를 가지고 있음
print('100의 예측값 : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) # 예측한 값과 원래값 을 비교해 오차를 확인한다.
print('r2 스코어 : ', r2) 


'''
https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=handuelly&logNo=221824080339
x가 0보다 크면 기울기가 1인 직선, 0보다 작으면 함수 값이 0이 된다. 이는 0보다 작은 값들에서 뉴런이 죽을 수 있는 단점을 야기한다.d

또한 sigmoid, tanh 함수보다 학습이 빠르고, 연산 비용이 적고, 구현이 매우 간단하다는 특징이 있다.
'''