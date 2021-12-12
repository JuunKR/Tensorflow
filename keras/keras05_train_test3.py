from tensorflow.keras.models import Sequential # 모델에는 두가지가 있음 순차형/함수형 
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


# Deep 인공신경망의 layer의 깊이 / layer는 node로 이루어져 있다.d


#1. 데이터 전처리(특기)
x = np.array(range(100))   
y = np.array(range(1, 101))

s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=66)

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(1)) 

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=5000, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

result = model.predict([11]) 
print('10의 예측값 : ', result)



