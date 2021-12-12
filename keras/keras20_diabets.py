#
# 실습 diabets
# 1. loss 와 R2로 평가
# MinMax와 Standard 결과를 명시
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# 데이터 전처리
datasets = load_diabetes()
x = datasets.data
y = datasets.target

    
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=9) 


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

# 모델
model = Sequential()
model.add(Dense(64,activation='relu', input_dim=10))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=4, verbose=1)


# 평가, 예측
loss = model.evaluate(x_test,y_test)
# print('loss : ', loss)
y_predict = model.predict(x_test)
# print('예측값 : ', y_predict)
r2 = r2_score(y_test, y_predict) 
print('r2 스코어 : ', r2) 

# MinMaxScaler()
# r2 스코어 :  0.6178604186620594

# StandardScaler()
# r2 스코어 :  0.2253180768632298