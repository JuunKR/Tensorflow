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

print(x_train.shape) 
print(x_test.shape)
'''
(309, 10)
(133, 10)

'''
x_train = x_train.reshape(309, 10, 1)
x_test = x_test.reshape(133, 10, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool2D, Dropout, LSTM

# 모델
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(10,1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()



# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=3) # 
model.fit(x_train,y_train, epochs=100, batch_size=4, verbose=1, callbacks=[es])


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

'''
cnn
r2 스코어 :  0.5880539950668319
'''

'''
lstm
r2 스코어 :  0.3019837951823807
'''