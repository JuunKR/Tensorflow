import numpy as np
from numpy import array
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN , LSTM , Input
from datetime import datetime
import tensorflow as tf

#1 데이터
x1 = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])

x2 = array([[10,20,30], [20,30,40], [30,40,50], [40,50,60],
              [50,60,7], [60,70,80], [70,80,90], [80,90,100],
              [90,100,110],[100,110,120],
              [2,3,4],[3,4,5],[4,5,6]])


y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = array([55,65,75])
x2_predict = array([65,75,85])

x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1)

x1 = x1.reshape(13,3,1) 
x2 = x2.reshape(13,3,1) 

# 모델 1
input1 = Input(shape=(3,1))
xx = SimpleRNN(units = 16, activation='relu')(input1)
xx = Dense(16)(xx)
xx = Dense(8)(xx)
xx = Dense(4)(xx)
xx = Dense(2)(xx)
output1 = Dense(1)(xx)


# 모델 2
input2 = Input(shape=(3,1))
xx = SimpleRNN(units = 16, activation='relu')(input1)
xx = Dense(16)(xx)
xx = Dense(8)(xx)
xx = Dense(4)(xx)
xx = Dense(2)(xx)
output2 = Dense(1)(xx)

#합치기

from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(10,activation='relu')(merge1)
merge3 = Dense(5,activation='relu')(merge2)
last_output = Dense(1)(merge3)
model = Model(inputs=[input1, input2], outputs=last_output)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1,x2],y, epochs=500, batch_size=1)


#4. 평가, 예측
results = model.predict([x1_predict, x2_predict])
print(results) 


#[[85.90098]]
