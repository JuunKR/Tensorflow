import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN , LSTM ,GRU
from datetime import datetime
import tensorflow as tf
'''
tensorboard --logdir=./logs/fit/ 
'''


#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])


y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = np.array([50,60,70])
x_predict = x_predict.reshape(1,3,1)
# x_predict = x_predict.reshape(x_predict[0],x_predict[1],1)
# 행열 출력해볼 필요x 하지만 위에꺼는 3, 이기때문에 인덱스x

print(x.shape, y.shape) # (13, 3) (13,)


x = x.reshape(13,3,1) #@ 피쳐 몇 개씩 자르는지 = 1 
# #@ (batch_size, timesteps, feature) 위에꺼 모양나타내느거임 4 = batch_size 3 =timesteps


#2. 모델 
model = Sequential()
# model.add(SimpleRNN(units = 10, activation='relu', input_shape=(3,1))) 아래랑 같은  표현  timesteps, feature
model.add(GRU(units = 16, activation='relu', input_shape=(3,1))) 
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))
model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 10)                480
_________________________________________________________________
dense (Dense)                (None, 16)                176
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 136
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 18
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 3
=================================================================
Total params: 813
Trainable params: 813
Non-trainable params: 0
_________________________________________________________________
''' 

# Define the Keras TensorBoard callback.
logdir="logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=500, batch_size=1, callbacks=[tensorboard_callback])
'''
Total params = recurrent_weights + input_weights + biases
(num_units*num_units)+(num_features*num_units) + (1*num_units)
(num_features + num_units)* num_units + num_units

(input + bias) *  output + * output *output
=3 * ( input + bias + output) * output
'''



#4. 평가, 예측
results = model.predict(x_predict)
print(results) 

# [[8.628703]]
# [[8.019494]]