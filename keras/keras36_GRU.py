import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN , LSTM, GRU
from datetime import datetime
import tensorflow as tf
'''
tensorboard --logdir=./logs/fit/ 
'''


#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape) # (4, 3) (4,)

x = x.reshape(4,3,1) #@ 피쳐 몇 개씩 자르는지 = 1 
#@ (batch_size, timesteps, feature) 위에꺼 모양나타내느거임 4 = batch_size 3 =timesteps


#2. 모델 
model = Sequential()
# model.add(SimpleRNN(units = 10, activation='relu', input_shape=(3,1))) 아래랑 같은  표현  timesteps, feature
# model.add(LSTM(units = 10, activation='relu', input_length=3, input_dim=1)) 
model.add(GRU(units = 10, activation='relu', input_length=3, input_dim=1)) 
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()
'''
조경현 교수 lstm 만든사람 제자임
lstm은속도가 너무 느림
망각게이트를뺐다
두개 이름바꿈
Layer (type)                 Output Shape              Param #
=================================================================
gru (GRU)                    (None, 10)                390
_________________________________________________________________
dense (Dense)                (None, 16)                176
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 136
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 18
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 3
=================================================================
Total params: 723
Trainable params: 723
Non-trainable params: 0

(input + bias) *  output + * output *output
= 3 * ( input + bias + output + resetgate) * output = 360 :: 390

lstm보다는 조금 빠르고 성능도 거의 비슷 // 컴퓨터 성능이 좋으면 그냥 lstm
''' 

# Define the Keras TensorBoard callback.
logdir="logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x,y, epochs=500, batch_size=1, callbacks=[tensorboard_callback])
'''
Total params = recurrent_weights + input_weights + biases
(num_units*num_units)+(num_features*num_units) + (1*num_units)
(num_features + num_units)* num_units + num_units

(input + bias) *  output + * output *output
=( input + bias + output) * output
'''



#4. 평가, 예측
x_input = np.array([[5], [6], [7]]).reshape(1,3,1)
results = model.predict(x_input)
print(results) 

# [[8.628703]]
# [[8.019494]]