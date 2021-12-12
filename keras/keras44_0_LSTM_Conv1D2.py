import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN , LSTM, Conv1D, Flatten
from datetime import datetime
import tensorflow as tf

x_data = np.array(range(1, 101)) # 연속된 열개의 데이터
x_predict = np.array(range(96, 106))
'''
predict의 예상 결과값
96, 97, 98, 99, 100  /  101
...
101, 102, 103, 104, 105  /  106
'''

size = 6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)
    # 함수를 통과해서 나온 값


dataset = split_x(x_data, size)
print(dataset)

x = dataset[:, :5]
y = dataset[:, 5]

# print("x : \n", x )
# print("y : ", y )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=9) 


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 


print(x_train.shape) # 67, 5
print(x_test.shape) # 29,5
print(y_train.shape) # 67,
print(y_test.shape) # 29,

x_train = x_train.reshape(67,5,1)
x_test = x_test.reshape(29,5,1)

print(x_train.shape) # 67, 5
print(x_test.shape) # 29,5

#2. 모델 
model = Sequential()


# model.add(LSTM(units = 128, activation='relu', input_shape=(5,1))) 
model.add(Conv1D(64, 2, input_shape=(5,1)))
model.add(LSTM(64))
#@ model.add(LSTM(64, return_sequenct=True))
#@ model.add(Conv1D(64, 2))

# model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()

# Define the Keras TensorBoard callback.
logdir="logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',  )


from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=3)

import time
start_time = time.time()
hist = model.fit(x_train,y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es, tensorboard_callback])
end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)


print('걸린시간 : ', end_time)
print('loss값 : ', loss)
print('rmse값 = ', rmse)
print('r2 스코어 : ', r2)  

'''
걸린시간 :  23.042980670928955
loss값 :  2.9767794609069824
rmse =  1.7253345652831666
r2 스코어 :  0.9943712363330277
'''