import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from datetime import datetime
import tensorflow as tf
#완성하시오!!
#acc 0.8 이상 만들것!!!


# 데이터 전처리
datasets = load_wine()

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (178, 13) (178,)
print(y)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y) 
print(y[:5])
print(y.shape) # (178, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=66) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

# 모델구성
model = Sequential()
model.add(Dense(128,activation='relu', input_shape=(13,)))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(3, activation='softmax')) 
model.summary()

# Define the Keras TensorBoard callback.
logdir="logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

# 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1) # 

model.fit(x_train,y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es, tensorboard_callback])

# 평가 예측
loss = model.evaluate(x_test,y_test) 
print('loss : ', loss[0])
y_predict = model.predict(x_test)
print(y_predict)

'''
Epoch 00028: early stopping
2/2 [==============================] - 0s 0s/step - loss: 0.0833 - accuracy: 0.9815
loss :  0.08329673856496811
'''