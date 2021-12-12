from datetime import datetime
from typing import Sequence
from numpy.lib.arraysetops import unique
from tensorflow.keras.datasets import fashion_mnist

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras import activations

#np.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=No)

# np.set_printoptions(threshold=np.inf )

# 데이터 전처리
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# ic(x_train.shape, y_train.shape) #@ (60000, 28, 28) (60000,)
# ic(x_test.shape, y_test.shape)   #@ (10000, 28, 28) (10000,)

# ic(x_train)
# ic(y_train)
# ic(x_train[0], y_train[0])
# ic(unique(y_train)) #@ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# plt.imshow(x_train[6000], 'gray')
# plt.show()

# https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=ksg97031&logNo=221302568510

#@ 8비트로 표현되어 0 ~ 255 까지 만 있음
x_train = x_train.reshape(60000, 28* 28)
x_test = x_test.reshape(10000, 28*28 )

# (2,2,2,1)
# a = [[[1,2],[3,4]], [[5,6],[7,8]]]
# a = np.array(a)
# ic(a)
# ic(a.shape) #(2,2,2,1)

# a =a.reshape(2,2,2,1)
# print(a)
# ic(a.shape) [[[[1]   [2]]  [[3]  [4]]] [[[5]  [6]]  [[7]   [8]]]]
#@ b = [ [ [ [2],[3] ]  [ [2],[3] ] ] [ [ [2],[3] ]  [ [2],[3]] ] ]
#@ ic(b)

# ic(a[0:2])


# ic(len(y_train))
# ic(y_train.shape)
# y_train = y_train.reshape(-1,1)
# ic(len(y_train))
# ic(y_train.shape)
# # ic(y_train)

from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1) #! 여기 리쉐입 하는 이유
y_test = y_test.reshape(-1,1)
one = OneHotEncoder(sparse=False) #sparse=False
y_train = one.fit_transform(y_train)
y_test = one.fit_transform(y_test)

#(y_train.shape) #@ (60000, 10)
print(x_train.shape)
print(x_test.shape)
'''
(60000, 784)
(10000, 784)
'''
x_train =x_train.reshape(60000, 784, 1)
x_test =x_test.reshape(10000, 784, 1)

#2 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, LSTM

model = Sequential()
model.add(Conv1D(64, 2, input_shape=(784,1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# model = Sequential()
# model.add(Conv2D(filters=128, kernel_size=(2,2), padding='same',
#         activation='relu' ,input_shape=(28,28,1)))
# model.add(Conv2D(64, (2,2), activation='relu'))
# model.add(Conv2D(64, (2,2), activation='relu'))
# model.add(MaxPool2D())          
# model.add(Conv2D(32, (2,2), activation='relu'))
# model.add(Conv2D(32, (2,2), activation='relu'))
# model.add(Flatten()) 
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model.summary()

# #3 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=3)

import time
start_time = time.time()
hist = model.fit(x_train,y_train, epochs=100, batch_size=1000, validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test,y_test) 
print('걸린시간 : ', end_time)
print('loss값 : ', loss[0])
print('acc값: ', loss[1])

'''
cnn
ic| 'loss값 : ', loss[0]: 0.30474328994750977
ic| 'acc값: ', loss[1]: 0.9047999978065491
'''

'''
dnn
ic| 'loss값 : ', loss[0]: 0.4018581211566925
ic| 'acc값: ', loss[1]: 0.8651000261306763
'''