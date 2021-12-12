from tensorflow.keras.datasets import cifar100
from icecream import ic
import matplotlib.pyplot as plt

# 데이터 전처리
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# ic(x_train.shape, y_train.shape) #@ (50000, 32, 32, 3), (50000, 1)
# ic(x_test.shape, y_test.shape)   #@ (10000, 32, 32, 3), (10000, 1)

x_train = x_train.reshape(50000, 32, 32, 3)/255
x_test = x_test.reshape(10000, 32, 32, 3)/255

from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1) #! 여기 리쉐입 하는 이유
y_test = y_test.reshape(-1,1)
one = OneHotEncoder(sparse=False) #sparse=False
y_train = one.fit_transform(y_train)
y_test = one.fit_transform(y_test)

ic(y_train.shape) #@ (50000, 100)

#2 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), padding='same',
        activation='relu' ,input_shape=(32,32,3)))
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(MaxPool2D())          
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
model.add(MaxPool2D())          
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Flatten()) 
model.add(Dense(64, activation='relu'))
model.add(Dense(100, activation='softmax'))

# model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=3)

model.fit(x_train,y_train, epochs=500, batch_size=128, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test,y_test) 
ic('loss값 : ', loss[0])
ic('acc값: ', loss[1])

'''
ic| 'loss값 : ', loss[0]: 2.9052181243896484
ic| 'acc값: ', loss[1]: 0.3467000126838684
'''