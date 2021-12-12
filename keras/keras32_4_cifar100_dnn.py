# overfit 극복하자!!!!
# 1. 전체 훈련 데이터를 마니 마니
# 2. normalization 정규화  // 현재 우리는 모델 fit에서 overfit되는중 기존에 우리는 전처리에서 정규화를 해줌 // 모델에서 다음레이어로 가기전에 엑티베이션으로 감싸서 보내줌 그 값자체도 정규화하자 = 레이어별로 정규화를 해주자. 
# 3. dropout

# Fully connected layer

from numpy.core.numerictypes import ScalarType
from tensorflow.keras.datasets import cifar100
from icecream import ic
import matplotlib.pyplot as plt

# 데이터 전처리
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# ic(x_train.shape, y_train.shape) #@ (50000, 32, 32, 3), (50000, 1)
# ic(x_test.shape, y_test.shape)   #@ (10000, 32, 32, 3), (10000, 1)

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # 기존에 두개로 나누는 방식을 하나로 합침 #! but train만 해야한다 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32* 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)


from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1) #! 여기 리쉐입 하는 이유  5만개짜리 백터 하나 사이킥런에 있는것은 대부분 2차원으로 받아들임 -1은 전체를 의미함
y_test = y_test.reshape(-1,1)
one = OneHotEncoder(sparse=False) #sparse=False
y_train = one.fit_transform(y_train)
y_test = one.fit_transform(y_test)

ic(y_train.shape) #@ (50000, 100)

#2 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

model = Sequential()
model.add(Dense(128, input_shape=(32*32*3,)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))

# model = Sequential()
# model.add(Conv2D(filters=128, kernel_size=(2,2), padding='valid',
#         activation='relu' ,input_shape=(32,32,3)))

# model.add(Dropout(0.2))
# model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
# model.add(MaxPool2D())      

# model.add(Conv2D(128, (2,2), padding='valid', activation='relu'))
# model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
# model.add(MaxPool2D())  

# model.add(Conv2D(64, (2,2), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2,2), padding='same',  activation='relu'))
# model.add(MaxPool2D())  #! (3,3) 이과정 알아야해 560개의 노드?

# model.add(Flatten()) 
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(100, activation='softmax'))

#3. 컴파일 훈련
# model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=3)

import time
start_time = time.time()
hist = model.fit(x_train,y_train, epochs=100, batch_size=64, validation_split=0.25, callbacks=[es])
end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test,y_test, batch_size=64) 

print('걸린시간 : ', end_time)
ic('loss값 : ', loss[0])
ic('acc값: ', loss[1])

'''
ic| 'loss값 : ', loss[0]: 2.9052181243896484
ic| 'acc값: ', loss[1]: 0.3467000126838684
'''

'''
+MINMAX
걸린시간 :  146.67364525794983
ic| 'loss값 : ', loss[0]: 2.9243733882904053
ic| 'acc값: ', loss[1]: 0.3433000147342682

'''

'''
모델수정
걸린시간 :  150.6033854484558
ic| 'loss값 : ', loss[0]: 2.7462034225463867
ic| 'acc값: ', loss[1]: 0.3668000102043152
'''

'''
dnn
ic| 'loss값 : ', loss[0]: 3.475829601287842
ic| 'acc값: ', loss[1]: 0.1745000034570694
'''

##### plt 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

#1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

#2
plt.subplot(2,1,2)
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()