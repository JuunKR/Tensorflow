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

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)


from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1) #! 여기 리쉐입 하는 이유  5만개짜리 백터 하나 사이킥런에 있는것은 대부분 2차원으로 받아들임 -1은 전체를 의미함
y_test = y_test.reshape(-1,1)
one = OneHotEncoder(sparse=False) #sparse=False
y_train = one.fit_transform(y_train)
y_test = one.fit_transform(y_test)

ic(y_train.shape) #@ (50000, 100)

#2 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), padding='valid',
        activation='relu' ,input_shape=(32,32,3)))

model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(MaxPool2D())      

model.add(Conv2D(128, (2,2), padding='valid', activation='relu'))
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(MaxPool2D())  

model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), padding='same',  activation='relu'))
model.add(MaxPool2D())  #! (3,3) 이과정 알아야해 560개의 노드?



# #@ 여기부터가 달라요
# #@ 기본엔 여기까지 CNN을 끝냄 약 500개정도가 flatten이 나옴 but 마지막에 100개의 결과 값이 나와야함 기존에 128로 이어져서 5만번의 연산이 이어짐 그 다음엔 만번 또 만번.. flatten다음에 전부다 약 7만번 정도 연산을함 근데 이연산이 정확한걸까? 오히려 cnn값이 정확할 수 도 있지 않나.. 값들이 터짐 안좋은 결과가 나올 수도 있음 늘렸다 줄였다 할필요 없이 cnn결과를 그냥 믿어보자


# #@ 그럴바에 500개를 비율에 따라서 100개에 직접 보내준다. cnn연산을 믿는 다는 가정에 5개의 average를 구해서 하나주고 그렇게 500를 100개에 줌.
# #@ MNIST softmax 마지막에 받은 중에 가장 큰거를 기준으로 값을 결정함. 
# #@ 다시 설명 

# # model.add(Flatten()) 
# # model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(128, activation='relu'))
# # model.add(Dense(100, activation='softmax'))

model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='softmax'))
model.summary()

#3. 컴파일 훈련
# model.summary()

from tensorflow.keras.optimizers import Adam

optimizer = Adam(lr=0.1)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc']) #. 로스를 최적화하기위해 아담의lr을 0.1그러나 fit에서 실행이됨

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5) #. verbose는 es랑 같게 factor은 갱신이 없으면 0.5만큼 lr 감소

import time
start_time = time.time()
hist = model.fit(x_train,y_train, epochs=300, batch_size=512, validation_split=0.25, callbacks=[es, reduce_lr])
end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test,y_test, batch_size=64) 

print('걸린시간 : ', end_time)
ic('loss값 : ', loss[0])
ic('acc값: ', loss[1])

# '''
# ic| 'loss값 : ', loss[0]: 2.9052181243896484
# ic| 'acc값: ', loss[1]: 0.3467000126838684
# '''

# '''
# +MINMAX
# 걸린시간 :  146.67364525794983
# ic| 'loss값 : ', loss[0]: 2.9243733882904053
# ic| 'acc값: ', loss[1]: 0.3433000147342682

# '''

# '''
# 모델수정
# 걸린시간 :  150.6033854484558
# ic| 'loss값 : ', loss[0]: 2.7462034225463867
# ic| 'acc값: ', loss[1]: 0.3668000102043152
# '''

# '''
# gap
# ic| 'loss값 : ', loss[0]: 2.253070116043091
# ic| 'acc값: ', loss[1]: 0.44769999384880066
# '''


# ##### plt 시각화
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,5))

# #1
# plt.subplot(2,1,1)
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend(loc='upper right')

# #2
# plt.subplot(2,1,2)
# plt.plot(hist.history["acc"])
# plt.plot(hist.history['val_acc'])
# plt.grid()
# plt.title('acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['acc', 'val_acc'])

# plt.show()