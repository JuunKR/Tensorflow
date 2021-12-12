# overfit 극복하자!!!!
# 1. 전체 훈련 데이터를 마니 마니
# 2. normalization 정규화  // 현재 우리는 모델 fit에서 overfit되는중 기존에 우리는 전처리에서 정규화를 해줌 // 모델에서 다음레이어로 가기전에 엑티베이션으로 감싸서 보내줌 그 값자체도 정규화하자 = 레이어별로 정규화를 해주자. 
# 3. dropout

# Fully connected layer
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler



datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=9) 

x_train = x_train.reshape(309,10)
x_test = x_test.reshape(133, 10)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # 기존에 두개로 나누는 방식을 하나로 합침 #! but train만 해야한다 
x_test = scaler.transform(x_test)


x_train = x_train.reshape(309, 10, 1, 1)
x_test = x_test.reshape(133, 10, 1, 1)


# from sklearn.preprocessing import OneHotEncoder

# y_train = y_train.reshape(-1,1) #! 여기 리쉐입 하는 이유  5만개짜리 백터 하나 사이킥런에 있는것은 대부분 2차원으로 받아들임 -1은 전체를 의미함
# y_test = y_test.reshape(-1,1)
# one = OneHotEncoder(sparse=False) #sparse=False
# y_train = one.fit_transform(y_train)
# y_test = one.fit_transform(y_test)

# ic(y_train.shape) #@ (50000, 100)



#2 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(1,1),
        activation='relu' ,input_shape=(10,1,1)))

model.add(Dropout(0.2))
model.add(Conv2D(64, (1,1), activation='relu'))
   

model.add(Conv2D(128, (1,1), activation='relu'))
model.add(Conv2D(128, (1,1),  activation='relu'))


model.add(Conv2D(64, (1,1), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (1,1), activation='relu'))


model.add(Flatten()) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.summary()
model.save_weights('./_save/keras46_3_save_weights_1.h5')

# model.save('./_save/keras46_3_save_model_1.h5')

# model = load_model('./_save/keras46_3_save_model_1.h5')
# model = load_model('./_save/keras46_3_save_model_2.h5')


#3. 컴파일 훈련
# model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=3)

import time
start_time = time.time()
hist = model.fit(x_train,y_train, epochs=100, batch_size=64, validation_split=0.25, callbacks=[es])
# model.save('./_save/keras46_3_save_model_2.h5')
model.save_weights('./_save/keras46_3_save_weights_2.h5')
end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test,y_test, batch_size=64) 

print('걸린시간 : ', end_time)
print('loss값 : ', loss[0])
print('acc값: ', loss[1])

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