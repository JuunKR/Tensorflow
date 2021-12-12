import numpy as np
from icecream import ic
# np.save('../_save/_npy/k59_3_train_x.npy', arr=xy_train[0][0])
# np.save('../_save/_npy/k59_3_train_y.npy', arr=xy_train[0][1])
# np.save('../_save/_npy/k59_3_test_x.npy', arr=xy_test[0][0])
# np.save('../_save/_npy/k59_3_test_y.npy', arr=xy_test[0][1])


x_train = np.load('../_save/_npy/k59_3_train_x.npy')
y_train = np.load('../_save/_npy/k59_3_train_y.npy')

x_test = np.load('../_save/_npy/k59_3_test_x.npy')
y_test = np.load('../_save/_npy/k59_3_test_y.npy')


# ic(x_train)
# ic(y_train)
# ic(x_test)
# ic(y_test)

ic(x_train.shape)
ic(y_train.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# model.fit(x_train, y_train) 기존방식
hist = model.fit(x_train, y_train, epochs=30, validation_split=0.3)  
#@ validation_steps=4 이러한 옵션도 있음 - validation_data가 있어야함
#todos validation_steps 가 뭔지 알아오기                   
# 165/5 = 32 원에포당 들어가는 배치사이즈 훈련량?
# xy를 나눠줌 // x와 y가 쌍으로 되어있는 데이터의경우에는 이렇게


acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 위에거로 시각화 할것

print('acc : ', acc[-1])
print('val_acc : ', val_acc[:-1])




