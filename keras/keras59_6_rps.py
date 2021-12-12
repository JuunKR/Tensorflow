import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from icecream import ic

# #. 넘파이로 만들거당
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=5,
#     zoom_range=1.2,
#     shear_range=0.7,
#     fill_mode='nearest',
# )
# # 테스트 데이터는 통상적으로 안함

test_datagen = ImageDataGenerator(rescale=1./255)

# # xy_data = train_datagen.flow_from_directory(
# #     '../_data/rps',
# #     target_size=(150,150),
# #     batch_size=25600, # [1., 0., 1., 0., 1.] y값
# #     class_mode='binary',
# #     # shuffle=False 
# # )

# xy_test = test_datagen.flow_from_directory(
#     '../_data/rps',
#     target_size=(150,150),
#     batch_size=2600, # [1., 0., 1., 0., 1.] y값
#     class_mode='categorical',
#     # shuffle=False 
#     classes=['paper','rock','scissors'],
# )

# np.save('../_save/_npy/k59_6_x_data.npy', arr=xy_test[0][0])
# np.save('../_save/_npy/k59_6_y_data.npy', arr=xy_test[0][1])

#. 넘파이 불러오자!
x = np.load('../_save/_npy/k59_6_x_data.npy')
y = np.load('../_save/_npy/k59_6_y_data.npy')
# exit()
ic(x.shape)
ic(y.shape)


#. 나눠!
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=9) 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

#. 모델이다
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3)))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

#. 컴파일, 훈련시켜라!
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# 얼리는 fit 위
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',patience=10, mode='auto', verbose=1, restore_best_weights=True)

import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=7, validation_split=0.3, batch_size=50)  
end_time = time.time() - start_time


#. 평가, 예측 가즈아!

loss = model.evaluate(x_test,y_test) 

print('걸린시간 : ', end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])