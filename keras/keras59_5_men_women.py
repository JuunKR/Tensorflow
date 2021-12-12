# 실습
# men women 데이터로 모델링 구성할 것!!!

# 실습 2
# 본인 사진으로 predict 할 것!!!

#todos 내사진으로 내가 남자인지, 여자인지 //  최태영은 몇% 확률로 여자입니다.!! 스크린샷.

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
# #테스트 데이터는 통상적으로 안함

test_datagen = ImageDataGenerator(rescale=1./255)


# xy_data = train_datagen.flow_from_directory(
#     '../_data/archive',
#     target_size=(150,150),
#     batch_size=4000, # [1., 0., 1., 0., 1.] y값
#     class_mode='binary',
#     # shuffle=False 
# )

# xy_test = test_datagen.flow_from_directory(
#     '../_data/archive',
#     target_size=(150,150),
#     batch_size=4000, # [1., 0., 1., 0., 1.] y값
#     class_mode='binary',
#     # shuffle=False 
# )
# print(xy_test[0])



# # # # 안에 있는 폴더들 기준으로 다시 나눔 여기서는 add 0/ normal 1 -  종류별로 라벨링
# # # #@ Found 3309 images belonging to 2 classes.

# np.save('../_save/_npy/k59_5_x_data.npy', arr=xy_test[0][0])
# np.save('../_save/_npy/k59_5_y_data.npy', arr=xy_test[0][1])
# np.save('../_save/_npy/k59_5_onebin_0.npy', arr=xy_test[0][0])
# np.save('../_save/_npy/k59_5_onebin_1.npy', arr=xy_test[0][1])

# exit()
#. 넘파이 불러오자!
x = np.load('../_save/_npy/k59_5_x_data.npy')
y = np.load('../_save/_npy/k59_5_y_data.npy')

print(x.shape)

predict = np.load('../_save/_npy/k59_5_onebin_0.npy')

exit()
#. 나눠!
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=9) 



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

# ic(x_train.shape) # ic| x_train.shape: (2316, 150, 150, 3)
# ic(y_train.shape) # ic| y_train.shape: (2316,)

# ic(x_test.shape) # ic| x_test.shape: (993, 150, 150, 3)

#. 모델이다
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

#. 컴파일, 훈련시켜라!
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# 얼리는 fit 위
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',patience=10, mode='auto', verbose=1, restore_best_weights=True)

import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=30, validation_split=0.3, batch_size=200)  
end_time = time.time() - start_time


#. 평가, 예측 가즈아!

loss = model.evaluate(x_test,y_test) 

print('걸린시간 : ', end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_predict = model.predict([predict])
print('남자일 확률 : ',y_predict)
'''

걸린시간 :  164.91008687019348
loss :  0.7560098171234131
acc :  0.5961732268333435
남자일 확률 :  [[0.1484249]]
'''