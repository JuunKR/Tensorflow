# 이진분류이나 다중분류로 풀것

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from icecream import ic


# # #. 넘파이로 만들거당
# test_datagen = ImageDataGenerator(rescale=1./255)

# xy_data = test_datagen.flow_from_directory(
#     '../_data/horse-or-human',
#     target_size=(150,150),
#     batch_size=1030, # [1., 0., 1., 0., 1.] y값
#     class_mode='categorical',
#     # shuffle=False 
# )

# np.save('../_save/_npy/k59_7_x_data.npy', arr=xy_data[0][0])
# np.save('../_save/_npy/k59_7_y_data.npy', arr=xy_data[0][1])

#. 넘파이 불러오자!
x = np.load('../_save/_npy/k59_7_x_data.npy')
y = np.load('../_save/_npy/k59_7_y_data.npy')

print(type(x))
print(type(y))

#. 나눠!
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=9) 

print(x_train.shape)
exit()

    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

# ic(x_train.shape) # (718, 150, 150, 3)
# ic(y_train.shape) # (718, 2)

# ic(x_test.shape) # (309, 150, 150, 3)


#. 모델이다
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

#. 컴파일, 훈련시켜라!
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

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

'''
걸린시간 :  37.718353509902954
loss :  0.42628902196884155
acc :  0.8996763825416565
'''