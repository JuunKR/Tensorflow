from tensorflow.keras.datasets import fashion_mnist
import numpy as np


#@ 엠니스트 증폭시킬거야 1만장까지
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.10,
    shear_range=0.5,
    fill_mode='nearest',
)

# train_datagen = ImageDataGenerator(rescale=1./255,)
# xy_train = train_datagen.flow_from_directory(
#     '../_data/brain/train',
#     target_size=(150,150),
#     batch_size=5, # [1., 0., 1., 0., 1.] y값
#     class_mode='binary',
#     # shuffle=False 
# )

#@ 1. ImageDataGenerator 를 정의
#@ 2. 파일에서 땡겨올려면 -> flow_from_driectiory() // x,y 가 튜플형태로 뭉쳐있어
#@ 3. 데이터에서 땡겨올려면 -> flow()               // x,y 가 나뉘어있어


#증폭 사진을 돌리고 좌우반전하고 등등 해서 100장 을 더 만들거임
#@ np.tile 배열을 반복하고 옆으로 추가로 축을 생성
augment_size=40000

randidx = np.random.randint(x_train.shape[0], size=augment_size)
# 랜덤하게 4만개 가져오겠다 ! ?
 
print(x_train.shape[0]) # 60000
print(randidx) # [ 673 3993 3895 ... 4156 5515 2899]
print(randidx.shape) # (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)


x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
            batch_size=augment_size, shuffle=False).next()[0]

print(x_augmented.shape)

#넘파이 엮을 땐 concatenate
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# (100000, 28, 28, 1) (100000,)
# (10000, 28, 28) (10000,)

# 모델 완성
# 비교대상 lss val_loss, acc, val_acc
# 기존 fashion mnist 와 결과 비교

x_train = x_train.reshape(100000, 28* 28)
x_test = x_test.reshape(10000, 28* 28)

#2 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Dense(100, input_shape=(28*28,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=3)

model.fit(x_train,y_train, epochs=500, batch_size=128, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test,y_test) 
print('loss값 : ', loss[0])
print('acc값: ', loss[1])
