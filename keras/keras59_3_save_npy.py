import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from icecream import ic

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
   
    
)
#테스트 데이터는 통상적으로 안함

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../_data/brain/train',
    target_size=(150,150),
    batch_size=200, # [1., 0., 1., 0., 1.] y값
    class_mode='binary',
    # shuffle=False 
)
# 안에 있는 폴더들 기준으로 다시 나눔 여기서는 add 0/ normal 1 -  종류별로 라벨링

#@ Found 160 images belonging to 2 classes. 위에 실행하면 이게 나와용

xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test',
    target_size=(150,150),
    batch_size=200, # [1., 0., 1., 0., 1.] y값
    class_mode='binary',
    # shuffle=False 
)
#@ Found 120 images belonging to 2 classes.

# print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001A910898550>
# print(xy_train[0])
# print(type(xy_train))

# ic(xy_train[0][0]) # x값
# ic(xy_train[0][1]) # y값
# # ic(xy_train[0][2]) # 없어
# ic(xy_train[0][0].shape, xy_train[0][1].shape) # (5, 150, 150, 3) 가장앞에 오인 이유는 배치사이즈를 5로 줘서 나머지는 [1] [2].... 에 들어가 있음 (5,)
# ic(xy_train[31][1])

# ic(type(xy_train))                  # ic| type(xy_train): <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# ic(type(xy_train[0]))               # ic| type(xy_train[0]): <class 'tuple'>
# ic(type(xy_train[0][0]))            # ic| type(xy_train[0][0]): <class 'numpy.ndarray'>
# ic(type(xy_train[0][1]))            # ic| type(xy_train[0][1]): <class 'numpy.ndarray'>

#@ 배치사이즈를 위에서 500으로 늘리면 다 나옵니다. 대신 [0][1]은 없어지쥬
ic(xy_train[0][0]) # x값
ic(xy_train[0][1]) # y값
ic(xy_train[0][0].shape, xy_train[0][1].shape) # (5, 150, 150, 3) 가장앞에 오인 이유는 배치사이즈를 5로 줘서 나머지는 [1] [2].... 에 들어가 있음 (5,)


np.save('../_save/_npy/k59_3_train_x.npy', arr=xy_train[0][0])
np.save('../_save/_npy/k59_3_train_y.npy', arr=xy_train[0][1])
np.save('../_save/_npy/k59_3_test_x.npy', arr=xy_test[0][0])
np.save('../_save/_npy/k59_3_test_y.npy', arr=xy_test[0][1])