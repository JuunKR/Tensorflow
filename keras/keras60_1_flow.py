from tensorflow.keras.datasets import fashion_mnist
import numpy as np

(x_train, y_train), (y_train, y_test) = fashion_mnist.load_data()

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
argument_size=50    
x_data = train_datagen.flow(
    np.tile((x_train[0]).reshape(28*28), argument_size).reshape(-1, 28, 28, 1),
    np.zeros(argument_size),
    batch_size=argument_size,
    shuffle=False
).next()                        # iterator 방식으로 반환 ! 

print(type(x_data)) # <class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'>
                    # -> next후 튜플이 됨
print(type(x_data[0])) # <class 'tuple'>
                    # -> next후 <class 'numpy.ndarray'>
print(type(x_data[0][0])) # <class 'numpy.ndarray'>
print(x_data[0][0].shape) # (100, 28, 28, 1) x 값
                         # 28, 28, 1
# print(x_data[0][1].shape) # (100,) y 값

print(x_data[0].shape) # (100, 28, 28, 1)
print(x_data[1].shape) # (100,) y 값

#@ iterator
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for  i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')


# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(9,9))
# ax1 = fig.add_subplot(2,1,1)
# ax2 = fig.add_subplot(2,1,2)
# for  i in range(4):
#     plt.subplot(2,2,i)
#     plt.axis('off')
#     plt.imshow(x_data[0][i], cmap='gray')


plt.show()