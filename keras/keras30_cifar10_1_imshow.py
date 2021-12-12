from tensorflow.keras.datasets import cifar10
from icecream import ic
import matplotlib.pyplot as plt

# 이미지가 32,32,3

# 데이터 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

ic(x_train.shape, y_train.shape) # (50000, 32, 32, 3), (50000, 1)
ic(x_test.shape, y_test.shape)   # (10000, 32, 32, 3), (10000, 1)

ic(x_train)
ic(y_train)
ic(x_train[:5], y_train[5])

plt.imshow(x_train[49999], 'gray')
plt.show()