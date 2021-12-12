from datetime import datetime
from tensorflow.keras.datasets import fashion_mnist
from icecream import ic
import matplotlib.pyplot as plt


# 데이터 전처리
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

ic(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
ic(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

ic(x_train)
ic(y_train)
ic(x_train[:5], y_train[5])

plt.imshow(x_train[59999], 'gray')
plt.show()