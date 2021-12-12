from numpy.lib.arraypad import pad
import tensorflow as tf
import numpy as np
# from tensorflow.keras.utls import to_categorical
from keras.utils import to_categorical
# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
# 텐서에서 가져오지 말것


tf.set_random_seed(66)

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28,28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28,28,1).astype('float32')/255

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size)

x = tf.placeholder(tf.float32, [None, 28,28,1])
y = tf.placeholder(tf.float32, [None, 10])

# 모델구성

w1 = tf.get_variable('w1', shape=[3,3,1,32]) #. 앞의 3,3은 커널사이즈, 1은 채널의 수, 32는 아웃풋 = 필터
                                            #. [kernel_size, input, output]
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
                        #. stride 앞뒤의 1은 차원을 맞추기 위해 가운데 두개가 변함 1,1 / 2,2 .....
print(w1)
# <tf.Variable 'w1:0' shape=(3, 3, 1, 32) dtype=float32_ref>
print(L1)
# Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)

# model = Sequential()
# model.add(Conv2D(filtes=32, kernel_size=(3,3), strides=1, 
#             padding='same', input_shape=(28,28,1)))

################################ get_variable 연구 #####################################
# w2 = tf.Variable(tf.random_normal([3,3,1,32]), dtype=tf.float32)
# w3 = tf.Variable([1], dtype= tf.float32)
#. 위 두개의 차이점 ; 초기값 지정 여부, name과 shape 설정 여부

# print(w1)
# # <tf.Variable 'w1:0' shape=(3, 3, 1, 32) dtype=float32_ref>
# sess = Session()

# sess.run(tf.global_variables_initializer())
# # print(sess.run(w1))

# print(np.min(sess.run(w1)))
# print('===========================================================')
# print(np.max(sess.run(w1)))
# print('===========================================================')
# print(np.mean(sess.run(w1)))
# print('===========================================================')
# print(np.median(sess.run(w1)))
###########################################################################################

