from numpy.lib.arraypad import pad
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
# from keras.utils import to_categorical
# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from tensorflow.python.ops.gen_batch_ops import batch
# 텐서에서 가져오지 말것

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly()) # False
print(tf.__version__) # 1.14.0 -> 2.4.1(base)

# tf.set_random_seed(66)

#@ 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28,28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28,28,1).astype('float32')/255

learning_rate = 0.0001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size)

x = tf.compat.v1.placeholder(tf.float32, [None, 28,28,1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

#@ 모델구성
# layer 1
w1 = tf.compat.v1.get_variable('w1', shape=[3,3,1,32]) #. 앞의 3,3은 커널사이즈, 1은 채널의 수, 32는 아웃풋 = 필터
                                            #. [kernel_size, input, output]
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
                        #. stride 앞뒤의 1은 차원을 맞추기 위해 가운데 두개가 변함 1,1 / 2,2 .....
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# model = Sequential()
# model.add(Conv2D(filtes=32, kernel_size=(3,3), strides=1, 
#             padding='same', input_shape=(28,28,1),                              #(low, cols, channel)
#             activation='relu'))            
# model.add(MaxPool2D())   

print(L1)
# Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
print(L1_maxpool)
# Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

# layer 2
w2 = tf.compat.v1.get_variable('w2', shape=[3,3,32,64])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2)
# Tensor("Selu:0", shape=(?, 14, 14, 64), dtype=float32)
print(L2_maxpool)
# Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

# layer 3
w3 = tf.compat.v1.get_variable('w3', shape=[3,3,64,128])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.elu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(L3)
# Tensor("Elu:0", shape=(?, 7, 7, 128), dtype=float32)
print(L3_maxpool)
# Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)

# layer 4
w4 = tf.compat.v1.get_variable('w4', shape=[2,2,128,64], )
                        # initializer=tf.contrib.layers.xavier_initializer()) #. 너무큰 가중치가 들어왔을 때 값이 폭발하는 것을 막기위해 규제하는 것. 
L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1,1,1,1], padding='VALID')
L4 = tf.nn.leaky_relu(L4)
L4_maxpool = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(L4)
# Tensor("LeakyRelu:0", shape=(?, 3, 3, 64), dtype=float32)
print(L4_maxpool)
# Tensor("MaxPool_3:0", shape=(?, 2, 2, 64), dtype=float32)

# Flatten
L_flat = tf.reshape(L4_maxpool, [-1, 2 * 2 * 64])
print("플래튼 : ", L_flat)
# Tensor("Reshape:0", shape=(?, 256), dtype=float32)

# layer5
w5 = tf.compat.v1.get_variable('w5', shape=[2*2*64, 64], )
                # initializer=tf.contrib.layers.xavier_initializer())                                                      
b5 = tf.Variable(tf.random.normal([64]), name='b1')
L5 = tf.matmul(L_flat, w5) + b5
L5 = tf.nn.selu(L5)
# L5 = tf.nn.dropout(L5, keep_prob=0.2)
print(L5)
# Tensor("dropout/mul_1:0", shape=(?, 64), dtype=float32)

# layer6 DNN
w6 = tf.compat.v1.get_variable('w6', shape=[64, 32], )
                # initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random.normal([32]), name='b2')
L6 = tf.matmul(L5, w6) + b6
L6 = tf.nn.selu(L6)
# L6 = tf.nn.dropout(L6, keep_prob=0.2)
print(L6)
# Tensor("dropout_1/mul_1:0", shape=(?, 32), dtype=float32)

# layer7 softmax
w7 = tf.compat.v1.get_variable('w7', shape=[32, 10])
b7 = tf.Variable(tf.compat.v1.random_normal([10]), name='b3')
L7 = tf.matmul(L6, w7) + b7
hypothesis = tf.nn.softmax(L7)
print(hypothesis)
# Tensor("dropout_2/mul_1:0", shape=(?, 10), dtype=float32)

#@ 컴파일 훈련
# categorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01,).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# learning_rate = 0.001
# training_epochs = 15
# batch_size = 100
# total_batch = int(len(x_train) / batch_size)

for epoch in range(training_epochs):
    avg_loss = 0
    
    for i in range(total_batch):     # 몇번 돌까? 600번 돌지
        start = i * batch_size
        end = start + batch_size
        batch_x, batch_y = x_train[start:end], y_train[start:end]
        
        feed_dict = {x:batch_x, y:batch_y}

        batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

        avg_loss += batch_loss/total_batch

    print('Epoch : ', '%04d' %(epoch + 1), 'loss :  {:.9f}'.format(avg_loss))   

print('훈련 끝!!')

prediction = tf.equal(tf.compat.v1.argmax(hypothesis, 1), tf.compat.v1argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('ACC : ', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))


# 0.96이상
