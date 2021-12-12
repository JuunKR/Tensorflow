# y = wx + b

#@ x, y -> 입력되는 값 placeholder
#@ w, b -> 변수

import os
from numpy.core.arrayprint import dtype_is_implied
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


tf.compat.v1.set_random_seed(66)

x_train = [1,2,3,] # w = 1, b = 0
y_train = [1,2,3,]

W = tf.Variable([1], dtype=tf.float32, name='test')
b = tf.Variable([1], dtype=tf.float32) 
#@ 위의 1들은 랜덤하게 내맘대로 넣어준 값

hypothesis = x_train * W + b #@ 모델 구현
#@ y의 결괏값
#@ f(x) = wd + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) #@ mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2000):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(W), sess.run(b))

