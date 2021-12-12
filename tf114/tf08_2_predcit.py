# 실습
# 1. [4]
# 2. [5,6]
# 3. [5,6,7,]
# 예측
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import square
from tensorflow.python.training import optimizer


tf.compat.v1.set_random_seed(66)

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

hypothesis = x_train * W + b
loss = tf.reduce_mean(square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(4001):
       _, loss_val, W_val, b_val = sess.run([train, loss, W, b],
                                feed_dict={x_train:[1,2,3], y_train:[1,2,3,]})
       if step % 20 == 0:
           print(step, loss_val, W_val, b_val)

# predict 하는 코드를 추가하시오

# x_test = [4]
# x_test = [5, 6]
x_test = [5, 6, 7]

hypothesis = x_test * W_val + b_val

print(hypothesis)