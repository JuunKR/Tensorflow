import tensorflow as tf
from tensorflow.python.ops.gen_data_flow_ops import dynamic_partition
from tensorflow.python.ops.gen_math_ops import square
tf.set_random_seed(66)

x_train = [1,2,3] # w=1, b=0
y_train = [1,2,3]

# W = tf.Variable([1], dtype=tf.float32, name='test')
# B = tf.Variable([1], dtype=tf.float32)

W = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name='test')
b = tf.Variable(tf.random_normal([1], dtype=tf.float32))
#@ random_normal 정규분포에 의한 랜덤값
#@ 값이 같은 이유 random_seed를 정했기 때문

hypothesis = x_train * W + b 

loss = tf.reduce_mean(tf.square(hypothesis - y_train))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# print(sess.run(loss))
# print('dddddddd')
# print(sess.run(W), sess.run(b))
# print(sess.run(hypothesis))
# print(sess.run(hypothesis-y_train))
# print(sess.run(square(hypothesis-y_train)))
# print(sess.run(tf.reduce_mean(tf.square(hypothesis - y_train))))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(loss), sess.run(W), sess.run(b))

