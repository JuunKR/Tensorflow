import numpy as NP
import tensorflow as tf
tf.set_random_seed(66)

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]

y_data = [[0,0,1],   # 2
          [0,0,1],
          [0,0,1],
          [0,1,0],   # 1
          [0,1,0],
          [0,1,0],
          [1,0,0],   # 0
          [1,0,0]]


# 맹그러
# x 8, 4
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
# y 8, 3

# w
W = tf.Variable(tf.random.normal([4,3]), name='weight')  # 4, 3 그림 그려봐 노드숫자 인풋 노드 4개 아웃풋 노드 3개
# b 가 1,3?
b = tf.Variable(tf.random.normal([1, 3]), name='bias')  #  그림 그려봐 바이어스는 1개 아웃풋은 세개

# hypothesis = tf.matmul(x, w ) + b # 이것도 값은 세개가 나옴 하지만 합은 일일수도 아닐수도
hypothesis = tf.nn.softmax(tf.matmul(x, W) + b) # 값들의 합은 1


# categorical_crossentropy

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
#. 이전 소스의 optimizer와 train을 앞침

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_data, y:y_data})
    if step % 200 == 0:
        print(step, cost_val)

results = sess.run(hypothesis, feed_dict={x:[[1,11,7,9]]})
print(results, sess.run(tf.argmax(results, 1)))

