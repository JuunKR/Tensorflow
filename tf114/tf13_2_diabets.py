import tensorflow as tf
import numpy as np
from tensorflow.python.training import optimizer
tf.set_random_seed(66)
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

datasets = load_diabetes()

x_data = datasets.data # (442, 10) 
y_data = datasets.target # (442, )

# print(x.shape, y.shape)

y_data = y_data.reshape(-1,1) 

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size = 0.2,  random_state = 66)

x = tf.placeholder(tf.float32, shape=(None, 10))
y = tf.placeholder(tf.float32, shape=(None, 1))

W = tf.Variable(tf.zeros([10, 1], name='weight'))
b = tf.Variable(tf.zeros([1], name='bias'))

hypothesis = tf.matmul(x, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.9)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                    feed_dict={x:x_train, y:y_train})
    if epoch % 20 == 0:
        print(epoch, "cost : ", cost_val)

predicted = sess.run(hypothesis, feed_dict={x:x_test})

r2 = r2_score(y_test, predicted)
print('r2 : ', r2)

sess.close()


