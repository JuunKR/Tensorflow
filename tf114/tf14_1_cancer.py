# 실습 
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score


tf.set_random_seed(66)

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

#. 실습 : 맹그러!!

y = y.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.8, random_state=66, shuffle=True)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

# W = tf.Variable(tf.random_normal([30, 1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')

W = tf.Variable(tf.zeros([30,1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, W) + b)

cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=9e-7)
optimizer = tf.train.AdamOptimizer(learning_rate=2e-7) #아담

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

for step in range(10001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
        feed_dict={x:x_train, y:y_train})
    if step % 200 == 0:
        print(step, 'cost : ', cost_val)


h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_test, y:y_test})

print(f'predict : {h[0:5]} \n "original value: \n{c[0:5]} \naccuracy: : {a}')

sess.close()
