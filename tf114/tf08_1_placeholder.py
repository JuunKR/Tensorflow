import tensorflow as tf
tf.compat.v1.set_random_seed(66)

#@ x,y 를 placeholder로
x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32, name='test')
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)


hypothesis = x_train * W + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b],       #@ 실질적으로 train하나로 다 계산이 됨 단지 나눠서 보여줌
                   feed_dict={x_train:[1,2,3], y_train:[1,2,3]})
    if step % 20 == 0:
        print(step, loss_val, W_val, b_val)


