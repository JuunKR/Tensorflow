import tensorflow as tf
tf.set_random_seed(66)


#@ 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]] 
y_data = [[0], [1], [1], [0]]          


#@ 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random.normal([2,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypotheis = tf.sigmoid(tf.matmul(x, W) + b)

cost = -tf.reduce_mean(y*tf.log(hypotheis) + (1-y) * tf.log(1-hypotheis)) # binary_crossentropy 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

#@ 훈련
for epochs in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypotheis, train],
            feed_dict={x:x_data, y:y_data})
    if epochs % 200 == 0:
        print(epochs, "cost : ", cost_val, "\n", hy_val)


#@ 평가, 예측
predicted = tf.cast(hypotheis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

c, a = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})

print("==========================================================")
print('예측값 : \n', hy_val, 
        "\n 원래값 : \n", c, "\n Accuracy : ", a)

sess.close()



