# 인공지능의 겨울을 극복하자
# perceptron -> mlp

import tensorflow as tf
tf.set_random_seed(66)


#@ 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]] 
y_data = [[0], [1], [1], [0]]          


#@ 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

#. 히든레이어 1
W = tf.Variable(tf.random.normal([2,32]), name='weight1')
b = tf.Variable(tf.random.normal([32]), name='bias1')

#. hypothsis = x * W + b
layer1 = tf.sigmoid(tf.matmul(x, W) + b)
# hypotheis = tf.matmul(x, W) + b # 이건 linear

#. 히든레이어 2
W2 = tf.Variable(tf.random.normal([32,16]), name='weight2')
b2 = tf.Variable(tf.random.normal([16]), name='bias2')

layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

#. 아웃풋레이어
W3 = tf.Variable(tf.random.normal([16,1]), name='weight3')
b3 = tf.Variable(tf.random.normal([1]), name='bias3')

hypotheis = tf.sigmoid(tf.matmul(layer2, W3) + b3)

# hypothsis = x * W + b

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

#, Accuracy :  1.0

sess.close()



