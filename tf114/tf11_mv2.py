import tensorflow as tf

tf.set_random_seed(66)

x_data = [[73, 51, 66],                      # (5, 3)   ->  (3, ?) 랑 행렬연산해야함
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]
y_data = [[152], [185], [180], [205], [142]] # (5, 1)  

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random.normal([3,1]), name='weight') #@ w는 항상 x의 shape에 맞추기
b = tf.Variable(tf.random.normal([1]), name='bias')
#@ 행렬연산
#@ 들어가는 것에 따라 weight에 대한 shape도 구성해야함

hypothesis = tf.matmul(x, W) + b 

print('잘나왔닷')

#앞에꺼 붙여서 완성
cost = tf.reduce_mean(tf.square(hypothesis - y)) #mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5) #0.00001
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
        feed_dict={x: x_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, 'cost : ', cost_val, "\n", hy_val)