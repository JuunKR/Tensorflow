import tensorflow as tf
import matplotlib.pyplot as plt


x = [1., 2., 3.]
y = [2., 4., 6.]

W = tf.placeholder(tf.float32)

hypothesis = x * W

cost = tf.reduce_mean(tf.square(hypothesis - y))

w_history = []
#@ model.fit에서 나오는 것과 동일
cost_history = []
#@ weight와 loss 값의 epo당 변화의 확인

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i
        curr_cost = sess.run(cost, feed_dict={W:curr_w})

        w_history.append(curr_w)
        cost_history.append(curr_cost)


print('=================== W history =============================')
print(w_history)
print("============================ loss history ====================")
print(cost_history)
print("=====================================================")

plt.plot(w_history, cost_history)
plt.xlabel('Weight')
plt.ylabel('loss')
plt.show()