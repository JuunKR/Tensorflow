import tensorflow as tf
from tensorflow.python.ops.variables import global_variables_initializer
tf.compat.v1.set_random_seed(777)

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = x * W + b

# 실습
# tf09 1번의 방식 3가지고 hypothesis를 출력하시오!

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(hypothesis))
sess.close()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(hypothesis.eval())
sess.close()


sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(hypothesis.eval(session=sess))
sess.close()