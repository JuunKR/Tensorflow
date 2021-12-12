# 실습
# 덧셈
# 뺄셈
# 곱셈
# 나눗셈
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf



node1 = tf.constant(2.0)
node2 = tf.constant(3.0)
node3 = tf.add(node1, node2)
node4 = tf.subtract(node1, node2)
node5 = tf.multiply(node1, node2)
node6 = tf.divide(node1, node2)

sess = tf.compat.v1.Session()

print('node1 + node2 = ', sess.run(node3))
print('node1 - node2 = ', sess.run(node4) )
print('node1 * node2 = ', sess.run(node5) )
print('node1 / node2 = ', sess.run(node6) )

#@ Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
#@ 더빠른 연산을 진행할 수 있는 방법이 있어

#@ 무시
#@ import os
#@ os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'