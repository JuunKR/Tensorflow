import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print(node3)
#. Tensor("Add:0", shape=(), dtype=float32)

sess = tf.compat.v1.Session()
print('node1, node2 : ', sess.run([node1, node2]))
print('sess.run(node3) : ', sess.run(node3) )

#. node1, node2 :  [3.0, 4.0]
#. sess.run(node3) :  7.0
