import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import add


node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
#@ 선언 후 에 값을 전달하는 자료형
#@ placeholder(dtype, shape=None, name=None)

adder_node = a + b

print(sess.run(adder_node, feed_dict={a: 3, b :4.5}))
#. 7.5
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [3, 4]}))
#. [4. 7.]

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a: 4, b: 2}))
#. 18.0


