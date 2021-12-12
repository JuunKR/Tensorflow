import tensorflow as tf

sess = tf.Session()

'''
x = tf.Variable([2], dtype=tf.float32, name='test')

sess.run(x)
# tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value test
'''

#@ 텐서플로우의 변수는 반드시 초기화를 해주어야 한다.

x = tf.Variable([1], dtype=tf.float32) # name='test') # 이름 지정

init = tf.global_variables_initializer()

sess.run(init)
print("x : ", sess.run(x))
