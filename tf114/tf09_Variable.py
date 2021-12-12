import tensorflow as tf


#@ session을 여는 3가지 방식
tf.compat.v1.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]))
print(W)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(W)

print("aaa : ", aaa)

sess.close()
#@ session은 끝나고 닫아줘야함

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = W.eval() #@ 변수 .eval
print('bbb : ', bbb)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)
print('ccc : ', ccc)
sess.close()

