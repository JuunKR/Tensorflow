import tensorflow as tf
#@ Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
#@ 더빠른 연산을 진행할 수 있는 방법이 있어
#@ 무시
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

#@ 상수 텐서를 만듦
#@ tf.constant(value, dtype=None, shape=None, name='Const')


hello = tf.constant('hello world')

print(hello)
#. Tensor("Const:0", shape=(), dtype=string)

#@ 텐서의 자료형의 구조 ; 텐서도 자료형이다.
#@ shape = 0 즉 스칼라 값으로 인식하고 있다.

# sess = tf.Session()
sess = tf.compat.v1.Session()
#@ 둘다 동일하나 아래는 warning이 안나옴

print(sess.run(hello))
#. b'hello world'

#@ sess.run() 안에 변수를 넣어야함
#@ 변수를 그냥 출력하게 되면 자료구조형이 나온다. ; 반드시 sess에 넣어야함.