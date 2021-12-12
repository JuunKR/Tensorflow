import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
Eager execution은 더 파이썬적인(Pythonic) 방식으로 전체 그래프를 생성하지 않고 함수를 바로 실행하는 명령형 프로그래밍 환경입니다. 동적으로 계산 그래프를 작동하는 방식입니다. 
Session 객체를 생성하는 것과 전체 모델 그래프 빌드를 요구하는 것이 이전의 단점이어서, eager execution이 소개되었습니다.
Eager execution의 장점은 전체 모델이 빌드되어있을 필요가 없다는 것입니다. 따라서 즉시 operation이 evaluate되는 것 대신에, 모델을 빌딩하는 것을 시작하기 쉽습니다. 그래서 디버깅하기도 좋습니다.
하지만 단점이 역시 존재합니다! multi GPU를 위한 분산 strategy를 위해서는 disable해주어야 합니다.

'''

#@ 2.4.1 버전에서 사용해보기
# AttributeError: module 'tensorflow' has no attribute 'Session'

print(tf.__version__)

print(tf.executing_eagerly())
#. True

tf.compat.v1.disable_eager_execution()
#@ 2.4.1에서 session이 필요할 때 ; 즉시실행모드
'''
2.x에서는 1.x와 다르게 session을 정의하고 run을 수행하는 과정이 생략되고 바로 실행되는 형태로 변경되었기 때문이다.

hello = tf.constant('Hello, TensorFlow!') 
print(hello) 

a = tf.constant(10) 
b = tf.constant(32) 
c = tf.add(a, b) 
# a + b 로도 쓸 수 있음 
print(c) 

1.x의 경우 
sess = tf.Session() 
print(sess.run(hello)) 
print(sess.run([a, b, c])) 
sess.close() 

2.x의 경우 Session을 정의해주고 run 해주는 과정이 생략됨 
tf.print(hello) 
tf.print([a, b, c])
'''

print(tf.executing_eagerly())
#. False

hello = tf.constant('hello world')
print(hello)
#. Tensor("Const:0", shape=(), dtype=string)

sess = tf.compat.v1.Session()
print(sess.run(hello))
#. b'hello world'



