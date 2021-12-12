'''
TensorFlow 1.14

연산은 graph로 표현합니다.(graph는 점과 선, 업계 용어로는 노드와 엣지로 이뤄진 수학적인 구조를 의미) = graph는 노드와 엣지로 이루어진 수학적인 구조를 말한다. 
graph는 Session내에서 실행
데이터는 tensor로 표현
변수(Variable)는 (여러 graph들이 작동할 때도) 그 상태를 유지한다.
작업(operation 혹은 op)에서 데이터를 입출력 할 때 feed와 fetch를 사용할 수 있다.

개요
Tensorflow는 graph로 연산(computation)을 나타내는 프로그래밍 시스템이다.
graph에 있는 노드는 작업(operaiton = op)라고 부른다.
작업은 0개 혹은 그 이상의 Tensor를 가질 수 있고 연산도 수행하며 0개 혹은 그 이상의 Tensor를 만들어 낸다.
Tensorflow에서 Tensor는 정형화된 다차원 배열(a typed multi-dimensional array)이다.
ex) 이미지는 부동소수점 수(floating point number)를 이용한 4차원 배열([batch, height, width, channels)])로 나타낼 수 있다.

Tensorflowd에서 graph는 연산을 표현해 놓은 것이라서 연산을 하려면 graph가 Session 상에 실행이 되어야 한다. 
Session은 graph의 작업을 CPU나 GPU같은 Device에 배정하고 실행을 위한 매서드를 제공한다. 
이런 메서들은 작업을 실행해서 tensor를 만들어 낸다. 
tensor는 파이썬에서 numpy array 형식으로 나오고 C와 C++에서는 Tensorflow:Tensor 형식으로 나온다.

Graph
Tensorflow에서는 graph를 조립하는 구성단계(constan phase)와 Session을 이용해서 graph의 op를 실행시키는 실행단계(execution phase)로 구성된다.
ex) 뉴럴 네트워크를 표현하고 학습시키기 위해 구성 단계에는 graph를 만들고 실행단계에는 graph의 훈련용 작업들(set of training ops)를 반복해서 실행한다.

TensorFlow는 C, C++, 파이썬을 이용할 수 있다. 지금은 graph를 만들기 위해 파이썬 라이브러리를 사용하는 것이 훨씬 쉽다. C, C++에서는 제공하지 않는 많은 헬퍼 함수들을 쓸 수 있기 때문이다.
session 라이브러리들은 3개 언어에 동일한 기능을 제공합니다.

Tensors
TensorFlow 프로그램은 모든 데이터를 tensor 데이터 구조로 나타낸다. 
연산 graph에 있는 작업들(op) 간에는 tensor만 주고받을 수 있다. 
TensorFlow의 tensor를 n 차원의 배열이나 리스트라고 봐도 된다. 
tensor는 정적인 타입(static type), 차원(예를 들어 1차원, 2차원하는 차원), 형태(shape, 예를 들어 2차원이면 m x n) 값을 가진다. 
'''
import tensorflow as tf


#@ graph
# 1x2 행렬을 만드는 constant op
# 이 op는 default graph에 노드로 들어간다.

# 생성함수에서 나온 값은 constant op의 결과값.
matrix1 = tf.constant([[3., 3.]])

# 2x1 행렬을 만드는 constant op
matrix2 = tf.constant([[2.],[2.]])

# 'matrix1'과 'matrix2를 입력값으로 하는 Matmul op(행렬곱)
# 이 op의 결과값인 'product'는 행렬곱의 결과를 의미한다.
product = tf.matmul(matrix1, matrix2)

# default graph에는 이제 3개의 노드가 있다. 
# 2개는 상수(constant) 작업(op)이고 하나는 행렬곱(matmul) 작업(op)
# 행렬을 곱해서 결과값을 얻으려면 Session에다 graph를 실행해야 합니다.

# default graph를 실행
sess = tf.Session()

#@ sesscion cloese
# 행렬곱 작업(op)을 실행하기 위해 session의 'run()' 메서드를 호출해서 행렬곱 
# 작업의 결과값인 'product' 값을 넘긴다.

# 작업에 필요한 모든 입력값들은 자동적으로 session에서 실행되며 보통은 병렬로 처리
#
# 'run(product)'가 호출되면 op 3개가 실행됩니다. 2개는 상수고 1개는 행렬곱

# 작업의 결과물은 numpy `ndarray` 오브젝트인 result' 값으로 나온다.
result = sess.run(product)
print(result)
#. [[ 12.]]

# 실행을 마치면 Session을 닫읍시다.
sess.close()

#@ sesscion cloese ; auto
# 연산에 쓰인 시스템 자원을 돌려보내려면 session을 닫아야 한다
# 시스템 자원을 더 쉽게 관리하려면 with 구문을 쓰면 된다. 
# 각 Session에 컨텍스트 매니저가 있어서 'with' 구문 블락의 끝에서 자동으로 'close()'가 호출된다.

with tf.Session() as sess:
    result = sess.run([product])
    print(result)

#@ Device
# TensorFlow의 구현 코드(TensorFlow implementation)를 통해 graph에 정의된 내용이 실행가능한 작업들(operation)로 변환되고 CPU나 GPU같이 이용가능한 연산 자원들에 뿌린다.
# 코드로 어느 CPU 혹은 GPU를 사용할 지 명시적으로 지정할 필요는 없음
# 작업을 가능한 한 많이 처리하기 위해 TensorFlow는 (컴퓨터가 GPU를 가지고 있다면) 첫 번째 GPU를 이용한다.
# 만약 컴퓨터에 복수의 GPU가 있어서 이를 사용하려면, op을 어느 하드웨어에 할당할 지 명시적으로 밝혀야 한다. 
# 작업에 사용할 CPU 혹은 GPU를 지정하려면 with...Device 구문을 사용하면 된다.

with tf.Session() as sess:
    with tf.device("/gpu:1"):
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.],[2.]])
        product = tf.matmul(matrix1, matrix2)

# 이용할 CPU 혹은 GPU는 문자열로 지정할 수 있다.

# "/cpu:0": 컴퓨터의 CPU.
# "/gpu:0": 컴퓨터의 1번째 GPU.
# "/gpu:1": 컴퓨터의 2번쨰 GPU.

#@ Variables
# 그래프를 실행하더라도 변수(variable)의 상태는 유지된다. 

# 값이 0인 스칼라로 초기화된 변수를 만든다
state = tf.Variable(0, name="counter")

# 'state'에 1을 더하는 작업(op)을 만든다
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 그래프를 한 번 작동시킨 후에는 'init' 작업(op)을 실행해서 변수를 초기화해야 한다
#  먼저 'init' 작업(op)을 추가
init_op = tf.global_variables_initializer()

# graph와 작업(op)들을 실행시킵니다.
with tf.Session() as sess:
    # 'init' 작업(op)을 실행합니다.
    sess.run(init_op)
    # 'state'의 시작값을 출력합니다.
    print(sess.run(state))
    # 'state'값을 업데이트하고 출력하는 작업(op)을 실행합니다.
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
        # output:
        #. 0
        #. 1
        #. 2
        #. 3

# 이 코드에서 assign() 작업은 add() 작업처럼 graph의 한 부분이다. 
# 그래서 run()이 graph를 실행시킬 때까지 실제로 작동하지 않는다..
# 우리는 보통 통계 모델의 파라미터를 변수로 표현한다.
# 예를 들어 뉴럴 네트워크의 비중값을 변수인 tensor로 표현할 수 있다.
# 학습을 진행할 때 훈련용 graph를 반복해서 실행시키고 이 tensor 값을 업데이트 한다.

#@ Fetches
# 작업의 결과를 가져오기 위해 Session 오브젝트에서 run()을 호출해서 graph를 실행하고 tensor로 결과값을 끌어낸다.
# 앞의 예제에서는 'state' 하나의 노드만 가져왔지만 복수의 tensor를 받아올 수도 있다.

input1 = tf.constant([3.0])
input2 = tf.constant([2.0])
input3 = tf.constant([5.0])
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)
    # output:
    #. [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]
# 여러 tensor들의 값을 계산해내기 위해 수행되는 작업(op)들은 각 tensor 별로 각각 수행 되는 것이 아니라 전체적으로 한 번만 수행된다.``

#@ Feeds
# 위의 예제에서 살펴본 graph에서 tensor들은 상수(Constant) 와 변수(Variable)로 저장된다.
# TensorFlow에서는 graph의 연산에게 직접 tensor 값을 줄 수 있는 'feed 메커니즘'도 제공한다.
# feed 값은 일시적으로 연산의 출력값을 입력한 tensor 값으로 대체한다.
# feed 데이터는 run()으로 전달되어서 run()의 변수로만 사용된다.
# 가장 일반적인 사용방법은 tf.placeholder()를 사용해서 특정 작업(op)을 "feed" 작업으로 지정해 주는 것이다.

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = input1 * input2

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
    # output:
    #. [array([ 14.], dtype=float32)]
# 만약 feed 를 제대로 제공하지 않으면 placeholder() 연산은 에러를 출력할 것이다. 
