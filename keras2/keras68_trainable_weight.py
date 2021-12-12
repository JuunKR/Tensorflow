import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

print(model.weights)
print('=======================================================================')
print(model.trainable_weights)
print('=======================================================================')
print(len(model.weights))  # 3(w + b) 층별로 w와 b 가 하나씩 있음 
print(len(model.trainable_weights))


'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 3)                 6
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 8
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 3
=================================================================
Total params: 17
Trainable params: 17 ---------------> 일반 모델의 weight와 동일한 값 / 일반 weight 값은 랜덤 -> 훈련을 하면서 값들이 갱신이 됨.
Non-trainable params: 0
'''


'''
[<tf.Variable 'dense/kernel:0' shape=(1, 3)  -----> dense 인풋, 아웃풋 

 dtype=float32,      ---- > kernel은 통상적으로 weight를 이야기함

numpy=array([[ 0.56869924,  0.9017974 , -0.06263876]], dtype=float32)>, <tf.Variable 

'dense/bias:0' shape=(3,) ------> 3번 연산하기 때문에 위에서 아웃풋이 3/ 디폴트 0

dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, 


<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, 
numpy=array([[ 0.00118077,  0.6483333 ],
            [ 1.0013511 , -0.9749129 ],
            [-0.48227435,  0.05811262]], dtype=float32)>, <tf.Variable 

'dense_1/bias:0' shape=(2,) dtype=float32, 

numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=

array([[-1.0980737],
       [-1.0478472]], dtype=float32)>, <tf.Variable 

'dense_2/bias:0' shape=(1,) dtype=float32,
numpy=array([0.], dtype=float32)>]

'''

