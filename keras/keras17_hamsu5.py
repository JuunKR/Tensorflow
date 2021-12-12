import numpy as np

#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101),
                range(100), range(401, 501)])

x = np.transpose(x) # (100, 5)
print(x.shape)

y = np.array([range(711, 811), range(101, 201)])
y = np.transpose(y)
print(y.shape) # (100, 2)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input


input1 = Input(shape=(5,))
xx = Dense(3)(input1)
xx = Dense(4)(xx)
xx = Dense(10)(xx)
output1 = Dense(2)(xx)
# x를 사용하면 오버라이팅 // 시퀀셜한 구조에서는 사용하나 concatnate를 할 때는 불가능

model = Model(inputs=input1, outputs=output1)

# model = Sequential()
# model.add(Dense(3, input_shape=(5,)))d
# model.add(Dense(4))
# model.add(Dense(10))
# model.add(Dense(2))

model.summary()





#3. 컴파일, 훈련


#4. 평가 예측