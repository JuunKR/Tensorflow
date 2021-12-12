#1. R2를 음수가 아닌 0.5 이하로 만들어라
#2. 데이터 건들지 마
#3. 레이어는 인풋 아웃풋 포함 6개 이상
#4. batch_size = 1
#5. epochs는 100 이상
#6. 히든레이어의 노드는 10개 이상 1000개 이하
#7. train 70%

from tensorflow.keras.models import Sequential, Model # 모델에는 두가지가 있음 순차형/함수형 
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt



#1. 데이터 전처리(특기)d
x = np.array(range(100))   
y = np.array(range(1, 101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=66) 


input1 = Input(shape=(1,))
dense1 = Dense(1000)(input1)
dense2 = Dense(1000)(dense1)
dense3 = Dense(1000)(dense2)
dense4 = Dense(1000)(dense3)
dense5 = Dense(1000)(dense4)
dense6 = Dense(1000)(dense5)
dense7 = Dense(1000)(dense6)
dense8 = Dense(1000)(dense7)
dense9 = Dense(1000)(dense8)
dense10 = Dense(1000)(dense9)
dense11 = Dense(1000)(dense10)
dense12 = Dense(1000)(dense11)
dense13 = Dense(1000)(dense12)
output1 = Dense(1)(dense13)

model = Model(inputs=input1, outputs=output1)

#2. 모델 구성
# model = Sequential()
# model.add(Dense(1000, input_dim=1))
# model.add(Dense(1000))
# model.add(Dense(1000))
# model.add(Dense(1000))
# model.add(Dense(1000))
# model.add(Dense(1000))
# model.add(Dense(1000))
# model.add(Dense(1000))
# model.add(Dense(1000))
# model.add(Dense(1000))
# model.add(Dense(1000))
# model.add(Dense(1000))
# model.add(Dense(1000))
# model.add(Dense(1)) 

#3. 컴파일 훈련
model.compile(loss='kld', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) 
print('100의 예측값 : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2 스코어 : ', r2) 


#함수형으로 
#결과 출력