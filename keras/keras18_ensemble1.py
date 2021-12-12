import numpy as np
from sklearn.metrics import r2_score

# 데이터 전처리
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100,200)])

x1 = np.transpose(x1)   #(100, 3) 
x2 = np.transpose(x2)   #(100, 3)

y = np.array(range(1001, 1101))    #(100,)

print(x1.shape, x2.shape, y.shape)
from sklearn.model_selection import train_test_split  #train_size를 넣지 않으면 돌아갈까? 디폴트 확인해 보기 // 중위값과 평균값 알아보기

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, random_state=66, train_size=0.7) 

print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y_train.shape, y_test.shape)

#모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu',name='dense2')(dense1)
dense3 = Dense(5, activation='relu', name='dense3')(dense2)
output1 = Dense(11, name='output1')(dense3) # 여긴 아웃풋이 아님 merge하기 때문에


#2-2 모델2
input2 = Input(shape=(3,))
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(12, name='output2')(dense14) # 여긴 아웃풋이 아님 merge하기 때문에

from tensorflow.keras.layers import concatenate, Concatenate

merge1 = Concatenate()([output1, output2]) # 가중치 두개가 이어질뿐 연산되지는 않음 // 레이어랑 동일   #요거도 수정!
merge2 = Dense(10,activation='relu')(merge1)
merge3 = Dense(5,activation='relu')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)



model.summary()

#3.컴파일, 훈련 
model.compile(loss = 'mse', optimizer="adam", metrics=["mae"])
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=8)


#4. 평가, 예측
result = model.evaluate([x1_test, x2_test], y_test)
print('loss : ', result)

# y_predict = model.predict([x1_test, x2_test])
# print('예측값 : ', y_predict)

# r2 = r2_score(y_test, y_predict) # 예측한 값과 원래값 을 비교해 오차를 확인한다.d
# print('r2 스코어 : ', r2) 

