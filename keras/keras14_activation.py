import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)             # (442, 10) (442,)

print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

print(datasets.DESCR)
'''
     - age     age in years
      - sex
      - bmi     body mass index
      - bp      average blood pressure
      - s1      tc, T-Cells (a type of white blood cells)
      - s2      ldl, low-density lipoproteins
      - s3      hdl, high-density lipoproteins
      - s4      tch, thyroid stimulating hormone
      - s5      ltg, lamotrigine
      - s6      glu, blood sugar level
'''

print(y[:30])
print(np.min(y), np.max(y)) #25.0 346.0

from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=8) 


model = Sequential()
model.add(Dense(1, input_dim=10, activation='relu')) # 활성화 함수 , 레이어에서 다음 레이어보내는 값을 고정 // 통상적으로 사용하면 값이 좋아짐
model.add(Dense(79, activation='relu'))
model.add(Dense(20, activation='relu')) # 2의 배수로 하는게 좋아

model.add(Dense(1)) # 마지막 레이어에 고정되어있d는 엑티베이션이 있기 때문에 지금은 설정 ㄴㄴ


model.compile(loss='mse', optimizer='adam')

model.fit(x_train,y_train, epochs=700, batch_size=8, validation_split=0.2, verbose=1)

loss = model.evaluate(x_test,y_test)
# print('loss : ', loss)

y_predict = model.predict(x_test)
# print('예측값 : ', y_predict)

r2 = r2_score(y_test, y_predict) # 예측한 값과 원래값 을 비교해 오차를 확인한다.

print('r2 스코어 : ', r2) 




# 0.62 까지 올리기 
'''
random_state = 9
31/31 [==============================] - 0s 3ms/step - loss: 2858.3978 - val_loss: 4056.0303
5/5 [==============================] - 0s 2ms/step - loss: 2128.0845
r2 스코어 :  0.6149150854006515
PS D:\study> 
'''

'''
randome_state = 66
31/31 [==============================] - 0s 3ms/step - loss: 2699.5035 - val_loss: 2962.0149
5/5 [==============================] - 0s 997us/step - loss: 3152.5581
r2 스코어 :  0.49400147871808797
'''