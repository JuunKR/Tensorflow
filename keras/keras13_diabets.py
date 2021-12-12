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
      - s5      ltg, lamotriginedat
'''

print(y[:30])
print(np.min(y), np.max(y)) #25.0 346.0d



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.8, shuffle=True, random_state=66) 


model = Sequential()
model.add(Dense(1000, input_dim=10))
model.add(Dense(2))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

model.fit(x_train,y_train, epochs=200, batch_size=1, validation_split=0.4, verbose=3)

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print('예측값 : ', y_predict)

r2 = r2_score(y_test, y_predict) # 예측한 값과 원래값 을 비교해 오차를 확인한다.
print('r2 스코어 : ', r2) 

# 0.502877251712494

# 0.62 까지 올리기 