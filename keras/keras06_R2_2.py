from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


x = [1, 2, 3, 4, 5]
y = [1, 2, 4, 3, 5]

# 완성한뒤, 출력결과스샷

model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(4, input_dim=1))

model.add(Dense(5, input_dim=1))
model.add(Dense(10, input_dim=1))
model.add(Dense(2, input_dim=1))
model.add(Dense(5, input_dim=1))
model.add(Dense(1, input_dim=1))


model.compile(loss='mse', optimizer='adam')

model.fit(x,y, epochs=7000, batch_size=5)

loss = model.evaluate(x,y)
print('loss : ', loss)

y_predict = model.predict(x)
print('6의 예측값 : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict) # 예측한 값과 원래값 을 비교해 오차를 확인한다.
print('r2 스코어 : ', r2) 



# model = Sequential()
# model.add(Dense(1, input_dim=1))

# model.compile(loss='mse', optimizer='adam')

# model.fit(x,y, epoches='', optimizer='adam')

# loss = model.evaluate(x,y)



# 과제 2d
# R2를 0.9 올려라!!!


