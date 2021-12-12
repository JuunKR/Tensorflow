import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,7,9,11,2,3,4,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
# model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

#@ 0.1 ~ 0.001
optimizer = Adam(lr=0.1) # default 0.001
# loss :  10.047402381896973 결과물 :  [[7.166324]]
# loss :  10.58189582824707 결과물 :  [[8.854786]]
# loss :  10.021345138549805 결과물 :  [[8.155487]]

# optimizer = Adagrad(lr=0.001) # defailt 0.01
# loss :  9.981801986694336 결과물 :  [[7.8171043]
# loss :  10.353050231933594 결과물 :  [[8.616547]]
# loss :  11.709328651428223 결과물 :  [[9.317493]]

# optimizer = Adamax(lr=0.001) # default 0.002
# loss :  9.98204517364502 결과물 :  [[7.814204]]
# loss :  10.170432090759277 결과물 :  [[8.546999]]
# loss :  10.261199951171875 결과물 :  [[7.760238]]

# optimizer = Adadelta(lr=0.001) # default 1.0
# loss :  11.346941947937012 결과물 :  [[8.378799]]
# loss :  12.073348999023438 결과물 :  [[9.492452]]
# loss :  22.712060928344727 결과물 :  [[3.828607]]

# optimizer = RMSprop(lr=0.001) #default 0.001
# loss :  79436.5625 결과물 :  [[60.432804]]
# loss :  11.615727424621582 결과물 :  [[5.481827]]
# loss :  10.177339553833008 결과물 :  [[7.741863]]

# optimizer = SGD(lr=0.01) # default 0.01 안나옴

# optimizer = Nadam(lr=0.001) # default 0.002
# loss :  10.061327934265137 결과물 :  [[8.2508135]]
# loss :  23.180208206176758 결과물 :  [[4.0800734]]
# loss :  10.085018157958984 결과물 :  [[7.5248413]]

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x,y, epochs=100, batch_size=1)

#4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])

print('loss : ', loss, '결과물 : ', y_pred)
# loss :  0.00037413244717754424 결과물 :  [[10.969938]]