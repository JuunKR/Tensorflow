from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import numpy as np


datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape) # (506, 13)
print(y.shape) # (506,)

print(datasets.feature_names)
print(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=66) 

# print(np.min(x), np.max(x))  # 0.0     711.0
# print(np.min(y), np.max(y))




#데이터 전처리
# x = x/711.
# x = (x-np.min(x))/(np.max(x)-np.min(x))
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) #테스트 데이터는 트레인 데이터에 관여하면안된다.
x_test = scaler.transform(x_test)


# print(x_scale[:10])
# print(np.min(x_scale), np.max(x_scale))



'''
 - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

'''
# 모델구성
model = Sequential()
model.add(Dense(128,activation='relu', input_dim=13))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))


# 컴파일 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1) # patience는 epoches 기준으로 설정, mode #loss 에서 val_los로 바꿈 // 너무 빨리 끝나면 patience를 조절하자
hist = model.fit(x_train,y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])
# print(hist) # <tensorflow.python.keras.callbacks.History object at 0x000001BF025F73D0>

print(hist.history.keys()) # dict_keys(['loss', 'val_loss'])
print("====================== loss =============================")
print(hist.history['loss'])
print("======================= val_loss ============================")
print(hist.history['val_loss'])


print("======================= 평가 예측 ============================")
loss = model.evaluate(x_test,y_test) # evaluate도 배치 사이즈가 있음 디폴트 32
print('loss : ', loss)

y_predict = model.predict(x_test)
# print('예측값 : ', y_predict)

r2 = r2_score(y_test, y_predict) # 예측한 값과 원래값 을 비교해 오차를 확인한다.    
print('r2 스코어 : ', r2) 


# r2 스코어 :  0.8556611356006649

# MinMaxScaler 후
#  0.8944905752295937

#r2 스코어 :  0.9157576661544031

# standard scaler 후
# 0.9197386964313358
# 완료하시오!!!


# earlystopping 후
# r2 스코어 :  0.8997055420068706

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

#한글로 쓰면 깨짐 과제
plt.title('loss, val_loss')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
#범례 생성
plt.legend('train loss', 'val_loss')       
plt.show()