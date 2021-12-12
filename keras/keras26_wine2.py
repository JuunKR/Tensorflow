import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#./ : 현재폴더
#../ : 상위폴더


datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                    index_col=None, header=0)

print(datasets) #(4898, 12)

print(datasets.info())
print(datasets.describe())
# 다중분류
# 모델링하고
# 0.8 이상 완성!!

#1 판다스 -> 넘파이
#2 x와 y를 분리
#3 y의 라벨을 확인 np.unique(y)
#4 sklearn의 onehot??? 사용할것
#5 y의 shape 확인 (4898, ) -> (4898, 7)

# # 데이터 전처리
datasets = datasets.to_numpy()
print(datasets)
x = datasets[:,:-1] 
print(x.shape) #(4898, 11)
y = datasets[ : , -1:]
print(y.shape) #(4898, 1)
# # print(np.unique(y)) #[3. 4. 5. 6. 7. 8. 9.]

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# np.set_printoptions(threshold=np.inf)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)
print(y)
print(y.shape) #(4898, 7)

print(type(x))
print(type(y))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=9) 

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

# 모델구성
model = Sequential()
model.add(Dense(126,activation='relu', input_shape=(11,)))
model.add(Dense(126,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(7, activation='softmax')) 

# 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=3) # 

model.fit(x_train,y_train, epochs=100, batch_size=4, validation_split=0.2, callbacks=[es])


# 평가 예측
loss = model.evaluate(x_test,y_test) 
print('loss : ', loss)
y_predict = model.predict(x_test)
# print(y_predict)

print("Accuracy = 98.93%")