from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

# 데이터 전처리
datasets = load_boston()
x = datasets.data
y = datasets.target

# print(x.shape) # (506, 13)
# print(y.shape) # (506,)

# print(datasets.feature_names)
# print(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=66) 

# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

# print(y_train.shape)
# print(y_test.shape)


# print(x_train.shape)
# print(x_test.shape)
# (354, 13)
# (152, 13)

x_train = x_train.reshape(354, 13, 1, 1)
x_test = x_test.reshape(152, 13, 1, 1)




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

# 모델
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(1,1),
        activation='relu' ,input_shape=(13,1,1)))

model.add(Dropout(0.2))
model.add(Conv2D(64, (1,1), activation='relu'))
   

model.add(Conv2D(128, (1,1), activation='relu'))
model.add(Conv2D(128, (1,1),  activation='relu'))


model.add(Conv2D(64, (1,1), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (1,1), activation='relu'))


model.add(Flatten()) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))


# 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=3) # 
model.fit(x_train,y_train, epochs=100, batch_size=2, callbacks=[es])


# 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
# print('예측값 : ', y_predict)
r2 = r2_score(y_test, y_predict) 
print('r2 스코어 : ', r2) 

# # MaxAbsScaler()
# # r2 스코어 :  0.8017120236357702

# # RobustScaler()
# # r2 스코어 :  0.7905847881893635

# # QuantileTransformer()
# # r2 스코어 :  0.7422795542093643

# # PowerTransformer()
# # r2 스코어 :  0.8108507778033534

'''
cnn
loss :  9.009257316589355
r2 스코어 :  0.8909516642864456
'''
