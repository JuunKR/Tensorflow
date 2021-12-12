from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

# 데이터 전처리
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=66) 

scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(354, 13, 1)
x_test = x_test.reshape(152, 13, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool2D, Dropout, LSTM

# 모델
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(13,1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()

from tensorflow.keras.models import load_model


# 컴파일 훈련



model = load_model('./_save/ModelCheckPoint/keras48_1_boston_MCP.hdf5') 

# model = load_model('./_save/ModelCheckPoint/keras48_1_boston_model_save.h5') 

model.compile(loss='mse', optimizer='adam')


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=3)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', 
                        filepath='./_save/ModelCheckPoint/keras48_1_boston_MCP.hdf5')
model.fit(x_train,y_train, epochs=100, batch_size=2,validation_split=0.25, callbacks=[es, cp])

model.save('./_save/ModelCheckPoint/keras48_1_boston_model_save.h5')



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

'''
lstm
0.8057916873147833
'''