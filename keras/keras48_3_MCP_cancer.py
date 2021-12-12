import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 데이터 전처리
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

print(y[:20])
print(np.unique(y)) # [0 1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=66) 


#이전까지는 선형회귀 분류 모델 지금부터는 2진 분류 모델


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


print(x_train.shape) 
print(x_test.shape)
'''
(398, 30)
(171, 30)
'''
x_train = x_train.reshape(398, 30, 1)
x_test = x_test.reshape(171, 30, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool2D, LSTM,Dropout
# 모델구성
# model = Sequential()
# model.add(Dense(128,activation='relu', input_dim=30))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(1, activation='sigmoid')) # 0과 1사이의 값 출력
model = Sequential()

model.add(Conv1D(64, 2, input_shape=(30,1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
# 컴파일 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 2진분류를 위한
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=3)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', 
                        filepath='./_save/ModelCheckPoint/keras48_3_cancer_MCP.hdf5')

hist = model.fit(x_train,y_train, epochs=100, batch_size=20, validation_split=0.2, callbacks=[es, cp])
# print(hist) # <tensorflow.python.keras.callbacks.History object at 0x000001BF025F73D0>

model.save('./_save/ModelCheckPoint/keras48_3_cancer_model_save.h5')

# print(hist.history.keys()) # dict_keys(['loss', 'val_loss'])
# print("====================== loss =============================")
# print(hist.history['loss'])
# print("======================= val_loss ============================")
# print(hist.history['val_loss'])


print("======================= 평가 예측 ============================")
loss = model.evaluate(x_test,y_test) # evaluate도 배치 사이즈가 있음 디폴트 32
print('loss : ', loss[0])
print('acuracy : ', loss[1])


print('==================== 예 측 ==================================')
print(y_test[-5:-1])
y_predict = model.predict(x_test[-5:-1])
print(y_predict)

#r2 는 회귀 모델이서쓰고 acuracy는 분류에서 사용함 남자 여자 이 둘중하나 무조건 이어야함

# print('예측값 : ', y_predict)

# r2 = r2_score(y_test, y_predict) # 예측한 값과 원래값 을 비교해 오차를 확인한다.    
# print('r2 스코어 : ', r2) 

'''
cnn
loss :  0.12162502110004425
acuracy :  0.9590643048286438
'''

'''
lstm
loss :  0.20491619408130646
acuracy :  0.9473684430122375
'''

# import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
# font_path = "C:/Windows/Fonts/NGULIM.TTF"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])

# #한글로 쓰면 깨짐 과제
# plt.title('로스, 발로스')
# plt.xlabel('epochs')
# plt.ylabel('loss, val_loss')
# #범례 생성
# plt.legend('train loss', 'val_loss')       
# plt.show()