import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150,)
print(y)

# 원핫인코딩: One-Hot-Encoding (150, ) -> (150, 3)  // 라벨의 개수만큼 shope가 늘어난다. 
# 0 -> [1, 0, 0]
# 1 -> [0, 1, 0]
# 2 -> [0, 0, 1]

# [0, 1, 2, 1, 2] -> 기존 y데이터 아래로 바꿈
# [[1, 0 ,0]
# [0, 1, 0]
# [0. 0. 1]
# [0, 1, 0]]  (4, ) -> (4, 3) 라벨의 종류만큼 위에는 3개 

from tensorflow.keras.utils import to_categorical #얘는 0부터 시작함 만약 3부터 시작하면 0123을 채워 shape 크기를 늘림 : wind2 확인

y = to_categorical(y) #원핫인풋 // 수치에 대한 라벨링을 하는 작업 남자는 여자의 두배의 가치가 잇는게 아니다 그냥 단지 라벨을 통해 1 2로 표현할 뿐


print(y[:5])
print(y.shape) # (150, 3)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=66) 



from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer


# x = x/711.
# x = (x-np.min(x))/(np.max(x)-np.min(x))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, LSTM,Dropout, GlobalAveragePooling2D
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) #테스트 데이터는 트레인 데이터에 관여하면안된다.
x_test = scaler.transform(x_test)

x_train = x_train.reshape(105,4,1)
x_test = x_test.reshape(45,4,1)
#이전까지는 선형회귀 분류 모델 지금부터는 2진 분류 모델


# print(np.min(x), np.max(x))  # 0.0     711.0
# print(np.min(y), np.max(y))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool2D, Dropout

# 모델구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(4,1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3,  activation='softmax'))
model.summary()

# model = Sequential()
# model.add(Conv2D(filters=128, kernel_size=(1,1), padding='valid',
#         activation='relu' ,input_shape=(4,1,1)))

# model.add(Dropout(0.2))
# model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
# model.add(MaxPool2D())      

# model.add(Conv2D(128, (2,2), padding='valid', activation='relu'))
# model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
# model.add(MaxPool2D())  

# model.add(Conv2D(64, (2,2), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2,2), padding='same',  activation='relu'))
# model.add(MaxPool2D())  #! (3,3) 이과정 알아야해 560개의 노드? 
# model.add(GlobalAveragePooling2D())
# model.add(Dense(3, activation='softmax'))

#  model.add(Dense(128,activation='relu', input_shape=(4,)))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(3, activation='softmax')) # 다중 분류의 라벨의 수가 3개 



# 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 2진분류를 위한
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=3)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', 
                        filepath='./_save/ModelCheckPoint/keras48_4_iris_MCP.hdf5')
hist = model.fit(x_train,y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es, cp])
# print(hist) # <tensorflow.python.keras.callbacks.History object at 0x000001BF025F73D0>
model.save('./_save/ModelCheckPoint/keras48_4_iris_model_save.h5')

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
print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)


#r2 는 회귀 모델이서쓰고 acuracy는 분류에서 사용함 남자 여자 이 둘중하나 무조건 이어야함

# print('예측값 : ', y_predict)

# r2 = r2_score(y_test, y_predict) # 예측한 값과 원래값 을 비교해 오차를 확인한다.    
# print('r2 스코어 : ', r2) 


# import matplotlib.pyplot as plt

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])

# #한글로 쓰면 깨짐 과제
# plt.title('loss, val_loss')
# plt.xlabel('epochs')
# plt.ylabel('loss, val_loss')
# #범례 생성
# plt.legend('train loss', 'val_loss')       
# plt.show()


'''
cnn
loss :  0.19331702589988708
acuracy :  0.9555555582046509
'''


'''
lmst
loss :  0.16695906221866608
acuracy :  0.9333333373069763

'''