# 실습
# mnist 데이터를 pca 통해 cnn으로 구성
# (28,28) -> 784 -> 차원축소 (400) -> (20, 20) -> CNN 모델 구성

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, MaxPooling2D
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

#@ 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

pca = PCA(n_components=400)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 20, 20, 1)
x_test = x_test.reshape(10000, 20, 20, 1)

#@ 모델
model = Sequential()
model.add(Conv2D(filters=150, activation='relu', kernel_size=(1), padding='same', input_shape=(20, 20, 1)))
model.add(Conv2D(150, (1), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(75, (1), activation='relu', padding='same'))
model.add(Conv2D(50, (1), activation='relu', padding='same'))          # (N, 9, 9, 20)
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

#@ 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
model.fit(x_train, y_train, batch_size=256, epochs=100, validation_split=0.1, callbacks=[es])

#@ 평가, 예측
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict)
acc = accuracy_score(y_predict, y_test)
f1 = f1_score(y_predict, y_test)

print('acc : ', acc)
print('f1 : ', f1)

