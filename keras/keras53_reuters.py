from numpy.lib.arraypad import pad
from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)   # 단어사전의 개수

print(x_train[0], type(x_train[0]))
print(x_train[1], type(x_train[1]))


'''
0
[1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 102, 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 11, 15, 7, 48, 9, 4579, 1005, 504, 6, 
258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 1245, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12] 

<class 'list'> 리스트는 크기제한이 없다. 0번쨰와 1번쨰의 크기가 다를 수 있음

1
[1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 102, 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 11, 15, 7, 48, 9, 4579, 1005, 504, 6, 
258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 1245, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12] <class 'list'>
[1, 3267, 699, 3434, 2295, 56, 2, 7511, 9, 56, 3906, 1073, 81, 5, 1198, 57, 366, 737, 132, 20, 4093, 7, 2, 49, 2295, 2, 1037, 3267, 699, 3434, 8, 7, 10, 241, 16, 855, 129, 231, 783, 5, 4, 587, 2295, 2, 2, 775, 7, 48, 34, 191, 44, 35, 1795, 505, 
17, 12]
<class 'list'>
'''

print(y_train[0]) # 3
print(len(x_train[0]), len(x_train[1])) #87, 56

print(x_train.shape, x_test.shape) # (8982,) (2246,)
print(y_train.shape, y_test.shape) # (8982,) (2246,)

print(type(x_train)) # <class 'numpy.ndarray'>

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)) # 2376
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train)) # 145.5

plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

#@ 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre') #(8982, 100)
x_test = pad_sequences(x_test, maxlen=100, padding='pre')  # (2246, 100)

print(x_train.shape, x_test.shape)
print(type(x_train), type(x_train[0])) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(x_train[1])


#@ y확인
print(np.unique(y_train))
'''
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
46개중 찾아야한다. ! 떠올라야할 것 categorical entropy / softmax
'''
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (8982, 46) (2246, 46)


#@ 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation="softmax"))

#@ 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
            metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=30)

acc = model.evaluate(x_test, y_test)[1]
print("acc : ", acc)

'''
acc :  0.683882474899292
'''
