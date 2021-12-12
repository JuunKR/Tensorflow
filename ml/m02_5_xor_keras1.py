from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#. 단층 퍼셉트론 만들기
#1. 데이터
x_data = [[0,0], [0,1],[1,0], [1,1]]
y_data = [0,1,1,0]

#2. 모델
# model = LinearSVC()
# model = SVC() 
model = Sequential() 
model.add(Dense(1, input_dim=2, activation='sigmoid'))

#3 컴파일,  훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

import tensorflow as tf
#4. 평가, 예측
y_predict = model.predict(x_data)
# y_predict = tf.argmax(y_predict, axis=1) #.argmax 말고 다른거쓰라는데 뭘 써야함?
y_pred = np.round(y_predict, 0)


# print(x_data, "의 예측결과 : ", y_predict)
print(x_data, "의 예측결과 : ", y_pred)


results = model.evaluate(x_data, y_data)
print('model_loss : ', results[0])
print('acc : ', results[1])

# acc = accuracy_score(y_data, y_predict)
acc = accuracy_score(y_data, y_pred)


print("acc_score : ", acc)